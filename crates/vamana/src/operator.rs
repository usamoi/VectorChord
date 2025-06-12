// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025 TensorChord Inc.

use distance::Distance;
use half::f16;
use rabitq8::CodeMetadata;
use simd::Floating;
use std::fmt::Debug;
use std::marker::PhantomData;
use vector::vect::{VectBorrowed, VectOwned};
use vector::{VectorOwned};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub trait Accessor2<E0, E1, M0, M1> {
    type Output;
    fn push(&mut self, input: &[E0], target: &[E1]);
    fn finish(self, input: M0, target: M1) -> Self::Output;
}

impl<E0, E1, M0: Copy, M1: Copy> Accessor2<E0, E1, M0, M1> for () {
    type Output = ();

    fn push(&mut self, _: &[E0], _: &[E1]) {}

    fn finish(self, _: M0, _: M1) -> Self::Output {}
}

impl<E0, E1, M0: Copy, M1: Copy, A: Accessor2<E0, E1, M0, M1>> Accessor2<E0, E1, M0, M1> for (A,) {
    type Output = (A::Output,);

    fn push(&mut self, input: &[E0], target: &[E1]) {
        self.0.push(input, target);
    }

    fn finish(self, input: M0, target: M1) -> Self::Output {
        (self.0.finish(input, target),)
    }
}

impl<E0, E1, M0: Copy, M1: Copy, A: Accessor2<E0, E1, M0, M1>, B: Accessor2<E0, E1, M0, M1>>
    Accessor2<E0, E1, M0, M1> for (A, B)
{
    type Output = (A::Output, B::Output);

    fn push(&mut self, input: &[E0], target: &[E1]) {
        self.0.push(input, target);
        self.1.push(input, target);
    }

    fn finish(self, input: M0, target: M1) -> Self::Output {
        (self.0.finish(input, target), self.1.finish(input, target))
    }
}

pub trait Accessor1<E, M> {
    type Output;
    fn push(&mut self, input: &[E]);
    fn finish(self, input: M) -> Self::Output;
}

impl<E, M: Copy> Accessor1<E, M> for () {
    type Output = ();

    fn push(&mut self, _: &[E]) {}

    fn finish(self, _: M) -> Self::Output {}
}

impl<E, M: Copy, A> Accessor1<E, M> for (A,)
where
    A: Accessor1<E, M>,
{
    type Output = (A::Output,);

    fn push(&mut self, input: &[E]) {
        self.0.push(input);
    }

    fn finish(self, input: M) -> Self::Output {
        (self.0.finish(input),)
    }
}

impl<E, M: Copy, A, B> Accessor1<E, M> for (A, B)
where
    A: Accessor1<E, M>,
    B: Accessor1<E, M>,
{
    type Output = (A::Output, B::Output);

    fn push(&mut self, input: &[E]) {
        self.0.push(input);
        self.1.push(input);
    }

    fn finish(self, input: M) -> Self::Output {
        (self.0.finish(input), self.1.finish(input))
    }
}

#[derive(Debug)]
pub struct DistanceAccessor<V, D>(f32, PhantomData<fn(V) -> V>, PhantomData<fn(D) -> D>);

impl<V, D> Default for DistanceAccessor<V, D> {
    fn default() -> Self {
        Self(0.0, PhantomData, PhantomData)
    }
}

impl Accessor2<f32, f32, (), ()> for DistanceAccessor<VectOwned<f32>, L2> {
    type Output = Distance;

    fn push(&mut self, target: &[f32], input: &[f32]) {
        self.0 += f32::reduce_sum_of_d2(target, input)
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        Distance::from_f32(self.0)
    }
}

impl Accessor2<f32, f32, (), ()> for DistanceAccessor<VectOwned<f32>, Dot> {
    type Output = Distance;

    fn push(&mut self, target: &[f32], input: &[f32]) {
        self.0 += f32::reduce_sum_of_xy(target, input)
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        Distance::from_f32(-self.0)
    }
}

impl Accessor2<f16, f16, (), ()> for DistanceAccessor<VectOwned<f16>, L2> {
    type Output = Distance;

    fn push(&mut self, target: &[f16], input: &[f16]) {
        self.0 += f16::reduce_sum_of_d2(target, input)
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        Distance::from_f32(self.0)
    }
}

impl Accessor2<f16, f16, (), ()> for DistanceAccessor<VectOwned<f16>, Dot> {
    type Output = Distance;

    fn push(&mut self, target: &[f16], input: &[f16]) {
        self.0 += f16::reduce_sum_of_xy(target, input)
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        Distance::from_f32(-self.0)
    }
}

#[derive(Debug, Clone)]
pub struct CloneAccessor<V: Vector>(Vec<V::Element>);

impl<V: Vector> Default for CloneAccessor<V> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<V: Vector> Accessor1<V::Element, V::Metadata> for CloneAccessor<V> {
    type Output = V;

    fn push(&mut self, input: &[V::Element]) {
        self.0.extend(input);
    }

    fn finish(self, metadata: V::Metadata) -> Self::Output {
        V::pack(self.0, metadata)
    }
}

pub trait Vector: VectorOwned {
    type Element: Debug + Copy + FromBytes + IntoBytes + Immutable + KnownLayout;

    type Metadata: Debug + Copy + FromBytes + IntoBytes + Immutable + KnownLayout;

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata);

    fn pack(elements: Vec<Self::Element>, metadata: Self::Metadata) -> Self;

    fn binary_preprocess(vector: Self::Borrowed<'_>) -> rabitq8::Code;

    fn code(vector: Self::Borrowed<'_>) -> rabitq8::Code;
}

impl Vector for VectOwned<f32> {
    type Metadata = ();

    type Element = f32;

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (vector.slice(), ())
    }

    fn pack(elements: Vec<Self::Element>, (): Self::Metadata) -> Self {
        VectOwned::new(elements)
    }

    fn binary_preprocess(vector: Self::Borrowed<'_>) -> rabitq8::Code {
        rabitq8::code(vector.slice())
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq8::Code {
        rabitq8::code(vector.slice())
    }
}

impl Vector for VectOwned<f16> {
    type Metadata = ();

    type Element = f16;

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (vector.slice(), ())
    }

    fn pack(elements: Vec<Self::Element>, (): Self::Metadata) -> Self {
        VectOwned::new(elements)
    }

    fn binary_preprocess(vector: Self::Borrowed<'_>) -> rabitq8::Code {
        rabitq8::code(&f16::vector_to_f32(vector.slice()))
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq8::Code {
        rabitq8::code(&f16::vector_to_f32(vector.slice()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct L2;

#[derive(Debug, Clone, Copy)]
pub struct Dot;

pub trait Operator: 'static + Debug + Copy {
    type Vector: Vector;

    type DistanceAccessor: Default
        + Accessor2<
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Metadata,
            <Self::Vector as Vector>::Metadata,
            Output = Distance,
        >;

    fn binary_access<F: Call<u32, CodeMetadata, CodeMetadata>>(
        lut: &rabitq8::Code,
        f: F,
    ) -> impl FnMut([f32; 3], &[u8]) -> F::Output;

    fn binary_process(n: u32, value: u32, code: CodeMetadata, lut: CodeMetadata) -> f32;

    fn build(vector: <Self::Vector as VectorOwned>::Borrowed<'_>) -> rabitq8::Code;
}

#[derive(Debug)]
pub struct Op<V, D>(PhantomData<fn(V) -> V>, PhantomData<fn(D) -> D>);

impl<V, D> Clone for Op<V, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V, D> Copy for Op<V, D> {}

impl Operator for Op<VectOwned<f32>, L2> {
    type Vector = VectOwned<f32>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f32>, L2>;

    fn binary_access<F: Call<u32, CodeMetadata, CodeMetadata>>(
        lut: &rabitq8::Code,
        mut f: F,
    ) -> impl FnMut([f32; 3], &[u8]) -> F::Output {
        move |metadata: [f32; 3], elements: &[u8]| {
            let value = simd::u8::reduce_sum_of_x_as_u32_y_as_u32(elements, &lut.1);
            f.call(
                value,
                CodeMetadata {
                    dis_u_2: metadata[0],
                    norm_of_lattice: metadata[1],
                    sum_of_code: metadata[2],
                },
                lut.0,
            )
        }
    }

    fn binary_process(n: u32, value: u32, code: CodeMetadata, lut: CodeMetadata) -> f32 {
        rabitq8::half_process_l2(n, value, code, lut)
    }

    fn build(vector: VectBorrowed<'_, f32>) -> rabitq8::Code {
        Self::Vector::code(vector)
    }
}

impl Operator for Op<VectOwned<f32>, Dot> {
    type Vector = VectOwned<f32>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f32>, Dot>;

    fn binary_access<F: Call<u32, CodeMetadata, CodeMetadata>>(
        lut: &rabitq8::Code,
        mut f: F,
    ) -> impl FnMut([f32; 3], &[u8]) -> F::Output {
        move |metadata: [f32; 3], elements: &[u8]| {
            let value = simd::u8::reduce_sum_of_x_as_u32_y_as_u32(elements, &lut.1);
            f.call(
                value,
                CodeMetadata {
                    dis_u_2: metadata[0],
                    norm_of_lattice: metadata[1],
                    sum_of_code: metadata[2],
                },
                lut.0,
            )
        }
    }

    fn binary_process(n: u32, value: u32, code: CodeMetadata, lut: CodeMetadata) -> f32 {
        rabitq8::half_process_dot(n, value, code, lut)
    }

    fn build(vector: VectBorrowed<'_, f32>) -> rabitq8::Code {
        Self::Vector::code(vector)
    }
}

impl Operator for Op<VectOwned<f16>, L2> {
    type Vector = VectOwned<f16>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f16>, L2>;

    fn binary_access<F: Call<u32, CodeMetadata, CodeMetadata>>(
        lut: &rabitq8::Code,
        mut f: F,
    ) -> impl FnMut([f32; 3], &[u8]) -> F::Output {
        move |metadata: [f32; 3], elements: &[u8]| {
            let value = simd::u8::reduce_sum_of_x_as_u32_y_as_u32(elements, &lut.1);
            f.call(
                value,
                CodeMetadata {
                    dis_u_2: metadata[0],
                    norm_of_lattice: metadata[1],
                    sum_of_code: metadata[2],
                },
                lut.0,
            )
        }
    }

    fn binary_process(n: u32, value: u32, code: CodeMetadata, lut: CodeMetadata) -> f32 {
        rabitq8::half_process_l2(n, value, code, lut)
    }

    fn build(vector: VectBorrowed<'_, f16>) -> rabitq8::Code {
        Self::Vector::code(vector)
    }
}

impl Operator for Op<VectOwned<f16>, Dot> {
    type Vector = VectOwned<f16>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f16>, Dot>;

    fn binary_access<F: Call<u32, CodeMetadata, CodeMetadata>>(
        lut: &rabitq8::Code,
        mut f: F,
    ) -> impl FnMut([f32; 3], &[u8]) -> F::Output {
        move |metadata: [f32; 3], elements: &[u8]| {
            let value = simd::u8::reduce_sum_of_x_as_u32_y_as_u32(elements, &lut.1);
            f.call(
                value,
                CodeMetadata {
                    dis_u_2: metadata[0],
                    norm_of_lattice: metadata[1],
                    sum_of_code: metadata[2],
                },
                lut.0,
            )
        }
    }

    fn binary_process(n: u32, value: u32, code: CodeMetadata, lut: CodeMetadata) -> f32 {
        rabitq8::half_process_dot(n, value, code, lut)
    }

    fn build(vector: VectBorrowed<'_, f16>) -> rabitq8::Code {
        Self::Vector::code(vector)
    }
}

pub trait Call<A, B, C> {
    type Output;

    fn call(&mut self, a: A, b: B, c: C) -> Self::Output;
}

impl<A, B, C, F: Fn(A, B, C) -> R, R> Call<A, B, C> for F {
    type Output = R;

    fn call(&mut self, a: A, b: B, c: C) -> R {
        (self)(a, b, c)
    }
}
