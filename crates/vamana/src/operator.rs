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

use algo::accessor::Accessor2;
use distance::Distance;
use half::f16;
use simd::Floating;
use std::fmt::Debug;
use std::marker::PhantomData;
use vector::vect::VectOwned;
use vector::{VectorBorrowed, VectorOwned};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::tuples::Neighbour;

pub trait Vector: VectorOwned {
    type Element: Debug + Copy + FromBytes + IntoBytes + Immutable + KnownLayout;
    type Metadata: Debug + Copy + FromBytes + IntoBytes + Immutable + KnownLayout;

    fn code(vector: Self::Borrowed<'_>) -> rabitq::b1::Code;
    fn preprocess(vector: Self::Borrowed<'_>) -> rabitq::b1::binary::BinaryLut;

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata);
    fn split(vector: Self::Borrowed<'_>, m: usize) -> (Vec<&[Self::Element]>, Self::Metadata);
}

impl Vector for VectOwned<f32> {
    type Metadata = ();

    type Element = f32;

    fn code(vector: Self::Borrowed<'_>) -> rabitq::b1::Code {
        rabitq::b1::code(vector.slice())
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> rabitq::b1::binary::BinaryLut {
        rabitq::b1::binary::preprocess(vector.slice())
    }

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (vector.slice(), ())
    }

    fn split(vector: Self::Borrowed<'_>, m: usize) -> (Vec<&[f32]>, ()) {
        let mut n = vector.dims() as usize;
        let mut mids = Vec::new();
        loop {
            if size_of::<Neighbour>() * m + size_of::<f32>() * n <= 8000 {
                mids.push(n);
                break;
            }
            if n == 0 {
                panic!("internal: too many neighbours")
            }
            let k = std::cmp::min(n, 8000 / size_of::<f32>());
            mids.push(k);
            n -= k;
        }
        mids.reverse();
        let mut slice = vector.slice();
        let mut result = Vec::new();
        for mid in mids {
            let r;
            (r, slice) = slice.split_at(mid);
            result.push(r);
        }
        (result, ())
    }
}

impl Vector for VectOwned<f16> {
    type Metadata = ();

    type Element = f16;

    fn code(vector: Self::Borrowed<'_>) -> rabitq::b1::Code {
        rabitq::b1::code(&f16::vector_to_f32(vector.slice()))
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> rabitq::b1::binary::BinaryLut {
        rabitq::b1::binary::preprocess(&f16::vector_to_f32(vector.slice()))
    }

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (vector.slice(), ())
    }

    fn split(vector: Self::Borrowed<'_>, m: usize) -> (Vec<&[f16]>, ()) {
        let mut n = vector.dims() as usize;
        let mut mids = Vec::new();
        loop {
            if size_of::<Neighbour>() * m + size_of::<f16>() * n <= 8000 {
                mids.push(n);
                break;
            }
            if n == 0 {
                panic!("internal: too many neighbours")
            }
            let k = std::cmp::min(n, 8000 / size_of::<f16>());
            mids.push(k);
            n -= k;
        }
        mids.reverse();
        let mut slice = vector.slice();
        let mut result = Vec::new();
        for mid in mids {
            let r;
            (r, slice) = slice.split_at(mid);
            result.push(r);
        }
        (result, ())
    }
}

pub type L2 = algo::accessor::L2;

pub trait Operator: 'static + Debug + Copy {
    type Vector: Vector;

    fn process(code: ([f32; 3], &[u64]), lut: &rabitq::b1::binary::BinaryLut) -> Distance;

    type DistanceAccessor: Default
        + Accessor2<
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Metadata,
            <Self::Vector as Vector>::Metadata,
            Output = Distance,
        >;
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

    fn process(code: ([f32; 3], &[u64]), lut: &rabitq::b1::binary::BinaryLut) -> Distance {
        use rabitq::b1::CodeMetadata;
        let value = rabitq::b1::binary::accumulate(code.1, &lut.1);
        let (distance,) =
            rabitq::b1::binary::half_process_l2(value, CodeMetadata::from_array(code.0), lut.0);
        Distance::from_f32(distance)
    }

    type DistanceAccessor = algo::accessor::DistanceAccessor<VectOwned<f32>, L2>;
}

impl Operator for Op<VectOwned<f16>, L2> {
    type Vector = VectOwned<f16>;

    fn process(code: ([f32; 3], &[u64]), lut: &rabitq::b1::binary::BinaryLut) -> Distance {
        use rabitq::b1::CodeMetadata;
        let value = rabitq::b1::binary::accumulate(code.1, &lut.1);
        let (distance,) =
            rabitq::b1::binary::half_process_l2(value, CodeMetadata::from_array(code.0), lut.0);
        Distance::from_f32(distance)
    }

    type DistanceAccessor = algo::accessor::DistanceAccessor<VectOwned<f16>, L2>;
}
