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

use crate::closure_lifetime_binder::id_2;
use crate::linked_vec::LinkedVec;
use crate::operator::*;
use crate::tape::{by_directory, by_next};
use crate::tuples::*;
use crate::{Opaque, centroids, tape};
use always_equal::AlwaysEqual;
use distance::Distance;
use index::accessor::{DefaultWithDimension, LAccess};
use index::bump::Bump;
use index::fetch::BorrowedIter;
use index::packed::PackedRefMut4;
use index::prefetcher::{Prefetcher, PrefetcherHeapFamily, PrefetcherSequenceFamily};
use index::relation::{Page, RelationRead};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

type Extra1<'b> = &'b mut (u32, f32, u16, BorrowedIter<'b>);
type Result = (Reverse<Distance>, AlwaysEqual<NonZero<u64>>);

pub fn eager_default_search<'b, R: RelationRead, O: Operator, F, P>(
    index: &'b R,
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    partitions: Vec<u32>,
    recalls: Vec<f32>,
    target_number: u32,
    target_recall: f32,
    epsilon: f32,
    bump: &'b impl Bump,
    mut prefetch_h1_vectors: impl PrefetcherHeapFamily<'b, R>,
    mut prefetch_h0_tuples: impl PrefetcherSequenceFamily<'b, R>,
    mut rerank: impl FnMut(
        Vec<(
            (Reverse<Distance>, AlwaysEqual<()>),
            AlwaysEqual<PackedRefMut4<'b, (NonZero<u64>, u16, BorrowedIter<'b>)>>,
        )>,
    ) -> crate::rerank::Reranker<
        (),
        F,
        P,
        PackedRefMut4<'b, (NonZero<u64>, u16, BorrowedIter<'b>)>,
    >,
) -> Vec<Result>
where
    R::Page: Page<Opaque = Opaque>,
    F: FnMut(NonZero<u64>, P::Guards, u16) -> Option<Distance>,
    P: Prefetcher<
            'b,
            Item = (
                (Reverse<Distance>, AlwaysEqual<()>),
                AlwaysEqual<PackedRefMut4<'b, (NonZero<u64>, u16, BorrowedIter<'b>)>>,
            ),
        >,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dim = meta_tuple.dim();
    let is_residual = meta_tuple.is_residual();
    let height_of_root = meta_tuple.height_of_root();
    let cells = meta_tuple.cells().to_vec();
    assert_eq!(dim, vector.dim(), "unmatched dimensions");
    if height_of_root as usize != 1 + partitions.len() {
        panic!(
            "usage: need {} partitions, but {} partitions provided",
            height_of_root - 1,
            partitions.len()
        );
    }
    if height_of_root as usize != 1 + recalls.len() {
        panic!(
            "usage: need {} recalls, but {} recalls provided",
            height_of_root - 1,
            recalls.len()
        );
    }
    debug_assert_eq!(cells[(height_of_root - 1) as usize], 1);

    type State<O> = Vec<(
        Reverse<Distance>,
        AlwaysEqual<<O as Operator>::Vector>,
        AlwaysEqual<f32>,
        AlwaysEqual<u32>,
    )>;
    let mut state: State<O> = {
        let prefetch =
            BorrowedIter::from_slice(meta_tuple.centroid_prefetch(), |x| bump.alloc_slice(x));
        let head = meta_tuple.centroid_head();
        let (centroid, distance) = centroids::read::<R, O, _>(
            prefetch.map(|id| index.read(id)),
            head,
            (
                CloneAccessor::default_with_dimension(dim),
                LAccess::new(
                    O::Vector::unpack(vector),
                    O::DistanceAccessor::default_with_dimension(dim),
                ),
            ),
        );
        let norm = meta_tuple.centroid_norm();
        let first = meta_tuple.first();
        vec![(
            Reverse(distance),
            AlwaysEqual(centroid),
            AlwaysEqual(norm),
            AlwaysEqual(first),
        )]
    };

    drop(meta_guard);
    let lut = O::Vector::preprocess(vector);

    let mut step = |state: State<O>, partition: u32, recall: f32| -> State<O> {
        let mut cache = Cache4::new(partition as usize);
        let mut oracle = Oracle::new(dim, state.len(), |i| {
            O::Vector::height(
                state[0].1.0.as_borrowed(),
                state[i].1.0.as_borrowed(),
                vector,
            )
        });
        for (i, &(Reverse(dis_f), AlwaysEqual(_), AlwaysEqual(norm), AlwaysEqual(first))) in
            state.iter().enumerate()
        {
            let mut results = LinkedVec::<(_, AlwaysEqual<Extra1<'b>>)>::new();
            tape::read_h1_tape::<R, _, _>(
                by_next(index, first),
                || O::block_access(&lut.0, is_residual, dis_f.to_f32(), norm),
                |(rough, err), head, norm, first, prefetch| {
                    let lowerbound = Distance::from_f32(rough - err * epsilon);
                    results.push((
                        Reverse(lowerbound),
                        AlwaysEqual(bump.alloc((
                            first,
                            norm,
                            head,
                            BorrowedIter::from_slice(prefetch, |x| bump.alloc_slice(x)),
                        ))),
                    ));
                },
            );
            let mut heap = prefetch_h1_vectors.prefetch(results.into_vec());
            while let Some(((Reverse(_), AlwaysEqual(&mut (first, norm, head, ..))), prefetch)) =
                heap.next_if(|&(Reverse(l), _)| l < cache.maximum())
            {
                let (centroid, distance) = centroids::read::<R, O, _>(
                    prefetch,
                    head,
                    (
                        CloneAccessor::default_with_dimension(dim),
                        LAccess::new(
                            O::Vector::unpack(vector),
                            O::DistanceAccessor::default_with_dimension(dim),
                        ),
                    ),
                );
                cache.push((
                    distance,
                    AlwaysEqual(centroid),
                    AlwaysEqual(norm),
                    AlwaysEqual(first),
                ));
            }
            oracle.update(Distance::from_f32(cache.maximum().to_f32().sqrt()));
            if oracle.query(i) > recall {
                break;
            }
        }
        let mut results = cache.into_iter().collect::<Vec<_>>();
        results.sort_unstable_by_key(|&(Reverse(distance), ..)| distance);
        results
    };

    for i in 1..height_of_root {
        state = step(state, partitions[i as usize - 1], recalls[i as usize - 1]);
    }

    let mut cache = Cache2::new(target_number as usize);
    let mut oracle = Oracle::new(dim, state.len(), |i| {
        O::Vector::height(
            state[0].1.0.as_borrowed(),
            state[i].1.0.as_borrowed(),
            vector,
        )
    });
    for (i, &(Reverse(dis_f), AlwaysEqual(_), AlwaysEqual(norm), AlwaysEqual(first))) in
        state.iter().enumerate()
    {
        let mut results = LinkedVec::<(_, AlwaysEqual<_>)>::new();
        let jump_guard = index.read(first);
        let jump_bytes = jump_guard.get(1).expect("data corruption");
        let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
        let mut callback = id_2(|(rough, err), head, payload, prefetch| {
            let lowerbound = Distance::from_f32(rough - err * epsilon);
            results.push((
                (Reverse(lowerbound), AlwaysEqual(())),
                AlwaysEqual(PackedRefMut4(bump.alloc((
                    payload,
                    head,
                    BorrowedIter::from_slice(prefetch, |x| bump.alloc_slice(x)),
                )))),
            ));
        });
        if prefetch_h0_tuples.is_not_plain() {
            let directory =
                tape::read_directory_tape::<R>(by_next(index, jump_tuple.directory_first()));
            tape::read_frozen_tape::<R, _, _>(
                by_directory(&mut prefetch_h0_tuples, directory),
                || O::block_access(&lut.0, is_residual, dis_f.to_f32(), norm),
                &mut callback,
            );
        } else {
            tape::read_frozen_tape::<R, _, _>(
                by_next(index, jump_tuple.frozen_first()),
                || O::block_access(&lut.0, is_residual, dis_f.to_f32(), norm),
                &mut callback,
            );
        }
        tape::read_appendable_tape::<R, _>(
            by_next(index, jump_tuple.appendable_first()),
            O::binary_access(&lut.1, is_residual, dis_f.to_f32(), norm),
            &mut callback,
        );
        let mut reranker = rerank(results.into_vec());
        reranker.termination = cache.maximum();
        while let Some((distance, payload)) = reranker.next() {
            cache.push((distance, AlwaysEqual(payload)));
            reranker.termination = cache.maximum();
        }
        oracle.update(Distance::from_f32(cache.maximum().to_f32().sqrt()));
        if oracle.query(i) > target_recall {
            break;
        }
    }
    let mut results = cache.into_iter().collect::<Vec<_>>();
    results.sort_unstable_by_key(|&(Reverse(distance), ..)| distance);
    results
}

struct Oracle {
    dim: u32,
    heights: Vec<f32>,
    contributions: Vec<f32>,
    rho: Distance,
}

impl Oracle {
    fn new<F: Fn(usize) -> f32>(dim: u32, n: usize, f: F) -> Self {
        let heights = {
            let mut heights = Vec::with_capacity(n - 1);
            for i in 1..n {
                heights.push(f(i));
            }
            heights
        };
        let mut result = Self {
            dim,
            heights,
            contributions: vec![0.0; n],
            rho: Distance::INFINITY,
        };
        result.maintain();
        result
    }
    fn update(&mut self, rho: Distance) {
        if rho < self.rho {
            self.rho = rho;
            self.maintain();
        }
    }
    fn maintain(&mut self) {
        use simd::Floating;
        let rho = self.rho.to_f32();
        for (&height, contribution) in std::iter::zip(&self.heights, &mut self.contributions[1..]) {
            *contribution = if height < rho {
                let x = 1.0 - (height / rho) * (height / rho);
                0.5 * statrs::function::beta::beta_reg(0.5 * self.dim as f64 + 0.5, 0.5, x as f64)
                    as f32
            } else {
                0.0
            };
        }
        let l = Floating::reduce_sum_of_x(&self.contributions[1..]);
        if l > 1.19209290e-07 {
            Floating::vector_mul_scalar_inplace(&mut self.contributions[1..], 1.0 / l);
        }
        let c = self.contributions[1..].iter().map(|&x| 1.0 - x).product();
        self.contributions[0] = c;
        Floating::vector_mul_scalar_inplace(&mut self.contributions[1..], 1.0 - c);
    }
    fn query(&self, i: usize) -> f32 {
        self.contributions.iter().take(i + 1).sum()
    }
}

struct Cache2<A> {
    size: usize,
    heap: BinaryHeap<(Distance, AlwaysEqual<A>)>,
    maximum: Distance,
}

impl<A> Cache2<A> {
    fn new(size: usize) -> Self {
        assert!(size > 0, "size must be positive integer");
        Self {
            size,
            heap: BinaryHeap::with_capacity(size + 1),
            maximum: Distance::INFINITY,
        }
    }
    fn push(&mut self, item: (Distance, AlwaysEqual<A>)) {
        if item.0 < self.maximum || self.heap.len() < self.size {
            self.heap.push(item);
            if self.heap.len() > self.size {
                self.heap.pop();
            }
            if self.heap.len() == self.size {
                self.maximum = self
                    .heap
                    .peek()
                    .map(|&(distance, ..)| distance)
                    .unwrap_or_default();
            }
        }
    }
    fn maximum(&self) -> Distance {
        self.maximum
    }
    fn into_iter(self) -> impl Iterator<Item = (Reverse<Distance>, AlwaysEqual<A>)> {
        self.heap.into_iter().map(|(d, a)| (Reverse(d), a))
    }
}

struct Cache4<A, B, C> {
    size: usize,
    heap: BinaryHeap<(Distance, AlwaysEqual<A>, AlwaysEqual<B>, AlwaysEqual<C>)>,
    maximum: Distance,
}

impl<A, B, C> Cache4<A, B, C> {
    fn new(size: usize) -> Self {
        assert!(size > 0, "size must be positive integer");
        Self {
            size,
            heap: BinaryHeap::with_capacity(size + 1),
            maximum: Distance::INFINITY,
        }
    }
    fn push(&mut self, item: (Distance, AlwaysEqual<A>, AlwaysEqual<B>, AlwaysEqual<C>)) {
        if item.0 < self.maximum || self.heap.len() < self.size {
            self.heap.push(item);
            if self.heap.len() > self.size {
                self.heap.pop();
            }
            if self.heap.len() == self.size {
                self.maximum = self
                    .heap
                    .peek()
                    .map(|&(distance, ..)| distance)
                    .unwrap_or_default();
            }
        }
    }
    fn maximum(&self) -> Distance {
        self.maximum
    }
    fn into_iter(
        self,
    ) -> impl Iterator<
        Item = (
            Reverse<Distance>,
            AlwaysEqual<A>,
            AlwaysEqual<B>,
            AlwaysEqual<C>,
        ),
    > {
        self.heap
            .into_iter()
            .map(|(d, a, b, c)| (Reverse(d), a, b, c))
    }
}
