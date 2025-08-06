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

use crate::closure_lifetime_binder::id_4;
use crate::operator::*;
use crate::tuples::{MetaTuple, WithReader};
use crate::{Page, vectors};
use algo::accessor::{Accessor2, LTryAccess};
use algo::{RelationRead, RerankMethod};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vector::VectorOwned;

type Extra<'b> = &'b mut (NonZero<u64>, u16, &'b mut [u32]);

type Result = (Reverse<Distance>, AlwaysEqual<NonZero<u64>>);

pub fn how(index: &impl RelationRead) -> RerankMethod {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let rerank_in_heap = meta_tuple.rerank_in_heap();
    if rerank_in_heap {
        RerankMethod::Heap
    } else {
        RerankMethod::Index
    }
}

pub struct Reranker<'b, T, F> {
    prefetcher: BinaryHeap<((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<Extra<'b>>)>,
    cache: BinaryHeap<Result>,
    f: F,
}

impl<'b, T, F> Iterator for Reranker<'b, T, F>
where
    F: FnMut(NonZero<u64>, &[u32], u16) -> Option<Distance>,
{
    type Item = (Distance, NonZero<u64>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((
            (Reverse(_), AlwaysEqual(_)),
            AlwaysEqual(&mut (payload, head, ref list, ..)),
        )) = pop_if(&mut self.prefetcher, |((d, _), ..)| {
            Some(*d) > self.cache.peek().map(|(d, ..)| *d)
        }) {
            if let Some(distance) = (self.f)(payload, list, head) {
                self.cache.push((Reverse(distance), AlwaysEqual(payload)));
            };
        }
        let (Reverse(distance), AlwaysEqual(payload)) = self.cache.pop()?;
        Some((distance, payload))
    }
}

impl<'b, T, F> Reranker<'b, T, F> {
    pub fn finish(
        self,
    ) -> (
        BinaryHeap<(
            (Reverse<Distance>, AlwaysEqual<T>),
            AlwaysEqual<&'b mut (NonZero<u64>, u16, &'b mut [u32])>,
        )>,
        impl Iterator<Item = Result>,
    ) {
        (self.prefetcher, self.cache.into_iter())
    }
}

pub fn rerank_index<'b, R: RelationRead, O: Operator, T>(
    index: &R,
    vector: O::Vector,
    prefetcher: BinaryHeap<((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<Extra<'b>>)>,
) -> Reranker<'b, T, impl FnMut(NonZero<u64>, &[u32], u16) -> Option<Distance>> {
    Reranker {
        prefetcher,
        cache: BinaryHeap::new(),
        f: id_4::<_, _, _, _>(move |payload, list, head| {
            vectors::read_for_h0_tuple::<R, O, _>(
                list.into_iter().map(|&id| index.read(id)),
                head,
                payload,
                LTryAccess::new(
                    O::Vector::unpack(vector.as_borrowed()),
                    O::DistanceAccessor::default(),
                ),
            )
        }),
    }
}

pub fn rerank_heap<'b, R: RelationRead, O: Operator, T>(
    vector: O::Vector,
    prefetcher: BinaryHeap<((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<Extra<'b>>)>,
    mut fetch: impl FnMut(NonZero<u64>) -> Option<O::Vector> + 'b,
) -> Reranker<'b, T, impl FnMut(NonZero<u64>, &[u32], u16) -> Option<Distance>> {
    Reranker {
        prefetcher,
        cache: BinaryHeap::new(),
        f: id_4::<_, _, _, _>(move |payload, _, _| {
            let unpack = O::Vector::unpack(vector.as_borrowed());
            let vector = fetch(payload)?;
            let vector = O::Vector::unpack(vector.as_borrowed());
            let mut accessor = O::DistanceAccessor::default();
            accessor.push(unpack.0, vector.0);
            let distance = accessor.finish(unpack.1, vector.1);
            Some(distance)
        }),
    }
}

fn pop_if<T: Ord>(
    heap: &mut BinaryHeap<T>,
    mut predicate: impl FnMut(&mut T) -> bool,
) -> Option<T> {
    use std::collections::binary_heap::PeekMut;
    let mut peek = heap.peek_mut()?;
    if predicate(&mut peek) {
        Some(PeekMut::pop(peek))
    } else {
        None
    }
}
