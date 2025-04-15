use crate::operator::*;
use crate::tuples::{MetaTuple, WithReader};
use crate::window_heap::WindowHeap;
use crate::{IndexPointer, Page, RelationRead, RerankMethod, vectors};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vector::VectorOwned;

type Result<'b, T> = (
    Reverse<Distance>,
    AlwaysEqual<T>,
    AlwaysEqual<NonZero<u64>>,
    AlwaysEqual<IndexPointer>,
    AlwaysEqual<&'b mut [u32]>,
);

type Rerank = (Reverse<Distance>, AlwaysEqual<NonZero<u64>>);

pub fn how(index: impl RelationRead) -> RerankMethod {
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

pub struct Reranker<'b, T, F, P> {
    heap: WindowHeap<Result<'b, T>, P>,
    cache: BinaryHeap<(Reverse<Distance>, AlwaysEqual<NonZero<u64>>)>,
    f: F,
}

impl<'b, T, F: FnMut(IndexPointer, NonZero<u64>) -> Option<Distance>, P: FnMut(&mut Result<'b, T>)>
    Iterator for Reranker<'b, T, F, P>
{
    type Item = (Distance, NonZero<u64>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((_, _, AlwaysEqual(payload), AlwaysEqual(mean), _)) = self
            .heap
            .pop_if(|(d, ..)| Some(*d) > self.cache.peek().map(|(d, ..)| *d))
        {
            if let Some(dis_u) = (self.f)(mean, payload) {
                self.cache.push((Reverse(dis_u), AlwaysEqual(payload)));
            };
        }
        let (Reverse(dis_u), AlwaysEqual(pay_u)) = self.cache.pop()?;
        Some((dis_u, pay_u))
    }
}

impl<'b, T, F, P> Reranker<'b, T, F, P> {
    pub fn finish(
        self,
    ) -> (
        impl Iterator<Item = Result<'b, T>>,
        impl Iterator<Item = Rerank>,
    ) {
        (self.heap.into_iter(), self.cache.into_iter())
    }
}

pub fn rerank_index<O: Operator, T>(
    index: impl RelationRead,
    vector: O::Vector,
    results: Vec<Result<T>>,
) -> Reranker<
    T,
    impl FnMut(IndexPointer, NonZero<u64>) -> Option<Distance>,
    impl FnMut(&mut Result<T>),
> {
    Reranker {
        heap: WindowHeap::new(results, {
            let index = index.clone();
            move |(_, _, _, _, AlwaysEqual(prefetch)): &mut Result<T>| {
                for id in prefetch.iter().copied() {
                    index.prefetch(id);
                }
            }
        }),
        cache: BinaryHeap::<(Reverse<Distance>, _)>::new(),
        f: {
            let index = index.clone();
            move |mean, pay_u| {
                vectors::read_for_h0_tuple::<O, _>(
                    index.clone(),
                    mean,
                    pay_u,
                    LTryAccess::new(
                        O::Vector::unpack(vector.as_borrowed()),
                        O::DistanceAccessor::default(),
                    ),
                )
            }
        },
    }
}

pub fn rerank_heap<O: Operator, T>(
    vector: O::Vector,
    results: Vec<Result<T>>,
    mut fetch: impl FnMut(NonZero<u64>) -> Option<O::Vector>,
) -> Reranker<
    T,
    impl FnMut(IndexPointer, NonZero<u64>) -> Option<Distance>,
    impl FnMut(&mut Result<T>),
> {
    Reranker {
        heap: WindowHeap::new(results, |_: &mut Result<T>| ()),
        cache: BinaryHeap::<(Reverse<Distance>, _)>::new(),
        f: move |_: IndexPointer, pay_u| {
            let vector = O::Vector::unpack(vector.as_borrowed());
            let vec_u = fetch(pay_u)?;
            let vec_u = O::Vector::unpack(vec_u.as_borrowed());
            let mut accessor = O::DistanceAccessor::default();
            accessor.push(vector.0, vec_u.0);
            let dis_u = accessor.finish(vector.1, vec_u.1);
            Some(dis_u)
        },
    }
}
