use algo::prefetcher::{Prefetcher, PrefetcherSequenceFamily};
use algo::{Fetch, RelationRead};
use std::collections::{BinaryHeap, VecDeque};
use std::marker::PhantomData;

pub struct Candidates<'a, P, F, R>
where
    P: Prefetcher<'a>,
    P::Item: Fetch,
{
    beam: usize,
    front: Option<P>,
    heap: BinaryHeap<P::Item>,
    prefetch: F,
    _phantom: PhantomData<fn(&'a R) -> &'a R>,
}

impl<'a, P, F, R> Candidates<'a, P, F, R>
where
    P: Prefetcher<'a, R = R>,
    P::Item: Fetch + Ord,
    F: PrefetcherSequenceFamily<'a, R, P<VecDeque<P::Item>> = P>,
    R: RelationRead,
{
    pub fn new(beam: usize, prefetch: F) -> Self {
        assert_ne!(beam, 0);
        Self {
            beam,
            front: None,
            heap: BinaryHeap::new(),
            prefetch,
            _phantom: PhantomData,
        }
    }
    pub fn pop(&mut self) -> Option<(P::Item, Vec<R::ReadGuard<'a>>)> {
        if let Some(front) = self.front.as_mut()
            && let Some(item) = front.next()
        {
            return Some(item);
        }
        self.front = Some(
            self.prefetch.prefetch(
                (0..self.beam)
                    .flat_map(|_| self.heap.pop())
                    .collect::<VecDeque<_>>(),
            ),
        );
        if let Some(front) = self.front.as_mut()
            && let Some(item) = front.next()
        {
            return Some(item);
        }
        None
    }
    pub fn push(&mut self, item: P::Item) {
        self.heap.push(item);
    }
}
