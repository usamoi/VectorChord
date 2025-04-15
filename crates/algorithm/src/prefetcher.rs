use crate::{ReadStream, RelationPrefetch, RelationRead, RelationReadStream};
use always_equal::AlwaysEqual;
use std::collections::{BinaryHeap, VecDeque, binary_heap, vec_deque};
use std::iter::Chain;

pub trait Prefetcher<'b>: IntoIterator<Item = (Self::T, AlwaysEqual<&'b mut [u32]>)> {
    type R: RelationRead;
    type T;
    fn pop_if<'s>(
        &'s mut self,
        predicate: impl FnOnce(&mut Self::Item) -> bool,
    ) -> Option<(Self::T, Vec<<Self::R as RelationRead>::ReadGuard<'s>>)>;
}

pub struct PlainPrefetcher<'b, R, T> {
    relation: R,
    heap: BinaryHeap<(T, AlwaysEqual<&'b mut [u32]>)>,
}

impl<'b, R, T: Ord> PlainPrefetcher<'b, R, T> {
    pub fn new(relation: R, vec: Vec<(T, AlwaysEqual<&'b mut [u32]>)>) -> Self {
        Self {
            relation,
            heap: BinaryHeap::from(vec),
        }
    }
}

impl<'b, R, T> IntoIterator for PlainPrefetcher<'b, R, T> {
    type Item = (T, AlwaysEqual<&'b mut [u32]>);

    type IntoIter = binary_heap::IntoIter<(T, AlwaysEqual<&'b mut [u32]>)>;

    fn into_iter(self) -> Self::IntoIter {
        self.heap.into_iter()
    }
}

impl<'b, R: RelationRead, T: Ord> Prefetcher<'b> for PlainPrefetcher<'b, R, T> {
    type R = R;
    type T = T;
    fn pop_if<'s>(
        &'s mut self,
        predicate: impl FnOnce(&mut Self::Item) -> bool,
    ) -> Option<(T, Vec<R::ReadGuard<'s>>)> {
        let (item, AlwaysEqual(list)) = heap_pop_if(&mut self.heap, predicate)?;
        let list = list
            .into_iter()
            .map(|&mut id| self.relation.read(id))
            .collect();
        Some((item, list))
    }
}

pub struct SimplePrefetcher<'b, R, T> {
    relation: R,
    window: VecDeque<(T, AlwaysEqual<&'b mut [u32]>)>,
    heap: BinaryHeap<(T, AlwaysEqual<&'b mut [u32]>)>,
}

impl<'b, R, T: Ord> SimplePrefetcher<'b, R, T> {
    pub fn new(relation: R, vec: Vec<(T, AlwaysEqual<&'b mut [u32]>)>) -> Self {
        Self {
            relation,
            window: VecDeque::new(),
            heap: BinaryHeap::from(vec),
        }
    }
}

impl<'b, R, T> IntoIterator for SimplePrefetcher<'b, R, T> {
    type Item = (T, AlwaysEqual<&'b mut [u32]>);

    type IntoIter = Chain<
        vec_deque::IntoIter<(T, AlwaysEqual<&'b mut [u32]>)>,
        binary_heap::IntoIter<(T, AlwaysEqual<&'b mut [u32]>)>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.window.into_iter().chain(self.heap)
    }
}

impl<'b, R: RelationRead + RelationPrefetch, T: Ord> Prefetcher<'b> for SimplePrefetcher<'b, R, T> {
    type R = R;
    type T = T;
    fn pop_if<'s>(
        &'s mut self,
        predicate: impl FnOnce(&mut Self::Item) -> bool,
    ) -> Option<(T, Vec<R::ReadGuard<'s>>)> {
        while self.window.len() < 32
            && let Some(e) = self.heap.pop()
        {
            let (_, AlwaysEqual(ref list)) = e;
            for id in list.iter().copied() {
                self.relation.prefetch(id);
            }
            self.window.push_back(e);
        }
        let (item, AlwaysEqual(list)) = vec_deque_pop_front_if(&mut self.window, predicate)?;
        let list = list
            .into_iter()
            .map(|&mut id| self.relation.read(id))
            .collect();
        Some((item, list))
    }
}

pub struct StreamPrefetcherHeap<T>(BinaryHeap<T>);

impl<T: Ord> Iterator for StreamPrefetcherHeap<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
    }
}

pub struct StreamPrefetcher<'r, 'b, R, T>
where
    R: RelationReadStream + 'r,
    T: Ord,
{
    stream: R::ReadStream<'r, 'b, T, StreamPrefetcherHeap<(T, AlwaysEqual<&'b mut [u32]>)>>,
}

impl<'r, 'b, R: RelationReadStream, T: Ord> StreamPrefetcher<'r, 'b, R, T> {
    pub fn new(relation: &'r R, vec: Vec<(T, AlwaysEqual<&'b mut [u32]>)>) -> Self {
        let stream = relation.read_stream(StreamPrefetcherHeap(BinaryHeap::from(vec)));
        Self { stream }
    }
}

impl<'r, 'b, R: RelationReadStream, T: Ord> IntoIterator for StreamPrefetcher<'r, 'b, R, T> {
    type Item = (T, AlwaysEqual<&'b mut [u32]>);

    type IntoIter = <<R as RelationReadStream>::ReadStream<
        'r,
        'b,
        T,
        StreamPrefetcherHeap<(T, AlwaysEqual<&'b mut [u32]>)>,
    > as ReadStream<'b, T>>::Inner;

    fn into_iter(self) -> Self::IntoIter {
        self.stream.into_inner()
    }
}

impl<'r, 'b, R: RelationReadStream, T: Ord> Prefetcher<'b> for StreamPrefetcher<'r, 'b, R, T> {
    type R = R;
    type T = T;
    fn pop_if<'s>(
        &'s mut self,
        predicate: impl FnOnce(&mut Self::Item) -> bool,
    ) -> Option<(T, Vec<R::ReadGuard<'s>>)> {
        self.stream.next_if(predicate)
    }
}

fn heap_pop_if<T: Ord>(
    this: &mut BinaryHeap<T>,
    predicate: impl FnOnce(&mut T) -> bool,
) -> Option<T> {
    let mut peek = this.peek_mut()?;
    if predicate(&mut peek) {
        Some(binary_heap::PeekMut::pop(peek))
    } else {
        None
    }
}

// Emulate unstable library feature `vec_deque_pop_if`.
// See https://github.com/rust-lang/rust/issues/135889.

fn vec_deque_pop_front_if<T>(
    this: &mut VecDeque<T>,
    predicate: impl FnOnce(&mut T) -> bool,
) -> Option<T> {
    let first = this.front_mut()?;
    if predicate(first) {
        this.pop_front()
    } else {
        None
    }
}
