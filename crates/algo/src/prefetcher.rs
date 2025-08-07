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

use crate::{
    Fetch, Hints, ReadStream, RelationPrefetch, RelationRead, RelationReadStream,
    RelationReadTypes, Sequence,
};
use std::collections::{VecDeque, vec_deque};
use std::iter::Chain;

pub const WINDOW_SIZE: usize = 32;
const _: () = assert!(WINDOW_SIZE > 0);

pub trait Prefetcher<'r>: IntoIterator
where
    Self::Item: Fetch,
{
    type R: RelationRead + 'r;
    type Guards: ExactSizeIterator<Item = <Self::R as RelationReadTypes>::ReadGuard<'r>>;

    #[must_use]
    fn next(&mut self) -> Option<(Self::Item, Self::Guards)>;
    #[must_use]
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(Self::Item, Self::Guards)>;
}

pub struct PlainPrefetcher<'r, R, S: Sequence> {
    relation: &'r R,
    sequence: S,
}

impl<'r, R, S: Sequence> PlainPrefetcher<'r, R, S> {
    #[inline]
    pub fn new(relation: &'r R, sequence: S) -> Self {
        Self { relation, sequence }
    }
}

impl<'r, R, S: Sequence> IntoIterator for PlainPrefetcher<'r, R, S> {
    type Item = S::Item;

    type IntoIter = S::Inner;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.sequence.into_inner()
    }
}

impl<'r, R: RelationRead, S: Sequence> Prefetcher<'r> for PlainPrefetcher<'r, R, S>
where
    S::Item: Fetch,
{
    type R = R;
    type Guards = PlainPrefetcherGuards<'r, R>;

    #[inline]
    fn next(&mut self) -> Option<(Self::Item, PlainPrefetcherGuards<'r, R>)> {
        let e = self.sequence.next()?;
        let list = e.fetch();
        Some((
            e,
            PlainPrefetcherGuards {
                relation: self.relation,
                list,
            },
        ))
    }

    #[inline]
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(S::Item, PlainPrefetcherGuards<'r, R>)> {
        if !predicate(self.sequence.peek()?) {
            return None;
        }
        let e = self.sequence.next()?;
        let list = e.fetch();
        Some((
            e,
            PlainPrefetcherGuards {
                relation: self.relation,
                list,
            },
        ))
    }
}

pub struct PlainPrefetcherGuards<'r, R> {
    relation: &'r R,
    list: crate::OwnedIter,
}

impl<'r, R: RelationRead> Iterator for PlainPrefetcherGuards<'r, R> {
    type Item = R::ReadGuard<'r>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let id = self.list.next()?;
        Some(self.relation.read(id))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.list.size_hint()
    }
}

impl<'r, R: RelationRead> ExactSizeIterator for PlainPrefetcherGuards<'r, R> {}

pub struct SimplePrefetcher<'r, R, S: Sequence> {
    relation: &'r R,
    window: VecDeque<S::Item>,
    sequence: S,
}

impl<'r, R, S: Sequence> SimplePrefetcher<'r, R, S> {
    #[inline]
    pub fn new(relation: &'r R, sequence: S) -> Self {
        Self {
            relation,
            window: VecDeque::new(),
            sequence,
        }
    }
}

impl<'r, R, S: Sequence> IntoIterator for SimplePrefetcher<'r, R, S> {
    type Item = S::Item;

    type IntoIter = Chain<vec_deque::IntoIter<S::Item>, S::Inner>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.window.into_iter().chain(self.sequence.into_inner())
    }
}

impl<'r, R: RelationRead + RelationPrefetch, S: Sequence> Prefetcher<'r>
    for SimplePrefetcher<'r, R, S>
where
    S::Item: Fetch,
{
    type R = R;
    type Guards = SimplePrefetcherGuards<'r, R>;

    #[inline]
    fn next(&mut self) -> Option<(Self::Item, SimplePrefetcherGuards<'r, R>)> {
        while self.window.len() < WINDOW_SIZE
            && let Some(e) = self.sequence.next()
        {
            for id in e.fetch() {
                self.relation.prefetch(id);
            }
            self.window.push_back(e);
        }
        let e = self.window.pop_front()?;
        let list = e.fetch();
        Some((
            e,
            SimplePrefetcherGuards {
                relation: self.relation,
                list,
            },
        ))
    }

    #[inline]
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&S::Item) -> bool,
    ) -> Option<(S::Item, SimplePrefetcherGuards<'r, R>)> {
        while self.window.len() < WINDOW_SIZE
            && let Some(e) = self.sequence.next()
        {
            for id in e.fetch() {
                self.relation.prefetch(id);
            }
            self.window.push_back(e);
        }
        let e = vec_deque_pop_front_if(&mut self.window, predicate)?;
        let list = e.fetch();
        Some((
            e,
            SimplePrefetcherGuards {
                relation: self.relation,
                list,
            },
        ))
    }
}

pub struct SimplePrefetcherGuards<'r, R> {
    relation: &'r R,
    list: crate::OwnedIter,
}

impl<'r, R: RelationRead> Iterator for SimplePrefetcherGuards<'r, R> {
    type Item = R::ReadGuard<'r>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let id = self.list.next()?;
        Some(self.relation.read(id))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.list.size_hint()
    }
}

impl<'r, R: RelationRead> ExactSizeIterator for SimplePrefetcherGuards<'r, R> {}

pub struct StreamPrefetcherSequence<S>(S);

impl<S: Sequence> Iterator for StreamPrefetcherSequence<S> {
    type Item = S::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

pub struct StreamPrefetcher<'r, R: RelationReadStream + 'r, S: Sequence>
where
    S::Item: Fetch,
{
    stream: R::ReadStream<'r, StreamPrefetcherSequence<S>>,
}

impl<'r, R: RelationReadStream, S: Sequence> StreamPrefetcher<'r, R, S>
where
    S::Item: Fetch,
{
    #[inline]
    pub fn new(relation: &'r R, sequence: S, hints: Hints) -> Self {
        let stream = relation.read_stream(StreamPrefetcherSequence(sequence), hints);
        Self { stream }
    }
}

impl<'r, R: RelationReadStream, S: Sequence> IntoIterator for StreamPrefetcher<'r, R, S>
where
    S::Item: Fetch,
{
    type Item = S::Item;

    type IntoIter = <R::ReadStream<'r, StreamPrefetcherSequence<S>> as ReadStream<'r>>::Inner;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.stream.into_inner()
    }
}

impl<'r, R: RelationRead + RelationReadStream, S: Sequence> Prefetcher<'r>
    for StreamPrefetcher<'r, R, S>
where
    S::Item: Fetch,
{
    type R = R;

    type Guards = <R::ReadStream<'r, StreamPrefetcherSequence<S>> as ReadStream<'r>>::Guards;

    #[inline]
    fn next(&mut self) -> Option<(S::Item, Self::Guards)> {
        self.stream.next()
    }

    #[inline]
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(S::Item, Self::Guards)> {
        self.stream.next_if(predicate)
    }
}

pub trait PrefetcherSequenceFamily<'r, R> {
    type P<S: Sequence>: Prefetcher<'r, R = R, Item = S::Item>
    where
        S::Item: Fetch;

    fn prefetch<S: Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: Fetch;

    fn is_not_plain(&self) -> bool;
}

pub trait PrefetcherHeapFamily<'r, R> {
    type P<T>: Prefetcher<'r, R = R, Item = T>
    where
        T: Ord + Fetch + 'r;

    fn prefetch<T>(&mut self, seq: Vec<T>) -> Self::P<T>
    where
        T: Ord + Fetch + 'r;

    fn is_not_plain(&self) -> bool;
}

// Emulate unstable library feature `vec_deque_pop_if`.
// See https://github.com/rust-lang/rust/issues/135889.

fn vec_deque_pop_front_if<T>(
    this: &mut VecDeque<T>,
    predicate: impl FnOnce(&T) -> bool,
) -> Option<T> {
    let first = this.front()?;
    if predicate(first) {
        this.pop_front()
    } else {
        None
    }
}
