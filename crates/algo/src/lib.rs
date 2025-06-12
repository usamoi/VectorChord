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

#![allow(clippy::type_complexity)]

pub mod prefetcher;
pub mod tuples;

use always_equal::AlwaysEqual;
use std::collections::{BinaryHeap, VecDeque};
use std::iter::Peekable;
use std::ops::{Deref, DerefMut};
use zerocopy::IntoBytes;

pub trait Page: Sized {
    type Opaque: Opaque;

    #[must_use]
    fn get_opaque(&self) -> &Self::Opaque;
    #[must_use]
    fn get_opaque_mut(&mut self) -> &mut Self::Opaque;
    #[must_use]
    fn len(&self) -> u16;
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[must_use]
    fn get(&self, i: u16) -> Option<&[u8]>;
    #[must_use]
    fn get_mut(&mut self, i: u16) -> Option<&mut [u8]>;
    #[must_use]
    fn alloc(&mut self, data: &[u8]) -> Option<u16>;
    fn free(&mut self, i: u16);
    #[must_use]
    fn freespace(&self) -> u16;
    fn clear(&mut self, opaque: Self::Opaque);
}

pub trait PageGuard {
    fn id(&self) -> u32;
}

pub trait ReadStream<'s> {
    type Relation: RelationRead;
    type Item;
    type Inner: Iterator<Item = Self::Item>;
    fn next(
        &mut self,
    ) -> Option<(
        Self::Item,
        Vec<<Self::Relation as RelationRead>::ReadGuard<'s>>,
    )>;
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(
        Self::Item,
        Vec<<Self::Relation as RelationRead>::ReadGuard<'s>>,
    )>;
    fn into_inner(self) -> Self::Inner;
}

pub trait WriteStream<'s> {
    type Relation: RelationWrite;
    type Item;
    type Inner: Iterator<Item = Self::Item>;
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(
        Self::Item,
        Vec<<Self::Relation as RelationWrite>::WriteGuard<'s>>,
    )>;
    fn into_inner(self) -> Self::Inner;
}

pub trait Relation {
    type Page: Page;
}

pub trait RelationRead: Relation {
    type ReadGuard<'a>: PageGuard + Deref<Target = Self::Page>
    where
        Self: 'a;
    fn read(&self, id: u32) -> Self::ReadGuard<'_>;
}

pub trait RelationWrite: Relation {
    type WriteGuard<'a>: PageGuard + DerefMut<Target = Self::Page>
    where
        Self: 'a;
    fn write(&self, id: u32, tracking_freespace: bool) -> Self::WriteGuard<'_>;
    fn extend(
        &self,
        opaque: <Self::Page as Page>::Opaque,
        tracking_freespace: bool,
    ) -> Self::WriteGuard<'_>;
    fn search(&self, freespace: usize) -> Option<Self::WriteGuard<'_>>;
}

pub trait RelationLength: Relation {
    fn len(&self) -> u32;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait RelationPrefetch: Relation {
    fn prefetch(&self, id: u32);
}

#[derive(Debug, Default, Clone)]
pub struct Hints {
    pub full: bool,
}

impl Hints {
    #[allow(clippy::needless_update)]
    pub fn full(self, full: bool) -> Self {
        Self { full, ..self }
    }
}

pub trait RelationReadStream: RelationRead {
    type ReadStream<'s, I: Iterator>: ReadStream<'s, Item = I::Item, Relation = Self>
    where
        I::Item: Fetch,
        Self: 's;
    fn read_stream<I: Iterator>(&self, iter: I, hints: Hints) -> Self::ReadStream<'_, I>
    where
        I::Item: Fetch;
}

pub trait RelationWriteStream: RelationWrite {
    type WriteStream<'s, I: Iterator>: WriteStream<'s, Item = I::Item, Relation = Self>
    where
        I::Item: Fetch,
        Self: 's;
    fn write_stream<I: Iterator>(&self, iter: I, hints: Hints) -> Self::WriteStream<'_, I>
    where
        I::Item: Fetch;
}

#[derive(Debug, Clone, Copy)]
pub enum RerankMethod {
    Index,
    Heap,
}

pub trait Bump: 'static {
    #[allow(clippy::mut_from_ref)]
    fn alloc<T>(&self, value: T) -> &mut T;
    #[allow(clippy::mut_from_ref)]
    fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &mut [T];
}

pub trait Fetch {
    fn fetch(&self) -> &[u32];
}

impl Fetch for u32 {
    fn fetch(&self) -> &[u32] {
        std::slice::from_ref(self)
    }
}

impl<'b, T, A, B> Fetch for (T, AlwaysEqual<&'b mut (A, B, &'b mut [u32])>) {
    fn fetch(&self) -> &[u32] {
        let (_, AlwaysEqual((.., list))) = self;
        list
    }
}

impl<'b, T, A, B, C> Fetch for (T, AlwaysEqual<&'b mut (A, B, C, &'b mut [u32])>) {
    fn fetch(&self) -> &[u32] {
        let (_, AlwaysEqual((.., list))) = self;
        list
    }
}

impl<T> Fetch for (T, AlwaysEqual<(u32, u16)>) {
    fn fetch(&self) -> &[u32] {
        let (_, AlwaysEqual((x, _))) = self;
        std::slice::from_ref(x)
    }
}

impl Fetch for (u32, u16) {
    fn fetch(&self) -> &[u32] {
        std::slice::from_ref(&self.0)
    }
}

pub trait Sequence {
    type Item;
    type Inner: Iterator<Item = Self::Item>;
    #[must_use]
    fn next(&mut self) -> Option<Self::Item>;
    #[must_use]
    fn peek(&mut self) -> Option<&Self::Item>;
    #[must_use]
    fn into_inner(self) -> Self::Inner;
}

impl<T: Ord> Sequence for BinaryHeap<T> {
    type Item = T;
    type Inner = std::vec::IntoIter<T>;
    fn next(&mut self) -> Option<T> {
        self.pop()
    }
    fn peek(&mut self) -> Option<&T> {
        (self as &Self).peek()
    }
    fn into_inner(self) -> Self::Inner {
        self.into_vec().into_iter()
    }
}

impl<I: Iterator> Sequence for Peekable<I> {
    type Item = I::Item;
    type Inner = Peekable<I>;
    fn next(&mut self) -> Option<I::Item> {
        Iterator::next(self)
    }
    fn peek(&mut self) -> Option<&I::Item> {
        self.peek()
    }
    fn into_inner(self) -> Self::Inner {
        self
    }
}

impl<T> Sequence for VecDeque<T> {
    type Item = T;
    type Inner = std::collections::vec_deque::IntoIter<T>;
    fn next(&mut self) -> Option<T> {
        self.pop_front()
    }
    fn peek(&mut self) -> Option<&T> {
        self.front()
    }
    fn into_inner(self) -> Self::Inner {
        self.into_iter()
    }
}

#[allow(unsafe_code)]
pub unsafe trait Opaque: Copy + Send + Sync + IntoBytes + 'static {}
