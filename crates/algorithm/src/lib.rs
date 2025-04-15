#![feature(let_chains)]

mod build;
mod bulkdelete;
mod cache;
mod closure_lifetime_binder;
mod cost;
mod freepages;
mod insert;
mod linked_vec;
mod maintain;
mod prefetcher;
mod prewarm;
mod rerank;
mod search;
mod tape;
mod tuples;
mod vectors;

pub mod operator;
pub mod types;

use always_equal::AlwaysEqual;
pub use build::build;
pub use bulkdelete::bulkdelete;
pub use cache::cache;
pub use cost::cost;
pub use insert::insert;
pub use maintain::maintain;
pub use prefetcher::{PlainPrefetcher, Prefetcher, SimplePrefetcher, StreamPrefetcher};
pub use prewarm::prewarm;
pub use rerank::{how, rerank_heap, rerank_index};
pub use search::{default_search, maxsim_search};

use std::ops::{Deref, DerefMut};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct Opaque {
    pub next: u32,
    pub skip: u32,
}

pub trait Page: Sized {
    #[must_use]
    fn get_opaque(&self) -> &Opaque;
    #[must_use]
    fn get_opaque_mut(&mut self) -> &mut Opaque;
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
    fn clear(&mut self);
}

pub trait PageGuard {
    fn id(&self) -> u32;
}

pub trait ReadStream<'b, T> {
    type Relation: RelationRead;
    type Inner: Iterator<Item = (T, AlwaysEqual<&'b mut [u32]>)>;
    fn next_if<'s>(
        &'s mut self,
        predicate: impl FnOnce(&mut (T, AlwaysEqual<&'b mut [u32]>)) -> bool,
    ) -> Option<(T, Vec<<Self::Relation as RelationRead>::ReadGuard<'s>>)>;
    fn into_inner(self) -> Self::Inner;
}

pub trait WriteStream<'b, T> {
    type Relation: RelationWrite;
    type Inner: Iterator<Item = (T, AlwaysEqual<&'b mut [u32]>)>;
    fn next_if<'s>(
        &'s mut self,
        predicate: impl FnOnce(&mut (T, AlwaysEqual<&'b mut [u32]>)) -> bool,
    ) -> Option<(T, Vec<<Self::Relation as RelationWrite>::WriteGuard<'s>>)>;
    fn into_inner(self) -> Self::Inner;
}

pub trait Relation: Clone {
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
    fn extend(&self, tracking_freespace: bool) -> Self::WriteGuard<'_>;
    fn search(&self, freespace: usize) -> Option<Self::WriteGuard<'_>>;
}

pub trait RelationPrefetch: Relation {
    fn prefetch(&self, id: u32);
}

pub trait RelationReadStream: RelationRead {
    type ReadStream<'s, 'b, T, I: Iterator<Item = (T, AlwaysEqual<&'b mut [u32]>)>>: ReadStream<'b, T, Relation = Self>
    where
        Self: 's;
    fn read_stream<'s, 'b, T, I: Iterator<Item = (T, AlwaysEqual<&'b mut [u32]>)>>(
        &'s self,
        iter: I,
    ) -> Self::ReadStream<'s, 'b, T, I>;
}

pub trait RelationWriteStream: RelationWrite {
    type WriteStream<'s, 'b, T, I: Iterator<Item = (T, AlwaysEqual<&'b mut [u32]>)>>: WriteStream<'b, T, Relation = Self>
    where
        Self: 's;
    fn write_stream<'s, 'b, T, I: Iterator<Item = (T, AlwaysEqual<&'b mut [u32]>)>>(
        &'s self,
        iter: I,
    ) -> Self::WriteStream<'s, 'b, T, I>;
}

#[derive(Debug, Clone, Copy)]
pub enum RerankMethod {
    Index,
    Heap,
}

pub(crate) struct Branch<T> {
    pub head: u16,
    pub dis_u_2: f32,
    pub factor_ppc: f32,
    pub factor_ip: f32,
    pub factor_err: f32,
    pub signs: Vec<bool>,
    pub prefetch: Vec<u32>,
    pub extra: T,
}
