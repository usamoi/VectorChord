use algo::prefetcher::{PlainPrefetcher, PrefetcherSequenceFamily};
use algo::tuples::ALIGN;
use algo::{
    Fetch, Page, PageGuard, Relation, RelationRead, RelationReadTypes, RelationWrite,
    RelationWriteTypes, Sequence,
};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::ops::{Deref, DerefMut};

thread_local! {
    static BUMP: &'static bumpalo::Bump<ALIGN> = Box::leak(Box::new(bumpalo::Bump::with_min_align()));
}

pub struct MockPage {
    tuples: Option<&'static mut [u8]>,
    opaque: vchordg::Opaque,
}

impl Page for MockPage {
    type Opaque = vchordg::Opaque;

    fn get_opaque(&self) -> &Self::Opaque {
        &self.opaque
    }

    fn get_opaque_mut(&mut self) -> &mut Self::Opaque {
        &mut self.opaque
    }

    fn len(&self) -> u16 {
        self.tuples.is_some() as u16
    }

    fn get(&self, i: u16) -> Option<&[u8]> {
        assert!(i == 1);
        if let Some(x) = self.tuples.as_ref() {
            Some(x)
        } else {
            None
        }
    }

    fn get_mut(&mut self, i: u16) -> Option<&mut [u8]> {
        assert!(i == 1);
        if let Some(x) = self.tuples.as_mut() {
            Some(x)
        } else {
            None
        }
    }

    fn alloc(&mut self, data: &[u8]) -> Option<u16> {
        assert!(self.tuples.is_none());
        self.tuples = Some(BUMP.with(|bump| *bump).alloc_slice_copy(data));
        Some(1)
    }

    fn free(&mut self, i: u16) {
        assert!(i == 1);
        assert!(self.tuples.is_some());
        self.tuples = None;
    }

    fn freespace(&self) -> u16 {
        if self.tuples.is_some() { 0 } else { u16::MAX }
    }

    fn clear(&mut self, opaque: Self::Opaque) {
        self.tuples = None;
        self.opaque = opaque;
    }
}

pub struct MockReadGuard {
    id: u32,
    inner: RwLockReadGuard<'static, MockPage>,
}

impl PageGuard for MockReadGuard {
    fn id(&self) -> u32 {
        self.id
    }
}

impl Deref for MockReadGuard {
    type Target = MockPage;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

pub struct MockWriteGuard {
    id: u32,
    inner: RwLockWriteGuard<'static, MockPage>,
}

impl PageGuard for MockWriteGuard {
    fn id(&self) -> u32 {
        self.id
    }
}

impl Deref for MockWriteGuard {
    type Target = MockPage;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl DerefMut for MockWriteGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

pub struct MockRelation {
    pages: boxcar::Vec<&'static RwLock<MockPage>>,
}

impl MockRelation {
    pub fn new() -> &'static Self {
        BUMP.with(|x| *x).alloc(MockRelation {
            pages: boxcar::Vec::new(),
        })
    }
}

impl Relation for &'static MockRelation {
    type Page = MockPage;
}

impl RelationReadTypes for &'static MockRelation {
    type ReadGuard<'r> = MockReadGuard;
}

impl RelationWriteTypes for &'static MockRelation {
    type WriteGuard<'r> = MockWriteGuard;
}

impl RelationRead for &'static MockRelation {
    fn read(&self, id: u32) -> Self::ReadGuard<'_> {
        let lock = self.pages[id as usize];
        MockReadGuard {
            id,
            inner: lock.read(),
        }
    }
}

impl RelationWrite for &'static MockRelation {
    fn write(&self, id: u32, _: bool) -> Self::WriteGuard<'_> {
        let lock = self.pages[id as usize];
        MockWriteGuard {
            id,
            inner: lock.write(),
        }
    }

    fn extend(&self, opaque: vchordg::Opaque, _: bool) -> Self::WriteGuard<'_> {
        let id = self
            .pages
            .push(BUMP.with(|bump| *bump).alloc(RwLock::new(MockPage {
                tuples: None,
                opaque,
            }))) as u32;
        let lock = self.pages[id as usize];
        MockWriteGuard {
            id,
            inner: lock.write(),
        }
    }

    fn search(&self, _: usize) -> Option<Self::WriteGuard<'_>> {
        None
    }
}

#[derive(Debug)]
pub struct MakePlainPrefetcher<'r, R> {
    pub index: &'r R,
}

impl<'r, R> Clone for MakePlainPrefetcher<'r, R> {
    fn clone(&self) -> Self {
        Self { index: self.index }
    }
}

impl<'r, R: RelationRead> PrefetcherSequenceFamily<'r, R> for MakePlainPrefetcher<'r, R> {
    type P<S: Sequence>
        = PlainPrefetcher<'r, R, S>
    where
        S::Item: Fetch;

    fn prefetch<S: Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: Fetch,
    {
        PlainPrefetcher::new(self.index, seq)
    }
}
