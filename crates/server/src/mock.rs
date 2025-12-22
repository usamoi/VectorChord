#![allow(unsafe_code)]

use index::relation::{
    Opaque, Page, PageGuard, Relation, RelationRead, RelationReadTypes, RelationWrite,
    RelationWriteTypes,
};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use zerocopy::FromZeros;

#[repr(C, align(8))]
#[derive(Debug, zerocopy::FromZeros)]
pub struct PageHeaderData {
    pub _unknown_0: [u8; 0xc],
    pub pd_lower: u16,
    pub pd_upper: u16,
    pub pd_special: u16,
    pub _unknown_1: [u8; 0x6],
}

#[repr(C, align(8))]
pub struct MockPage<O> {
    pub header: PageHeaderData,
    pub content: [u8; 8192_usize - size_of::<PageHeaderData>()],
    _opaque: PhantomData<fn(O) -> O>,
}

const _: () = assert!(size_of::<MockPage<()>>() == 8192);

impl<O: Opaque> Page for MockPage<O> {
    type Opaque = O;
    fn get_opaque(&self) -> &O {
        assert!(self.header.pd_special as usize == size_of::<Self>() - size_of::<O>());
        let offset = self.header.pd_special as usize - size_of::<PageHeaderData>();
        unsafe { &*self.content.as_ptr().add(offset).cast::<O>() }
    }
    fn get_opaque_mut(&mut self) -> &mut O {
        assert!(self.header.pd_special as usize == size_of::<Self>() - size_of::<O>());
        let offset = self.header.pd_special as usize - size_of::<PageHeaderData>();
        unsafe { &mut *self.content.as_mut_ptr().add(offset).cast::<O>() }
    }
    fn len(&self) -> u16 {
        use u32 as ItemIdData;
        assert!(self.header.pd_lower as usize <= size_of::<Self>());
        assert!(self.header.pd_upper as usize <= size_of::<Self>());
        let lower = self.header.pd_lower as usize;
        let upper = self.header.pd_upper as usize;
        assert!(size_of::<PageHeaderData>() <= lower && lower <= upper);
        ((lower - size_of::<PageHeaderData>()) / size_of::<ItemIdData>()) as u16
    }
    fn get(&self, i: u16) -> Option<&[u8]> {
        use u32 as ItemIdData;
        if i == 0 {
            return None;
        }
        assert!(self.header.pd_lower as usize <= size_of::<Self>());
        let lower = self.header.pd_lower as usize;
        assert!(size_of::<PageHeaderData>() <= lower);
        let n = ((lower - size_of::<PageHeaderData>()) / size_of::<ItemIdData>()) as u16;
        if i > n {
            return None;
        }
        let pd_linp = self.content.as_ptr().cast::<ItemIdData>();
        let lp = unsafe { pd_linp.add((i - 1) as _).read() };
        let lp_off = lp_off(lp) as usize;
        match lp_flags(lp) {
            0 => return None,
            1 => (),
            2 => unimplemented!(),
            3 => unimplemented!(),
            _ => unreachable!(),
        }
        let lp_len = lp_len(lp) as usize;
        assert!(size_of::<PageHeaderData>() <= lp_off);
        assert!(lp_off <= size_of::<Self>());
        assert!(lp_len <= size_of::<Self>());
        assert!(lp_off + lp_len <= size_of::<Self>());
        let offset = lp_off - size_of::<PageHeaderData>();
        unsafe {
            let ptr = self.content.as_ptr().add(offset);
            Some(std::slice::from_raw_parts(ptr, lp_len as _))
        }
    }
    fn get_mut(&mut self, i: u16) -> Option<&mut [u8]> {
        use u32 as ItemIdData;
        if i == 0 {
            return None;
        }
        assert!(self.header.pd_lower as usize <= size_of::<Self>());
        let lower = self.header.pd_lower as usize;
        assert!(size_of::<PageHeaderData>() <= lower);
        let n = ((lower - size_of::<PageHeaderData>()) / size_of::<ItemIdData>()) as u16;
        if i > n {
            return None;
        }
        let pd_linp = self.content.as_ptr().cast::<ItemIdData>();
        let lp = unsafe { pd_linp.add((i - 1) as _).read() };
        let lp_off = lp_off(lp) as usize;
        match lp_flags(lp) {
            0 => return None,
            1 => (),
            2 => unimplemented!(),
            3 => unimplemented!(),
            _ => unreachable!(),
        }
        let lp_len = lp_len(lp) as usize;
        assert!(size_of::<PageHeaderData>() <= lp_off);
        assert!(lp_off <= size_of::<Self>());
        assert!(lp_len <= size_of::<Self>());
        assert!(lp_off + lp_len <= size_of::<Self>());
        let offset = lp_off - size_of::<PageHeaderData>();
        unsafe {
            let ptr = self.content.as_mut_ptr().add(offset);
            Some(std::slice::from_raw_parts_mut(ptr, lp_len as _))
        }
    }
    fn alloc(&mut self, data: &[u8]) -> Option<u16> {
        let i = page_alloc(self, data);
        if i == 0 { None } else { Some(i) }
    }
    fn free(&mut self, i: u16) {
        page_free(self, i)
    }
    fn freespace(&self) -> u16 {
        page_freespace(self)
    }
    fn clear(&mut self, opaque: O) {
        unsafe { page_init(self, opaque) }
    }
}

unsafe fn page_init<O: Opaque>(this: *mut MockPage<O>, opaque: O) {
    unsafe {
        this.write_bytes(0, 1);
        (*this).header.pd_lower = size_of::<PageHeaderData>() as _;
        let offset = size_of::<MockPage<O>>() - size_of::<O>();
        (*this).header.pd_upper = offset as _;
        (*this).header.pd_special = offset as _;
        (*this)
            .content
            .as_mut_ptr()
            .add(offset - 24)
            .cast::<O>()
            .write(opaque);
    }
}

fn page_alloc<O: Opaque>(this: &mut MockPage<O>, data: &[u8]) -> u16 {
    use u32 as ItemIdData;
    if (page_freespace(this) as usize) < data.len() {
        return 0;
    }
    let limit = this.len();
    let lower = this.header.pd_lower + size_of::<ItemIdData>() as u16;
    let upper = this.header.pd_upper - data.len().next_multiple_of(8) as u16;
    assert!(lower <= upper);
    this.content[lower as usize - 24 - 4..][..4]
        .copy_from_slice(&lp(upper as _, 1, data.len() as _).to_ne_bytes());
    this.content[upper as usize - 24..][..data.len()].copy_from_slice(data);
    this.header.pd_lower = lower;
    this.header.pd_upper = upper;
    limit + 1
}

fn page_free<O: Opaque>(_this: &MockPage<O>, _i: u16) {
    unimplemented!()
}

fn page_freespace<O: Opaque>(this: &MockPage<O>) -> u16 {
    use u32 as ItemIdData;
    this.header
        .pd_upper
        .saturating_sub(this.header.pd_lower)
        .saturating_sub(size_of::<ItemIdData>() as _)
}

#[inline(always)]
fn lp(off: u32, flags: u32, len: u32) -> u32 {
    assert!(off < (1 << 15));
    assert!(flags < (1 << 2));
    assert!(len < (1 << 15));
    #[cfg(target_endian = "little")]
    {
        (off << 0) | (flags << 15) | (len << 17)
    }
    #[cfg(target_endian = "big")]
    {
        (off << 17) | (flags << 15) | (len << 0)
    }
}

#[inline(always)]
fn lp_off(x: u32) -> u32 {
    let x: u32 = unsafe { std::mem::transmute(x) };
    #[cfg(target_endian = "little")]
    {
        (x >> 0) & ((1 << 15) - 1)
    }
    #[cfg(target_endian = "big")]
    {
        (x >> 17) & ((1 << 15) - 1)
    }
}

#[inline(always)]
fn lp_flags(x: u32) -> u32 {
    let x: u32 = unsafe { std::mem::transmute(x) };
    #[cfg(target_endian = "little")]
    {
        (x >> 15) & ((1 << 2) - 1)
    }
    #[cfg(target_endian = "big")]
    {
        (x >> 15) & ((1 << 2) - 1)
    }
}

#[inline(always)]
fn lp_len(x: u32) -> u32 {
    let x: u32 = unsafe { std::mem::transmute(x) };
    #[cfg(target_endian = "little")]
    {
        (x >> 17) & ((1 << 15) - 1)
    }
    #[cfg(target_endian = "big")]
    {
        (x >> 0) & ((1 << 15) - 1)
    }
}

pub struct Registry<O> {
    pub pages: Vec<RwLock<MockPage<O>>>,
    pub n: AtomicU32,
    fsm: Mutex<Fsm>,
}

#[derive(Clone)]
pub struct MockRelation<O> {
    pub registry: Arc<Registry<O>>,
    _phantom: PhantomData<fn(O) -> O>,
}

impl<O: Opaque> MockRelation<O> {
    pub fn new(capacity: usize) -> Self {
        Self {
            registry: Arc::new(Registry {
                pages: {
                    let mut result = Vec::new();
                    result.resize_with(capacity, || {
                        RwLock::new(MockPage {
                            header: FromZeros::new_zeroed(),
                            content: FromZeros::new_zeroed(),
                            _opaque: FromZeros::new_zeroed(),
                        })
                    });
                    result
                },
                n: AtomicU32::new(0),
                fsm: Mutex::new(Fsm {
                    free: Default::default(),
                }),
            }),
            _phantom: PhantomData,
        }
    }
}

pub struct MockRelationReadGuard<'b, O> {
    id: u32,
    page: RwLockReadGuard<'b, MockPage<O>>,
}

pub struct MockRelationWriteGuard<'b, O: Opaque> {
    id: u32,
    page: RwLockWriteGuard<'b, MockPage<O>>,
    fsm: Option<&'b Mutex<Fsm>>,
}

impl<'b, O: Opaque> Drop for MockRelationWriteGuard<'b, O> {
    fn drop(&mut self) {
        if let Some(fsm) = self.fsm {
            let mut fsm = fsm.lock().unwrap();
            fsm.free.push(self.id, self.freespace());
        }
    }
}

impl<O: Opaque> Relation for MockRelation<O> {
    type Page = MockPage<O>;
}

impl<'b, O: Opaque> PageGuard for MockRelationReadGuard<'b, O> {
    fn id(&self) -> u32 {
        self.id
    }
}

impl<'b, O: Opaque> Deref for MockRelationReadGuard<'b, O> {
    type Target = MockPage<O>;

    fn deref(&self) -> &Self::Target {
        &self.page
    }
}

impl<O: Opaque> RelationReadTypes for MockRelation<O> {
    type ReadGuard<'b> = MockRelationReadGuard<'b, O>;
}

impl<O: Opaque> RelationRead for MockRelation<O> {
    fn read(&self, id: u32) -> Self::ReadGuard<'_> {
        assert!(id <= self.registry.n.load(Ordering::Acquire));
        MockRelationReadGuard {
            id,
            page: self.registry.pages[id as usize].read().unwrap(),
        }
    }
}

impl<'b, O: Opaque> PageGuard for MockRelationWriteGuard<'b, O> {
    fn id(&self) -> u32 {
        self.id
    }
}

impl<'b, O: Opaque> Deref for MockRelationWriteGuard<'b, O> {
    type Target = MockPage<O>;

    fn deref(&self) -> &Self::Target {
        &self.page
    }
}

impl<'b, O: Opaque> DerefMut for MockRelationWriteGuard<'b, O> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.page
    }
}

impl<O: Opaque> RelationWriteTypes for MockRelation<O> {
    type WriteGuard<'b> = MockRelationWriteGuard<'b, O>;
}

impl<O: Opaque> RelationWrite for MockRelation<O> {
    fn write(&self, id: u32, tracking_freespace: bool) -> Self::WriteGuard<'_> {
        assert!(id <= self.registry.n.load(Ordering::Acquire));
        let page = self.registry.pages[id as usize].write().unwrap();
        {
            let mut fsm = self.registry.fsm.lock().unwrap();
            fsm.free.remove(&id);
        }
        MockRelationWriteGuard {
            id,
            page,
            fsm: tracking_freespace.then_some(&self.registry.fsm),
        }
    }

    fn extend(
        &self,
        opaque: <Self::Page as Page>::Opaque,
        tracking_freespace: bool,
    ) -> Self::WriteGuard<'_> {
        let id = self.registry.n.fetch_add(1, Ordering::AcqRel);
        assert!((id as usize) < self.registry.pages.len());
        let mut page = self.registry.pages[id as usize].write().unwrap();
        unsafe {
            page_init(page.deref_mut(), opaque);
            debug_assert_eq!(page.get_opaque().as_bytes(), opaque.as_bytes());
        }
        MockRelationWriteGuard {
            id,
            page,
            fsm: tracking_freespace.then_some(&self.registry.fsm),
        }
    }

    fn search(&self, freespace: usize) -> Option<Self::WriteGuard<'_>> {
        if freespace >= 8192 {
            return None;
        }
        loop {
            let id = {
                let mut fsm = self.registry.fsm.lock().unwrap();
                fsm.free.pop_if(|_, &mut p| p as usize >= freespace).map(|(i, _)| i)
            };
            if let Some(id) = id {
                let guard = self.write(id, true);
                if (guard.freespace() as usize) < freespace {
                    continue;
                }
                break Some(guard);
            } else {
                break None;
            }
        }
    }
}

// todo: replace this bad implementation

struct Fsm {
    free: priority_queue::PriorityQueue<u32, u16>,
}
