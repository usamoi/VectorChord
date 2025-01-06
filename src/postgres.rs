use algorithm::{Opaque, Page, PageGuard, RelationRead, RelationWrite};
use std::mem::{MaybeUninit, offset_of};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

const _: () = assert!(
    offset_of!(pgrx::pg_sys::PageHeaderData, pd_linp) % pgrx::pg_sys::MAXIMUM_ALIGNOF as usize == 0
);

const fn size_of_contents() -> usize {
    use pgrx::pg_sys::{BLCKSZ, PageHeaderData};
    let size_of_page = BLCKSZ as usize;
    let size_of_header = offset_of!(PageHeaderData, pd_linp);
    let size_of_opaque = size_of::<Opaque>();
    size_of_page - size_of_header - size_of_opaque
}

#[repr(C, align(8))]
pub struct PostgresPage {
    header: pgrx::pg_sys::PageHeaderData,
    content: [u8; size_of_contents()],
    opaque: Opaque,
}

const _: () = assert!(align_of::<PostgresPage>() == pgrx::pg_sys::MAXIMUM_ALIGNOF as usize);
const _: () = assert!(size_of::<PostgresPage>() == pgrx::pg_sys::BLCKSZ as usize);

impl PostgresPage {
    fn init_mut(this: &mut MaybeUninit<Self>) -> &mut Self {
        unsafe {
            pgrx::pg_sys::PageInit(
                this.as_mut_ptr() as pgrx::pg_sys::Page,
                pgrx::pg_sys::BLCKSZ as usize,
                size_of::<Opaque>(),
            );
            (&raw mut (*this.as_mut_ptr()).opaque).write(Opaque {
                next: u32::MAX,
                skip: u32::MAX,
            });
        }
        let this = unsafe { MaybeUninit::assume_init_mut(this) };
        assert_eq!(offset_of!(Self, opaque), this.header.pd_special as usize);
        this
    }
    #[allow(dead_code)]
    unsafe fn assume_init_mut(this: &mut MaybeUninit<Self>) -> &mut Self {
        let this = unsafe { MaybeUninit::assume_init_mut(this) };
        assert_eq!(offset_of!(Self, opaque), this.header.pd_special as usize);
        this
    }
    #[allow(dead_code)]
    fn clone_into_boxed(&self) -> Box<Self> {
        let mut result = Box::new_uninit();
        unsafe {
            std::ptr::copy(self as *const Self, result.as_mut_ptr(), 1);
            result.assume_init()
        }
    }
}

impl Page for PostgresPage {
    fn get_opaque(&self) -> &Opaque {
        &self.opaque
    }
    fn get_opaque_mut(&mut self) -> &mut Opaque {
        &mut self.opaque
    }
    fn len(&self) -> u16 {
        use pgrx::pg_sys::{ItemIdData, PageHeaderData};
        assert!(self.header.pd_lower as usize <= size_of::<Self>());
        assert!(self.header.pd_upper as usize <= size_of::<Self>());
        let lower = self.header.pd_lower as usize;
        let upper = self.header.pd_upper as usize;
        assert!(lower <= upper);
        ((lower - offset_of!(PageHeaderData, pd_linp)) / size_of::<ItemIdData>()) as u16
    }
    fn get(&self, i: u16) -> Option<&[u8]> {
        use pgrx::pg_sys::{ItemIdData, PageHeaderData};
        if i == 0 {
            return None;
        }
        assert!(self.header.pd_lower as usize <= size_of::<Self>());
        let lower = self.header.pd_lower as usize;
        let n = ((lower - offset_of!(PageHeaderData, pd_linp)) / size_of::<ItemIdData>()) as u16;
        if i > n {
            return None;
        }
        let iid = unsafe { self.header.pd_linp.as_ptr().add((i - 1) as _).read() };
        let lp_off = iid.lp_off() as usize;
        let lp_len = iid.lp_len() as usize;
        match iid.lp_flags() {
            pgrx::pg_sys::LP_UNUSED => return None,
            pgrx::pg_sys::LP_NORMAL => (),
            pgrx::pg_sys::LP_REDIRECT => unimplemented!(),
            pgrx::pg_sys::LP_DEAD => unimplemented!(),
            _ => unreachable!(),
        }
        assert!(offset_of!(PageHeaderData, pd_linp) <= lp_off && lp_off <= size_of::<Self>());
        assert!(lp_len <= size_of::<Self>());
        assert!(lp_off + lp_len <= size_of::<Self>());
        unsafe {
            let ptr = (self as *const Self).cast::<u8>().add(lp_off as _);
            Some(std::slice::from_raw_parts(ptr, lp_len as _))
        }
    }
    fn get_mut(&mut self, i: u16) -> Option<&mut [u8]> {
        use pgrx::pg_sys::{ItemIdData, PageHeaderData};
        if i == 0 {
            return None;
        }
        assert!(self.header.pd_lower as usize <= size_of::<Self>());
        let lower = self.header.pd_lower as usize;
        let n = ((lower - offset_of!(PageHeaderData, pd_linp)) / size_of::<ItemIdData>()) as u16;
        if i > n {
            return None;
        }
        let iid = unsafe { self.header.pd_linp.as_ptr().add((i - 1) as _).read() };
        let lp_off = iid.lp_off() as usize;
        let lp_len = iid.lp_len() as usize;
        assert!(offset_of!(PageHeaderData, pd_linp) <= lp_off && lp_off <= size_of::<Self>());
        assert!(lp_len <= size_of::<Self>());
        assert!(lp_off + lp_len <= size_of::<Self>());
        unsafe {
            let ptr = (self as *mut Self).cast::<u8>().add(lp_off as _);
            Some(std::slice::from_raw_parts_mut(ptr, lp_len as _))
        }
    }
    fn alloc(&mut self, data: &[u8]) -> Option<u16> {
        unsafe {
            let i = pgrx::pg_sys::PageAddItemExtended(
                (self as *const Self).cast_mut().cast(),
                data.as_ptr().cast_mut().cast(),
                data.len(),
                0,
                0,
            );
            if i == 0 { None } else { Some(i) }
        }
    }
    fn free(&mut self, i: u16) {
        unsafe {
            pgrx::pg_sys::PageIndexTupleDeleteNoCompact((self as *mut Self).cast(), i);
        }
    }
    fn reconstruct(&mut self, removes: &[u16]) {
        let mut removes = removes.to_vec();
        removes.sort();
        removes.dedup();
        let n = removes.len();
        if n > 0 {
            assert!(removes[n - 1] <= self.len());
            unsafe {
                pgrx::pg_sys::PageIndexMultiDelete(
                    (self as *mut Self).cast(),
                    removes.as_ptr().cast_mut(),
                    removes.len() as _,
                );
            }
        }
    }
    fn freespace(&self) -> u16 {
        unsafe { pgrx::pg_sys::PageGetFreeSpace((self as *const Self).cast_mut().cast()) as u16 }
    }
}

const _: () = assert!(align_of::<Opaque>() == pgrx::pg_sys::MAXIMUM_ALIGNOF as usize);

pub struct PostgresBufferReadGuard {
    buf: i32,
    page: NonNull<PostgresPage>,
    id: u32,
}

impl PageGuard for PostgresBufferReadGuard {
    fn id(&self) -> u32 {
        self.id
    }
}

impl Deref for PostgresBufferReadGuard {
    type Target = PostgresPage;

    fn deref(&self) -> &PostgresPage {
        unsafe { self.page.as_ref() }
    }
}

impl Drop for PostgresBufferReadGuard {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::UnlockReleaseBuffer(self.buf);
        }
    }
}

pub struct PostgresBufferWriteGuard {
    raw: pgrx::pg_sys::Relation,
    buf: i32,
    page: NonNull<PostgresPage>,
    state: *mut pgrx::pg_sys::GenericXLogState,
    id: u32,
    tracking_freespace: bool,
}

impl PageGuard for PostgresBufferWriteGuard {
    fn id(&self) -> u32 {
        self.id
    }
}

impl Deref for PostgresBufferWriteGuard {
    type Target = PostgresPage;

    fn deref(&self) -> &PostgresPage {
        unsafe { self.page.as_ref() }
    }
}

impl DerefMut for PostgresBufferWriteGuard {
    fn deref_mut(&mut self) -> &mut PostgresPage {
        unsafe { self.page.as_mut() }
    }
}

impl Drop for PostgresBufferWriteGuard {
    fn drop(&mut self) {
        unsafe {
            if std::thread::panicking() {
                pgrx::pg_sys::GenericXLogAbort(self.state);
            } else {
                if self.tracking_freespace {
                    pgrx::pg_sys::RecordPageWithFreeSpace(self.raw, self.id, self.freespace() as _);
                    pgrx::pg_sys::FreeSpaceMapVacuumRange(self.raw, self.id, self.id + 1);
                }
                pgrx::pg_sys::GenericXLogFinish(self.state);
            }
            pgrx::pg_sys::UnlockReleaseBuffer(self.buf);
        }
    }
}

#[derive(Debug, Clone)]
pub struct PostgresRelation {
    raw: pgrx::pg_sys::Relation,
}

impl PostgresRelation {
    pub unsafe fn new(raw: pgrx::pg_sys::Relation) -> Self {
        Self { raw }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> u32 {
        unsafe {
            pgrx::pg_sys::RelationGetNumberOfBlocksInFork(
                self.raw,
                pgrx::pg_sys::ForkNumber::MAIN_FORKNUM,
            )
        }
    }
}

impl RelationRead for PostgresRelation {
    type Page = PostgresPage;

    type ReadGuard<'a> = PostgresBufferReadGuard;

    fn read(&self, id: u32) -> Self::ReadGuard<'_> {
        assert!(id != u32::MAX, "no such page");
        unsafe {
            use pgrx::pg_sys::{
                BUFFER_LOCK_SHARE, BufferGetPage, LockBuffer, ReadBufferExtended, ReadBufferMode,
            };
            let buf = ReadBufferExtended(
                self.raw,
                0,
                id,
                ReadBufferMode::RBM_NORMAL,
                std::ptr::null_mut(),
            );
            LockBuffer(buf, BUFFER_LOCK_SHARE as _);
            let page = NonNull::new(BufferGetPage(buf).cast()).expect("failed to get page");
            PostgresBufferReadGuard { buf, page, id }
        }
    }
}

impl RelationWrite for PostgresRelation {
    type WriteGuard<'a> = PostgresBufferWriteGuard;

    fn write(&self, id: u32, tracking_freespace: bool) -> PostgresBufferWriteGuard {
        assert!(id != u32::MAX, "no such page");
        unsafe {
            use pgrx::pg_sys::{
                BUFFER_LOCK_EXCLUSIVE, ForkNumber, GENERIC_XLOG_FULL_IMAGE,
                GenericXLogRegisterBuffer, GenericXLogStart, LockBuffer, ReadBufferExtended,
                ReadBufferMode,
            };
            let buf = ReadBufferExtended(
                self.raw,
                ForkNumber::MAIN_FORKNUM,
                id,
                ReadBufferMode::RBM_NORMAL,
                std::ptr::null_mut(),
            );
            LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE as _);
            let state = GenericXLogStart(self.raw);
            let page = NonNull::new(
                GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE as _)
                    .cast::<MaybeUninit<PostgresPage>>(),
            )
            .expect("failed to get page");
            PostgresBufferWriteGuard {
                raw: self.raw,
                buf,
                page: page.cast(),
                state,
                id,
                tracking_freespace,
            }
        }
    }
    fn extend(&self, tracking_freespace: bool) -> PostgresBufferWriteGuard {
        unsafe {
            use pgrx::pg_sys::{
                BUFFER_LOCK_EXCLUSIVE, ExclusiveLock, ForkNumber, GENERIC_XLOG_FULL_IMAGE,
                GenericXLogRegisterBuffer, GenericXLogStart, LockBuffer, LockRelationForExtension,
                ReadBufferExtended, ReadBufferMode, UnlockRelationForExtension,
            };
            LockRelationForExtension(self.raw, ExclusiveLock as _);
            let buf = ReadBufferExtended(
                self.raw,
                ForkNumber::MAIN_FORKNUM,
                u32::MAX,
                ReadBufferMode::RBM_NORMAL,
                std::ptr::null_mut(),
            );
            UnlockRelationForExtension(self.raw, ExclusiveLock as _);
            LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE as _);
            let state = GenericXLogStart(self.raw);
            let mut page = NonNull::new(
                GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE as _)
                    .cast::<MaybeUninit<PostgresPage>>(),
            )
            .expect("failed to get page");
            PostgresPage::init_mut(page.as_mut());
            PostgresBufferWriteGuard {
                raw: self.raw,
                buf,
                page: page.cast(),
                state,
                id: pgrx::pg_sys::BufferGetBlockNumber(buf),
                tracking_freespace,
            }
        }
    }
    fn search(&self, freespace: usize) -> Option<PostgresBufferWriteGuard> {
        unsafe {
            loop {
                let id = pgrx::pg_sys::GetPageWithFreeSpace(self.raw, freespace);
                if id == u32::MAX {
                    return None;
                }
                let write = self.write(id, true);
                if write.freespace() < freespace as _ {
                    // the free space is recorded incorrectly
                    pgrx::pg_sys::RecordPageWithFreeSpace(self.raw, id, write.freespace() as _);
                    pgrx::pg_sys::FreeSpaceMapVacuumRange(self.raw, id, id + 1);
                    continue;
                }
                return Some(write);
            }
        }
    }
}
