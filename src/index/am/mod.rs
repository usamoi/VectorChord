pub mod am_build;

use crate::index::gucs::{epsilon, max_maxsim_tuples, max_scan_tuples, maxsim_threshold, probes};
use crate::index::opclass::opfamily;
use crate::index::scanners::*;
use crate::index::storage::PostgresRelation;
use pgrx::datum::Internal;
use pgrx::pg_sys::{Datum, ItemPointerData};
use std::cell::LazyCell;
use std::ffi::CStr;
use std::num::NonZeroU64;
use std::sync::OnceLock;

#[repr(C)]
struct Reloption {
    vl_len_: i32,
    pub options: i32,
}

impl Reloption {
    unsafe fn options<'a>(this: *const Self) -> &'a CStr {
        unsafe {
            let ptr = this
                .cast::<u8>()
                .add((&raw const (*this).options).read() as _);
            CStr::from_ptr(ptr.cast())
        }
    }
}

const TABLE: &[pgrx::pg_sys::relopt_parse_elt] = &[pgrx::pg_sys::relopt_parse_elt {
    optname: c"options".as_ptr(),
    opttype: pgrx::pg_sys::relopt_type::RELOPT_TYPE_STRING,
    offset: std::mem::offset_of!(Reloption, options) as i32,
}];

static RELOPT_KIND: OnceLock<pgrx::pg_sys::relopt_kind::Type> = OnceLock::new();

pub fn init() {
    RELOPT_KIND.get_or_init(|| {
        let kind;
        unsafe {
            kind = pgrx::pg_sys::add_reloption_kind();
            pgrx::pg_sys::add_string_reloption(
                kind,
                c"options".as_ptr(),
                c"Vector index options, represented as a TOML string.".as_ptr(),
                c"".as_ptr(),
                None,
                pgrx::pg_sys::AccessExclusiveLock as pgrx::pg_sys::LOCKMODE,
            );
        }
        kind
    });
}

#[pgrx::pg_extern(sql = "")]
fn _vchordrq_amhandler(_fcinfo: pgrx::pg_sys::FunctionCallInfo) -> Internal {
    type T = pgrx::pg_sys::IndexAmRoutine;
    unsafe {
        let index_am_routine = pgrx::pg_sys::palloc0(size_of::<T>()) as *mut T;
        index_am_routine.write(AM_HANDLER);
        Internal::from(Some(Datum::from(index_am_routine)))
    }
}

const AM_HANDLER: pgrx::pg_sys::IndexAmRoutine = const {
    let mut am_routine = unsafe { std::mem::zeroed::<pgrx::pg_sys::IndexAmRoutine>() };

    am_routine.type_ = pgrx::pg_sys::NodeTag::T_IndexAmRoutine;

    am_routine.amsupport = 1;
    am_routine.amcanorderbyop = true;

    #[cfg(feature = "pg17")]
    {
        am_routine.amcanbuildparallel = true;
    }

    // Index access methods that set `amoptionalkey` to `false`
    // must index all tuples, even if the first column is `NULL`.
    // However, PostgreSQL does not generate a path if there is no
    // index clauses, even if there is a `ORDER BY` clause.
    // So we have to set it to `true` and set costs of every path
    // for vector index scans without `ORDER BY` clauses a large number
    // and throw errors if someone really wants such a path.
    am_routine.amoptionalkey = true;

    am_routine.amvalidate = Some(amvalidate);
    am_routine.amoptions = Some(amoptions);
    am_routine.amcostestimate = Some(amcostestimate);

    am_routine.ambuild = Some(am_build::ambuild);
    am_routine.ambuildempty = Some(am_build::ambuildempty);
    am_routine.aminsert = Some(aminsert);
    am_routine.ambulkdelete = Some(ambulkdelete);
    am_routine.amvacuumcleanup = Some(amvacuumcleanup);

    am_routine.ambeginscan = Some(ambeginscan);
    am_routine.amrescan = Some(amrescan);
    am_routine.amgettuple = Some(amgettuple);
    am_routine.amendscan = Some(amendscan);

    am_routine
};

#[pgrx::pg_guard]
pub unsafe extern "C" fn amvalidate(_opclass_oid: pgrx::pg_sys::Oid) -> bool {
    true
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amoptions(reloptions: Datum, validate: bool) -> *mut pgrx::pg_sys::bytea {
    let relopt_kind = RELOPT_KIND.get().copied().expect("init is not called");
    let rdopts = unsafe {
        pgrx::pg_sys::build_reloptions(
            reloptions,
            validate,
            relopt_kind,
            size_of::<Reloption>(),
            TABLE.as_ptr(),
            TABLE.len() as _,
        )
    };
    rdopts as *mut pgrx::pg_sys::bytea
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amcostestimate(
    _root: *mut pgrx::pg_sys::PlannerInfo,
    path: *mut pgrx::pg_sys::IndexPath,
    _loop_count: f64,
    index_startup_cost: *mut pgrx::pg_sys::Cost,
    index_total_cost: *mut pgrx::pg_sys::Cost,
    index_selectivity: *mut pgrx::pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    unsafe {
        if (*path).indexorderbys.is_null() && (*path).indexclauses.is_null() {
            *index_startup_cost = f64::MAX;
            *index_total_cost = f64::MAX;
            *index_selectivity = 0.0;
            *index_correlation = 0.0;
            *index_pages = 0.0;
            return;
        }
        *index_startup_cost = 0.0;
        *index_total_cost = 0.0;
        *index_selectivity = 1.0;
        *index_correlation = 1.0;
        *index_pages = 0.0;
    }
}

#[cfg(feature = "pg13")]
#[pgrx::pg_guard]
pub unsafe extern "C" fn aminsert(
    index_relation: pgrx::pg_sys::Relation,
    values: *mut Datum,
    is_null: *mut bool,
    heap_tid: pgrx::pg_sys::ItemPointer,
    _heap_relation: pgrx::pg_sys::Relation,
    _check_unique: pgrx::pg_sys::IndexUniqueCheck::Type,
    _index_info: *mut pgrx::pg_sys::IndexInfo,
) -> bool {
    unsafe { aminsertinner(index_relation, values, is_null, heap_tid) }
}

#[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16", feature = "pg17"))]
#[pgrx::pg_guard]
pub unsafe extern "C" fn aminsert(
    index_relation: pgrx::pg_sys::Relation,
    values: *mut Datum,
    is_null: *mut bool,
    heap_tid: pgrx::pg_sys::ItemPointer,
    _heap_relation: pgrx::pg_sys::Relation,
    _check_unique: pgrx::pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pgrx::pg_sys::IndexInfo,
) -> bool {
    unsafe { aminsertinner(index_relation, values, is_null, heap_tid) }
}

unsafe fn aminsertinner(
    index_relation: pgrx::pg_sys::Relation,
    values: *mut Datum,
    is_null: *mut bool,
    ctid: pgrx::pg_sys::ItemPointer,
) -> bool {
    let opfamily = unsafe { opfamily(index_relation) };
    let index = unsafe { PostgresRelation::new(index_relation) };
    let datum = unsafe { (!is_null.add(0).read()).then_some(values.add(0).read()) };
    let ctid = unsafe { ctid.read() };
    if let Some(store) = unsafe { datum.and_then(|x| opfamily.store(x)) } {
        for (vector, extra) in store {
            let payload = ctid_to_pointer(ctid, extra);
            crate::index::algorithm::insert(opfamily, index.clone(), payload, vector);
        }
    }
    false
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn ambulkdelete(
    info: *mut pgrx::pg_sys::IndexVacuumInfo,
    stats: *mut pgrx::pg_sys::IndexBulkDeleteResult,
    callback: pgrx::pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut std::os::raw::c_void,
) -> *mut pgrx::pg_sys::IndexBulkDeleteResult {
    let mut stats = stats;
    if stats.is_null() {
        stats = unsafe {
            pgrx::pg_sys::palloc0(size_of::<pgrx::pg_sys::IndexBulkDeleteResult>()).cast()
        };
    }
    let opfamily = unsafe { opfamily((*info).index) };
    let index = unsafe { PostgresRelation::new((*info).index) };
    let check = || unsafe {
        pgrx::pg_sys::vacuum_delay_point();
    };
    let callback = callback.expect("null function pointer");
    let callback = |p: NonZeroU64| unsafe { callback(&mut pointer_to_ctid(p).0, callback_state) };
    crate::index::algorithm::bulkdelete(opfamily, index, check, callback);
    stats
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amvacuumcleanup(
    info: *mut pgrx::pg_sys::IndexVacuumInfo,
    stats: *mut pgrx::pg_sys::IndexBulkDeleteResult,
) -> *mut pgrx::pg_sys::IndexBulkDeleteResult {
    let mut stats = stats;
    if stats.is_null() {
        stats = unsafe {
            pgrx::pg_sys::palloc0(size_of::<pgrx::pg_sys::IndexBulkDeleteResult>()).cast()
        };
    }
    let opfamily = unsafe { opfamily((*info).index) };
    let index = unsafe { PostgresRelation::new((*info).index) };
    let check = || unsafe {
        pgrx::pg_sys::vacuum_delay_point();
    };
    crate::index::algorithm::maintain(opfamily, index, check);
    stats
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn ambeginscan(
    index_relation: pgrx::pg_sys::Relation,
    n_keys: std::os::raw::c_int,
    n_orderbys: std::os::raw::c_int,
) -> pgrx::pg_sys::IndexScanDesc {
    use pgrx::memcxt::PgMemoryContexts::CurrentMemoryContext;

    let scan = unsafe { pgrx::pg_sys::RelationGetIndexScan(index_relation, n_keys, n_orderbys) };
    let scanner: Scanner = LazyCell::new(Box::new(|| Box::new(std::iter::empty())));
    unsafe {
        (*scan).opaque = CurrentMemoryContext.leak_and_drop_on_delete(scanner).cast();
    }
    scan
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amrescan(
    scan: pgrx::pg_sys::IndexScanDesc,
    keys: pgrx::pg_sys::ScanKey,
    _n_keys: std::os::raw::c_int,
    orderbys: pgrx::pg_sys::ScanKey,
    _n_orderbys: std::os::raw::c_int,
) {
    unsafe {
        use crate::index::opclass::Opfamily;
        if !keys.is_null() && (*scan).numberOfKeys > 0 {
            std::ptr::copy(keys, (*scan).keyData, (*scan).numberOfKeys as _);
        }
        if !orderbys.is_null() && (*scan).numberOfOrderBys > 0 {
            std::ptr::copy(orderbys, (*scan).orderByData, (*scan).numberOfOrderBys as _);
        }
        if (*scan).numberOfOrderBys == 0 && (*scan).numberOfKeys == 0 {
            pgrx::error!(
                "vector search with no WHERE clause and no ORDER BY clause is not supported"
            );
        }
        let opfamily = opfamily((*scan).indexRelation);
        let relation = PostgresRelation::new((*scan).indexRelation);
        let options = SearchOptions {
            epsilon: epsilon(),
            probes: probes(),
            max_scan_tuples: max_scan_tuples(),
            max_maxsim_tuples: max_maxsim_tuples(),
            maxsim_threshold: maxsim_threshold(),
        };
        let fetcher = LazyCell::new(move || {
            HeapFetcher::new(
                (*scan).indexRelation,
                (*scan).heapRelation,
                (*scan).xs_snapshot,
            )
        });
        let scanner = &mut *(*scan).opaque.cast::<Scanner>();
        *scanner = match opfamily {
            Opfamily::VectorL2
            | Opfamily::VectorIp
            | Opfamily::VectorCosine
            | Opfamily::HalfvecL2
            | Opfamily::HalfvecIp
            | Opfamily::HalfvecCosine => {
                let mut builder = DefaultBuilder::new(opfamily);
                for i in 0..(*scan).numberOfOrderBys {
                    let data = (*scan).orderByData.add(i as usize);
                    let value = (*data).sk_argument;
                    let is_null = ((*data).sk_flags & pgrx::pg_sys::SK_ISNULL as i32) != 0;
                    builder.add((*data).sk_strategy, (!is_null).then_some(value));
                }
                for i in 0..(*scan).numberOfKeys {
                    let data = (*scan).keyData.add(i as usize);
                    let value = (*data).sk_argument;
                    let is_null = ((*data).sk_flags & pgrx::pg_sys::SK_ISNULL as i32) != 0;
                    builder.add((*data).sk_strategy, (!is_null).then_some(value));
                }
                LazyCell::new(Box::new(move || builder.build(relation, options, fetcher)))
            }
            Opfamily::VectorMaxsimL2
            | Opfamily::VectorMaxsimIp
            | Opfamily::VectorMaxsimCosine
            | Opfamily::HalfvecMaxsimL2
            | Opfamily::HalfvecMaxsimIp
            | Opfamily::HalfvecMaxsimCosine => {
                let mut builder = MaxsimBuilder::new(opfamily);
                for i in 0..(*scan).numberOfOrderBys {
                    let data = (*scan).orderByData.add(i as usize);
                    let value = (*data).sk_argument;
                    let is_null = ((*data).sk_flags & pgrx::pg_sys::SK_ISNULL as i32) != 0;
                    builder.add((*data).sk_strategy, (!is_null).then_some(value));
                }
                for i in 0..(*scan).numberOfKeys {
                    let data = (*scan).keyData.add(i as usize);
                    let value = (*data).sk_argument;
                    let is_null = ((*data).sk_flags & pgrx::pg_sys::SK_ISNULL as i32) != 0;
                    builder.add((*data).sk_strategy, (!is_null).then_some(value));
                }
                LazyCell::new(Box::new(move || builder.build(relation, options, fetcher)))
            }
        };
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amgettuple(
    scan: pgrx::pg_sys::IndexScanDesc,
    direction: pgrx::pg_sys::ScanDirection::Type,
) -> bool {
    if direction != pgrx::pg_sys::ScanDirection::ForwardScanDirection {
        pgrx::error!("vector search without a forward scan direction is not supported");
    }
    // https://www.postgresql.org/docs/current/index-locking.html
    // If heap entries referenced physical pointers are deleted before
    // they are consumed by PostgreSQL, PostgreSQL will received wrong
    // physical pointers: no rows or irreverent rows are referenced.
    if unsafe { (*(*scan).xs_snapshot).snapshot_type } != pgrx::pg_sys::SnapshotType::SNAPSHOT_MVCC
    {
        pgrx::error!("scanning with a non-MVCC-compliant snapshot is not supported");
    }
    let scanner = unsafe { (*scan).opaque.cast::<Scanner>().as_mut().unwrap_unchecked() };
    if let Some((_, ctid, recheck)) = LazyCell::force_mut(scanner).next() {
        unsafe {
            (*scan).xs_heaptid = ctid;
            (*scan).xs_recheck = recheck;
            (*scan).xs_recheckorderby = false;
        }
        true
    } else {
        false
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amendscan(scan: pgrx::pg_sys::IndexScanDesc) {
    let scanner = unsafe { &mut *(*scan).opaque.cast::<Scanner>() };
    *scanner = LazyCell::new(Box::new(|| Box::new(std::iter::empty())));
}

type Iter = Box<dyn Iterator<Item = (f32, ItemPointerData, bool)>>;

type Scanner = LazyCell<Iter, Box<dyn FnOnce() -> Iter>>;

struct HeapFetcher {
    index_info: *mut pgrx::pg_sys::IndexInfo,
    estate: *mut pgrx::pg_sys::EState,
    econtext: *mut pgrx::pg_sys::ExprContext,
    heap_relation: pgrx::pg_sys::Relation,
    snapshot: pgrx::pg_sys::Snapshot,
    slot: *mut pgrx::pg_sys::TupleTableSlot,
    values: [Datum; 32],
    is_nulls: [bool; 32],
}

impl HeapFetcher {
    unsafe fn new(
        index_relation: pgrx::pg_sys::Relation,
        heap_relation: pgrx::pg_sys::Relation,
        snapshot: pgrx::pg_sys::Snapshot,
    ) -> Self {
        unsafe {
            let index_info = pgrx::pg_sys::BuildIndexInfo(index_relation);
            let estate = pgrx::pg_sys::CreateExecutorState();
            let econtext = pgrx::pg_sys::MakePerTupleExprContext(estate);
            Self {
                index_info,
                estate,
                econtext,
                heap_relation,
                snapshot,
                slot: pgrx::pg_sys::table_slot_create(heap_relation, std::ptr::null_mut()),
                values: [Datum::null(); 32],
                is_nulls: [true; 32],
            }
        }
    }
}

impl Drop for HeapFetcher {
    fn drop(&mut self) {
        unsafe {
            // free resources for last `fetch` call
            pgrx::pg_sys::MemoryContextReset((*self.econtext).ecxt_per_tuple_memory);
            // free common resources
            pgrx::pg_sys::ExecDropSingleTupleTableSlot(self.slot);
            pgrx::pg_sys::FreeExecutorState(self.estate);
        }
    }
}

impl SearchFetcher for HeapFetcher {
    fn fetch(&mut self, mut ctid: ItemPointerData) -> Option<(&[Datum; 32], &[bool; 32])> {
        unsafe {
            // free resources for last `fetch` call
            pgrx::pg_sys::MemoryContextReset((*self.econtext).ecxt_per_tuple_memory);
            // perform operation
            (*self.econtext).ecxt_scantuple = self.slot;
            let table_am = (*self.heap_relation).rd_tableam;
            let fetch_row_version = (*table_am)
                .tuple_fetch_row_version
                .expect("unsupported heap access method");
            if !fetch_row_version(self.heap_relation, &mut ctid, self.snapshot, self.slot) {
                return None;
            }
            pgrx::pg_sys::FormIndexDatum(
                self.index_info,
                self.slot,
                self.estate,
                self.values.as_mut_ptr(),
                self.is_nulls.as_mut_ptr(),
            );
            Some((&self.values, &self.is_nulls))
        }
    }
}

pub const fn pointer_to_ctid(pointer: NonZeroU64) -> (ItemPointerData, u16) {
    let value = pointer.get();
    let bi_hi = ((value >> 48) & 0xffff) as u16;
    let bi_lo = ((value >> 32) & 0xffff) as u16;
    let ip_posid = ((value >> 16) & 0xffff) as u16;
    let extra = value as u16;
    (
        ItemPointerData {
            ip_blkid: pgrx::pg_sys::BlockIdData { bi_hi, bi_lo },
            ip_posid,
        },
        extra,
    )
}

pub const fn ctid_to_pointer(ctid: ItemPointerData, extra: u16) -> NonZeroU64 {
    let mut value = 0;
    value |= (ctid.ip_blkid.bi_hi as u64) << 48;
    value |= (ctid.ip_blkid.bi_lo as u64) << 32;
    value |= (ctid.ip_posid as u64) << 16;
    value |= extra as u64;
    NonZeroU64::new(value).expect("invalid pointer")
}

#[test]
const fn soundness_check() {
    let bi_hi = 1;
    let bi_lo = 2;
    let ip_posid = 3;
    let extra = 7;
    let r = pointer_to_ctid(ctid_to_pointer(
        ItemPointerData {
            ip_blkid: pgrx::pg_sys::BlockIdData { bi_hi, bi_lo },
            ip_posid,
        },
        extra,
    ));
    assert!(r.0.ip_blkid.bi_hi == bi_hi);
    assert!(r.0.ip_blkid.bi_lo == bi_lo);
    assert!(r.0.ip_posid == ip_posid);
    assert!(r.1 == extra);
}
