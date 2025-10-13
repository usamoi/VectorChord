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

use crate::index::vchordrq::am::PostgresRelation;
use crate::index::vchordrq::opclass::opfamily;
use std::ffi::CStr;
use vchordrq::MaintainChooser;

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amvacuumcleanup(
    info: *mut pgrx::pg_sys::IndexVacuumInfo,
    stats: *mut pgrx::pg_sys::IndexBulkDeleteResult,
) -> *mut pgrx::pg_sys::IndexBulkDeleteResult {
    let mut stats = stats;
    if stats.is_null() {
        stats = unsafe {
            pgrx::pg_sys::palloc0(size_of::<pgrx::pg_sys::IndexBulkDeleteResult>()).cast()
        };
    }
    let index_relation = unsafe { (*info).index };
    if let Some(leader) =
        unsafe { VchordrqLeader::enter(c"vchordrq_parallel_vacuumcleanup_main", index_relation) }
    {
        unsafe {
            leader.wait();
            parallel_vacuumcleanup(
                index_relation,
                leader.vchordrqshared,
                || {
                    #[allow(clippy::needless_late_init)]
                    let order;
                    // enter the barrier
                    let shared = leader.vchordrqshared;
                    pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                    (*shared).nparticipants = leader.nparticipants as u32;
                    order = (*shared).barrier_enter_0 as u32;
                    (*shared).barrier_enter_0 += 1;
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                    pgrx::pg_sys::ConditionVariableBroadcast(
                        &raw mut (*shared).condvar_barrier_enter_0,
                    );
                    // leave the barrier
                    let total = leader.nparticipants;
                    loop {
                        pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                        if (*shared).barrier_enter_0 == total {
                            pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                            break;
                        }
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                        pgrx::pg_sys::ConditionVariableSleep(
                            &raw mut (*shared).condvar_barrier_enter_0,
                            pgrx::pg_sys::WaitEventIPC::WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN as _,
                        );
                    }
                    pgrx::pg_sys::ConditionVariableCancelSleep();
                    pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                    (*shared).barrier_leave_0 = true;
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                    pgrx::pg_sys::ConditionVariableBroadcast(
                        &raw mut (*shared).condvar_barrier_leave_0,
                    );
                    order
                },
                || {
                    // enter the barrier
                    let shared = leader.vchordrqshared;
                    pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                    (*shared).barrier_enter_1 += 1;
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                    pgrx::pg_sys::ConditionVariableBroadcast(
                        &raw mut (*shared).condvar_barrier_enter_1,
                    );
                    // leave the barrier
                    let total = leader.nparticipants;
                    loop {
                        pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                        if (*shared).barrier_enter_1 == total {
                            pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                            break;
                        }
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                        pgrx::pg_sys::ConditionVariableSleep(
                            &raw mut (*shared).condvar_barrier_enter_1,
                            pgrx::pg_sys::WaitEventIPC::WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN as _,
                        );
                    }
                    pgrx::pg_sys::ConditionVariableCancelSleep();
                    pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                    (*shared).barrier_leave_1 = true;
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                    pgrx::pg_sys::ConditionVariableBroadcast(
                        &raw mut (*shared).condvar_barrier_leave_1,
                    );
                },
            );
        }
    } else {
        unsafe {
            sequential_vacuumcleanup(index_relation, || (), || ());
        }
    }
    stats
}

struct VchordrqShared {
    /* immutable state */
    indexrelid: pgrx::pg_sys::Oid,

    /* locking */
    mutex: pgrx::pg_sys::slock_t,
    condvar_barrier_enter_0: pgrx::pg_sys::ConditionVariable,
    condvar_barrier_leave_0: pgrx::pg_sys::ConditionVariable,
    condvar_barrier_enter_1: pgrx::pg_sys::ConditionVariable,
    condvar_barrier_leave_1: pgrx::pg_sys::ConditionVariable,

    /* mutable state */
    barrier_enter_0: i32,
    nparticipants: u32,
    barrier_leave_0: bool,
    barrier_enter_1: i32,
    barrier_leave_1: bool,
}

struct VchordrqLeader {
    pcxt: *mut pgrx::pg_sys::ParallelContext,
    nparticipants: i32,
    vchordrqshared: *mut VchordrqShared,
}

impl VchordrqLeader {
    pub unsafe fn enter(
        main: &'static CStr,
        index_relation: pgrx::pg_sys::Relation,
    ) -> Option<Self> {
        unsafe fn compute_parallel_workers(_index_relation: pgrx::pg_sys::Relation) -> i32 {
            crate::index::gucs::vchordrq_max_parallel_vacuum_workers()
        }

        let request = unsafe { compute_parallel_workers(index_relation) };
        if request <= 0 {
            return None;
        }

        unsafe {
            pgrx::pg_sys::EnterParallelMode();
        }
        let pcxt = unsafe {
            pgrx::pg_sys::CreateParallelContext(c"vchord".as_ptr(), main.as_ptr(), request)
        };

        fn estimate_chunk(e: &mut pgrx::pg_sys::shm_toc_estimator, x: usize) {
            e.space_for_chunks += x.next_multiple_of(pgrx::pg_sys::ALIGNOF_BUFFER as _);
        }
        fn estimate_keys(e: &mut pgrx::pg_sys::shm_toc_estimator, x: usize) {
            e.number_of_keys += x;
        }
        unsafe {
            estimate_chunk(&mut (*pcxt).estimator, size_of::<VchordrqShared>());
            estimate_keys(&mut (*pcxt).estimator, 1);
        }

        unsafe {
            pgrx::pg_sys::InitializeParallelDSM(pcxt);
            if (*pcxt).seg.is_null() {
                pgrx::pg_sys::DestroyParallelContext(pcxt);
                pgrx::pg_sys::ExitParallelMode();
                return None;
            }
        }

        let vchordrqshared = unsafe {
            let vchordrqshared =
                pgrx::pg_sys::shm_toc_allocate((*pcxt).toc, size_of::<VchordrqShared>())
                    .cast::<VchordrqShared>();
            vchordrqshared.write(VchordrqShared {
                indexrelid: (*index_relation).rd_id,
                nparticipants: 0,
                condvar_barrier_enter_0: std::mem::zeroed(),
                condvar_barrier_leave_0: std::mem::zeroed(),
                condvar_barrier_enter_1: std::mem::zeroed(),
                condvar_barrier_leave_1: std::mem::zeroed(),
                barrier_enter_0: 0,
                barrier_leave_0: false,
                barrier_enter_1: 0,
                barrier_leave_1: false,
                mutex: std::mem::zeroed(),
            });
            pgrx::pg_sys::ConditionVariableInit(&raw mut (*vchordrqshared).condvar_barrier_enter_0);
            pgrx::pg_sys::ConditionVariableInit(&raw mut (*vchordrqshared).condvar_barrier_leave_0);
            pgrx::pg_sys::ConditionVariableInit(&raw mut (*vchordrqshared).condvar_barrier_enter_1);
            pgrx::pg_sys::ConditionVariableInit(&raw mut (*vchordrqshared).condvar_barrier_leave_1);
            pgrx::pg_sys::SpinLockInit(&raw mut (*vchordrqshared).mutex);
            vchordrqshared
        };

        unsafe {
            pgrx::pg_sys::shm_toc_insert((*pcxt).toc, 0xA000000000000001, vchordrqshared.cast());
        }

        unsafe {
            pgrx::pg_sys::LaunchParallelWorkers(pcxt);
        }

        let nworkers_launched = unsafe { (*pcxt).nworkers_launched };

        unsafe {
            if nworkers_launched == 0 {
                pgrx::pg_sys::WaitForParallelWorkersToFinish(pcxt);
                pgrx::pg_sys::DestroyParallelContext(pcxt);
                pgrx::pg_sys::ExitParallelMode();
                return None;
            }
        }

        Some(Self {
            pcxt,
            nparticipants: nworkers_launched + 1,
            vchordrqshared,
        })
    }

    pub fn wait(&self) {
        unsafe {
            pgrx::pg_sys::WaitForParallelWorkersToAttach(self.pcxt);
        }
    }
}

impl Drop for VchordrqLeader {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            unsafe {
                pgrx::pg_sys::WaitForParallelWorkersToFinish(self.pcxt);
                pgrx::pg_sys::DestroyParallelContext(self.pcxt);
                pgrx::pg_sys::ExitParallelMode();
            }
        }
    }
}

#[pgrx::pg_guard]
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vchordrq_parallel_vacuumcleanup_main(
    _seg: *mut pgrx::pg_sys::dsm_segment,
    toc: *mut pgrx::pg_sys::shm_toc,
) {
    let _ = rand::rng().reseed();
    let vchordrqshared = unsafe {
        pgrx::pg_sys::shm_toc_lookup(toc, 0xA000000000000001, false).cast::<VchordrqShared>()
    };
    let index_lockmode = pgrx::pg_sys::RowExclusiveLock as pgrx::pg_sys::LOCKMODE;
    let index = unsafe { pgrx::pg_sys::index_open((*vchordrqshared).indexrelid, index_lockmode) };

    unsafe {
        parallel_vacuumcleanup(
            index,
            vchordrqshared,
            || {
                #[allow(clippy::needless_late_init)]
                let order;
                // enter the barrier
                let shared = vchordrqshared;
                pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                order = (*shared).barrier_enter_0 as u32;
                (*shared).barrier_enter_0 += 1;
                pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                pgrx::pg_sys::ConditionVariableBroadcast(
                    &raw mut (*shared).condvar_barrier_enter_0,
                );
                // leave the barrier
                loop {
                    pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                    if (*shared).barrier_leave_0 {
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                        break;
                    }
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                    pgrx::pg_sys::ConditionVariableSleep(
                        &raw mut (*shared).condvar_barrier_leave_0,
                        pgrx::pg_sys::WaitEventIPC::WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN as _,
                    );
                }
                pgrx::pg_sys::ConditionVariableCancelSleep();
                order
            },
            || {
                // enter the barrier
                let shared = vchordrqshared;
                pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                (*shared).barrier_enter_1 += 1;
                pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                pgrx::pg_sys::ConditionVariableBroadcast(
                    &raw mut (*shared).condvar_barrier_enter_1,
                );
                // leave the barrier
                loop {
                    pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                    if (*shared).barrier_leave_1 {
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                        break;
                    }
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                    pgrx::pg_sys::ConditionVariableSleep(
                        &raw mut (*shared).condvar_barrier_leave_1,
                        pgrx::pg_sys::WaitEventIPC::WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN as _,
                    );
                }
                pgrx::pg_sys::ConditionVariableCancelSleep();
            },
        );
    }

    unsafe {
        pgrx::pg_sys::index_close(index, index_lockmode);
    }
}

unsafe fn parallel_vacuumcleanup(
    index_relation: pgrx::pg_sys::Relation,
    vchordrqshared: *mut VchordrqShared,
    sync_0: impl FnOnce() -> u32,
    sync_1: impl FnOnce(),
) {
    let opfamily = unsafe { opfamily(index_relation) };
    let index = unsafe { PostgresRelation::new(index_relation) };
    let check = || unsafe {
        #[cfg(any(
            feature = "pg13",
            feature = "pg14",
            feature = "pg15",
            feature = "pg16",
            feature = "pg17"
        ))]
        pgrx::pg_sys::vacuum_delay_point();
        #[cfg(feature = "pg18")]
        pgrx::pg_sys::vacuum_delay_point(false);
    };

    let order = sync_0();

    struct ChooseSome {
        n: usize,
        k: usize,
    }
    impl MaintainChooser for ChooseSome {
        fn choose(&mut self, i: usize) -> bool {
            i % self.n == self.k
        }
    }

    let mut chooser = ChooseSome {
        n: unsafe { (*vchordrqshared).nparticipants as usize },
        k: order as usize,
    };
    crate::index::vchordrq::dispatch::maintain(opfamily, &index, &mut chooser, check);

    sync_1();
}

unsafe fn sequential_vacuumcleanup(
    index_relation: pgrx::pg_sys::Relation,
    sync_0: impl FnOnce(),
    sync_1: impl FnOnce(),
) {
    struct ChooseAll;
    impl MaintainChooser for ChooseAll {
        fn choose(&mut self, _: usize) -> bool {
            true
        }
    }

    let opfamily = unsafe { opfamily(index_relation) };
    let index = unsafe { PostgresRelation::new(index_relation) };
    let check = || unsafe {
        #[cfg(any(
            feature = "pg13",
            feature = "pg14",
            feature = "pg15",
            feature = "pg16",
            feature = "pg17"
        ))]
        pgrx::pg_sys::vacuum_delay_point();
        #[cfg(feature = "pg18")]
        pgrx::pg_sys::vacuum_delay_point(false);
    };

    sync_0();

    let mut chooser = ChooseAll;
    crate::index::vchordrq::dispatch::maintain(opfamily, &index, &mut chooser, check);

    sync_1();
}
