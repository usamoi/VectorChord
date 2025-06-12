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

use pgrx::PostgresGucEnum;
use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PostgresGucEnum)]
pub enum PostgresIo {
    read_buffer,
    prefetch_buffer,
    #[cfg(feature = "pg17")]
    read_stream,
}

static EF_SEARCH: GucSetting<i32> = GucSetting::<i32>::new(64);
static MAX_SCAN_TUPLES: GucSetting<i32> = GucSetting::<i32>::new(-1);

pub fn init() {
    GucRegistry::define_int_guc(
        "vamana.ef_search",
        "`ef_search` argument of vamana.",
        "`ef_search` argument of vamana.",
        &EF_SEARCH,
        1,
        65535,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        "vamana.max_scan_tuples",
        "`max_scan_tuples` argument of vamana.",
        "`max_scan_tuples` argument of vamana.",
        &MAX_SCAN_TUPLES,
        -1,
        i32::MAX,
        GucContext::Userset,
        GucFlags::default(),
    );
    unsafe {
        #[cfg(any(feature = "pg13", feature = "pg14"))]
        pgrx::pg_sys::EmitWarningsOnPlaceholders(c"vamana".as_ptr());
        #[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17"))]
        pgrx::pg_sys::MarkGUCPrefixReserved(c"vamana".as_ptr());
    }
}

pub fn ef_search() -> u32 {
    EF_SEARCH.get() as u32
}

pub fn max_scan_tuples() -> Option<u32> {
    let x = MAX_SCAN_TUPLES.get();
    if x < 0 { None } else { Some(x as u32) }
}
