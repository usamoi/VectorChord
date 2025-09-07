#![allow(unsafe_code)]

use crate::assign::ffi::size;
use crate::operator::{Operator, Vector};
use crate::tuples::{AppendableTuple, JumpTuple, MetaTuple, Tuple, WithReader};
use crate::{Opaque, Page, tape, vectors};
use algo::accessor::FunctionalAccessor;
use algo::{RelationRead, RelationWrite};
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub fn assign<'b, R: RelationRead + RelationWrite, O: Operator>(
    index: &'b R,
    payload: NonZero<u64>,
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    key: (Vec<u32>, u16),
    skip_freespaces: bool,
    labels: &[u32],
    best: u32,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let is_residual = meta_tuple.is_residual();
    let freepages_first = meta_tuple.freepages_first();
    assert_eq!(dims, vector.dims(), "unmatched dimensions");

    let first = labels[best as usize];

    let jump_guard = index.read(first);
    let jump_bytes = jump_guard.get(1).expect("data corruption");
    let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);

    let (code, delta) = O::build(
        vector,
        is_residual.then(|| {
            vectors::read_for_h1_tuple::<R, O, _>(
                jump_tuple
                    .centroid_prefetch()
                    .iter()
                    .map(|&id| index.read(id)),
                jump_tuple.centroid_head(),
                FunctionalAccessor::new(Vec::new(), Vec::extend_from_slice, O::Vector::pack),
            )
        }),
    );

    let (prefetch, head) = key;
    let serialized = AppendableTuple::serialize(&AppendableTuple {
        metadata: [
            code.0.dis_u_2,
            code.0.factor_cnt,
            code.0.factor_ip,
            code.0.factor_err,
        ],
        delta,
        payload: Some(payload),
        prefetch,
        head,
        elements: rabitq::bit::binary::pack_code(&code.1),
    });

    tape::append(
        index,
        jump_tuple.appendable_first(),
        &serialized,
        false,
        (!skip_freespaces).then_some(freepages_first),
    );
}

#[allow(non_camel_case_types)]
#[allow(non_upper_case_globals)]
pub mod ffi {
    #[repr(C)]
    pub struct cudaIpcMemHandle_t {
        pub reserved: [u8; 64],
    }

    pub mod op_t {
        pub type Type = std::ffi::c_uint;
        pub const vecf32_dot: Type = 0;
        pub const vecf32_l2s: Type = 1;
        pub const vecf16_dot: Type = 2;
        pub const vecf16_l2s: Type = 3;
    }

    pub fn size(op: op_t::Type) -> usize {
        match op {
            op_t::vecf32_dot => 4,
            op_t::vecf32_l2s => 4,
            op_t::vecf16_dot => 2,
            op_t::vecf16_l2s => 2,
            _ => 0,
        }
    }

    #[repr(C)]
    pub struct server_t {
        op: op_t::Type,
        d: usize,
        t: usize,
        centroids: *mut std::ffi::c_void,
    }

    #[repr(C)]
    pub struct client_t {
        tag: i32,
        op: op_t::Type,
        d: usize,
        n: usize,
        centroids: *mut std::ffi::c_void,
        m: usize,
        vectors: *mut std::ffi::c_void,
        buffer: *mut std::ffi::c_void,
        results: *mut u32,
    }

    unsafe extern "C" {
        pub unsafe fn vchordrq_assign_server_alloc(
            op: op_t::Type,
            d: usize,
            n: usize,
            centroids: *mut std::ffi::c_void,
        ) -> *mut server_t;
        pub unsafe fn vchordrq_assign_server_free(server: *mut server_t);
        pub unsafe fn vchordrq_assign_server_addr(
            server: *mut server_t,
            op: *mut op_t::Type,
            d: *mut usize,
            n: *mut usize,
            centroids_0: *mut *mut std::ffi::c_void,
            centroids_1: *mut cudaIpcMemHandle_t,
        ) -> std::ffi::c_int;
        pub unsafe fn vchordrq_assign_client_alloc_0(
            op: op_t::Type,
            d: usize,
            n: usize,
            centroids_0: *mut std::ffi::c_void,
            m: usize,
        ) -> *mut client_t;
        pub unsafe fn vchordrq_assign_client_alloc_1(
            op: op_t::Type,
            d: usize,
            n: usize,
            centroids_1: cudaIpcMemHandle_t,
            m: usize,
        ) -> *mut client_t;
        pub unsafe fn vchordrq_assign_client_free(client: *mut client_t);
        pub unsafe fn vchordrq_assign_client_query(
            client: *mut client_t,
            k: usize,
            vectors: *mut std::ffi::c_void,
            results: *mut u32,
        ) -> std::ffi::c_int;
    }
}

#[derive(Debug, Clone, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct Addr {
    pid: u32,
    op: u32,
    d: usize,
    n: usize,
    centroids_0: usize,
    centroids_1: [u8; 64],
}

pub struct Server {
    server: *mut ffi::server_t,
}

impl Server {
    pub unsafe fn new(op: ffi::op_t::Type, d: usize, n: usize, centroids: &[u8]) -> Option<Self> {
        unsafe {
            let server =
                ffi::vchordrq_assign_server_alloc(op, d, n, centroids.as_ptr().cast_mut().cast());
            if !server.is_null() {
                Some(Self { server })
            } else {
                None
            }
        }
    }
    pub fn addr(&mut self) -> Option<Addr> {
        unsafe {
            let mut op = std::mem::zeroed();
            let mut d = std::mem::zeroed();
            let mut n = std::mem::zeroed();
            let mut centroids_0 = std::mem::zeroed();
            let mut centroids_1 = std::mem::zeroed();
            let r = ffi::vchordrq_assign_server_addr(
                self.server,
                &mut op,
                &mut d,
                &mut n,
                &mut centroids_0,
                &mut centroids_1,
            );
            if r == 0 {
                Some(Addr {
                    pid: std::process::id(),
                    op,
                    d,
                    n,
                    centroids_0: centroids_0.expose_provenance(),
                    centroids_1: centroids_1.reserved,
                })
            } else {
                return None;
            }
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        unsafe {
            ffi::vchordrq_assign_server_free(self.server);
        }
    }
}

pub struct Client {
    client: *mut ffi::client_t,
    results: Vec<u32>,
    op: ffi::op_t::Type,
    d: usize,
}

impl Client {
    pub unsafe fn new(addr: Addr, m: usize) -> Option<Self> {
        let client = unsafe {
            if addr.pid == std::process::id() {
                ffi::vchordrq_assign_client_alloc_0(
                    addr.op,
                    addr.d,
                    addr.n,
                    std::ptr::with_exposed_provenance_mut(addr.centroids_0),
                    m,
                )
            } else {
                ffi::vchordrq_assign_client_alloc_1(
                    addr.op,
                    addr.d,
                    addr.n,
                    ffi::cudaIpcMemHandle_t {
                        reserved: addr.centroids_1,
                    },
                    m,
                )
            }
        };
        let results = vec![u32::MAX; m];
        if !client.is_null() {
            Some(Self {
                client,
                results,
                op: addr.op,
                d: addr.d,
            })
        } else {
            None
        }
    }
    pub fn query(&mut self, vectors: &[u8]) -> Option<&mut [u32]> {
        unsafe {
            assert!(vectors.len().is_multiple_of(size(self.op)));
            let k = vectors.len() / self.d / size(self.op);
            assert!(k <= self.results.len());
            let r = ffi::vchordrq_assign_client_query(
                self.client,
                k,
                vectors.as_ptr().cast_mut().cast(),
                self.results.as_mut_ptr(),
            );
            if r == 0 {
                Some(&mut self.results[..k])
            } else {
                None
            }
        }
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        unsafe {
            ffi::vchordrq_assign_client_free(self.client);
        }
    }
}
