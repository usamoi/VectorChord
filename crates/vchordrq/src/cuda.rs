#![allow(unsafe_code)]

use crate::operator::{Operator, Vector};
use crate::tuples::{AppendableTuple, JumpTuple, MetaTuple, Tuple, WithReader};
use crate::{Opaque, Page, tape, vectors};
use algo::accessor::FunctionalAccessor;
use algo::{RelationRead, RelationWrite};
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

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
        pub op: op_t::Type,
        pub d: usize,
        pub n: usize,
        pub centroids: *mut std::ffi::c_void,
    }

    #[repr(C)]
    pub struct client_t {
        pub stream: *mut std::ffi::c_void,
        pub op: op_t::Type,
        pub d: usize,
        pub n: usize,
        pub centroids: *mut std::ffi::c_void,
        pub m: usize,
        pub vectors: *mut std::ffi::c_void,
        pub buffer: *mut std::ffi::c_void,
        pub results: *mut u32,
    }

    unsafe extern "C" {
        pub unsafe fn vchordrq_assign_server_alloc(
            op: op_t::Type,
            d: usize,
            n: usize,
            centroids: *mut std::ffi::c_void,
        ) -> *mut server_t;
        pub unsafe fn vchordrq_assign_server_free(server: *mut server_t);
        pub unsafe fn vchordrq_assign_client_alloc(
            op: op_t::Type,
            d: usize,
            n: usize,
            centroids: *mut std::ffi::c_void,
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

pub mod rpc {
    use bincode::{Decode, Encode};

    #[derive(Encode, Decode)]
    pub struct InitRequest {
        pub op: crate::cuda::ffi::op_t::Type,
        pub d: usize,
        pub n: usize,
        pub centroids: Vec<u8>,
    }

    #[derive(Encode, Decode)]
    pub enum InitResponse {
        Ok {},
        Err { msg: String },
    }

    #[derive(Encode, Decode)]
    pub struct ConnectRequest {
        pub m: usize,
    }

    #[derive(Encode, Decode)]
    pub enum ConnectResponse {
        Ok {},
        Err { msg: String },
    }

    #[derive(Encode, Decode)]
    pub struct QueryRequest {
        pub k: usize,
        pub vectors: Vec<u8>,
    }

    #[derive(Encode, Decode)]
    pub enum QueryResponse {
        Ok { result: Vec<u32> },
        Err { msg: String },
    }

    #[derive(Encode, Decode)]
    pub enum Request {
        Init(InitRequest),
        Connect(ConnectRequest),
        Query(QueryRequest),
    }

    pub const CONFIG: bincode::config::Configuration = bincode::config::standard();
}
