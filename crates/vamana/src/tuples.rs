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

use algo::tuples::{MutChecker, RefChecker};
use distance::Distance;
use std::num::NonZero;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub const ALIGN: usize = 8;
pub type Tag = u64;
const MAGIC: Tag = Tag::from_ne_bytes(*b"vcvamana");
const VERSION: u64 = 10;

pub trait Tuple: 'static {
    fn serialize(&self) -> Vec<u8>;
}

pub trait WithReader: Tuple {
    type Reader<'a>;
    fn deserialize_ref(source: &[u8]) -> Self::Reader<'_>;
}

pub trait WithWriter: Tuple {
    type Writer<'a>;
    fn deserialize_mut(source: &mut [u8]) -> Self::Writer<'_>;
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct MetaTupleHeader {
    version: u64,
    dims: u32,
    rerank_in_heap: Bool,
    _padding_0: [ZeroU8; 3],
    ef_construction: u32,
    m: u32,
    start: Start,
}

pub struct MetaTuple {
    pub dims: u32,
    pub rerank_in_heap: bool,
    pub ef_construction: u32,
    pub m: u32,
    pub start: Start,
}

impl Tuple for MetaTuple {
    #[allow(clippy::match_single_binding)]
    fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        match self {
            MetaTuple {
                dims,
                rerank_in_heap,
                ef_construction,
                m,
                start,
            } => {
                buffer.extend((MAGIC as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<MetaTupleHeader>()));
                // header
                buffer[size_of::<Tag>()..][..size_of::<MetaTupleHeader>()].copy_from_slice(
                    MetaTupleHeader {
                        version: VERSION,
                        dims: *dims,
                        rerank_in_heap: (*rerank_in_heap).into(),
                        ef_construction: *ef_construction,
                        m: *m,
                        start: *start,
                        _padding_0: Default::default(),
                    }
                    .as_bytes(),
                );
            }
        }
        buffer
    }
}

impl WithReader for MetaTuple {
    type Reader<'a> = MetaTupleReader<'a>;
    fn deserialize_ref(source: &[u8]) -> MetaTupleReader<'_> {
        let tag = Tag::from_ne_bytes(std::array::from_fn(|i| source[i]));
        match tag {
            MAGIC => {
                let checker = RefChecker::new(source);
                if VERSION != *checker.prefix::<u64>(size_of::<Tag>()) {
                    panic!("deserialization: bad version number");
                }
                let header: &MetaTupleHeader = checker.prefix(size_of::<Tag>());
                MetaTupleReader { header }
            }
            _ => panic!("deserialization: bad magic number"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MetaTupleReader<'a> {
    header: &'a MetaTupleHeader,
}

impl<'a> MetaTupleReader<'a> {
    pub fn dims(self) -> u32 {
        self.header.dims
    }
    pub fn ef_construction(self) -> u32 {
        self.header.ef_construction
    }
    pub fn m(self) -> u32 {
        self.header.m
    }
    pub fn start(self) -> Start {
        self.header.start
    }
}

impl WithWriter for MetaTuple {
    type Writer<'a> = MetaTupleWriter<'a>;
    fn deserialize_mut(source: &mut [u8]) -> MetaTupleWriter<'_> {
        let tag = Tag::from_ne_bytes(std::array::from_fn(|i| source[i]));
        match tag {
            MAGIC => {
                let checker = RefChecker::new(source);
                if VERSION != *checker.prefix::<u64>(size_of::<Tag>()) {
                    panic!("deserialization: bad version number");
                }
                let mut checker = MutChecker::new(source);
                let header: &mut MetaTupleHeader = checker.prefix(size_of::<Tag>());
                MetaTupleWriter { header }
            }
            _ => panic!("deserialization: bad magic number"),
        }
    }
}

#[derive(Debug)]
pub struct MetaTupleWriter<'a> {
    header: &'a mut MetaTupleHeader,
}

impl<'a> MetaTupleWriter<'a> {
    pub fn start(&mut self) -> &mut Start {
        &mut self.header.start
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct VertexTupleHeader {
    metadata: [f32; 3],
    _padding_0: [ZeroU8; 4],
    elements_s: u16,
    elements_e: u16,
    neighbours_s: u16,
    neighbours_e: u16,
    payload: Option<NonZero<u64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VertexTuple {
    pub metadata: [f32; 3],
    pub elements: Vec<u8>,
    pub neighbours: Vec<Neighbour>,
    pub payload: Option<NonZero<u64>>,
}

impl Tuple for VertexTuple {
    fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        buffer.extend(std::iter::repeat_n(0, size_of::<VertexTupleHeader>()));
        // elements
        let elements_s = buffer.len() as u16;
        buffer.extend(self.elements.as_bytes());
        let elements_e = buffer.len() as u16;
        while buffer.len() % ALIGN != 0 {
            buffer.push(0);
        }
        // neighbours
        let neighbours_s = buffer.len() as u16;
        buffer.extend(self.neighbours.as_bytes());
        let neighbours_e = buffer.len() as u16;
        while buffer.len() % ALIGN != 0 {
            buffer.push(0);
        }
        // header
        buffer[..size_of::<VertexTupleHeader>()].copy_from_slice(
            VertexTupleHeader {
                metadata: self.metadata,
                _padding_0: Default::default(),
                elements_s,
                elements_e,
                neighbours_s,
                neighbours_e,
                payload: self.payload,
            }
            .as_bytes(),
        );
        buffer
    }
}

impl WithReader for VertexTuple {
    type Reader<'a> = VertexTupleReader<'a>;

    fn deserialize_ref(source: &[u8]) -> VertexTupleReader<'_> {
        let checker = RefChecker::new(source);
        let header: &VertexTupleHeader = checker.prefix(0_u16);
        let elements = checker.bytes(header.elements_s, header.elements_e);
        let neighbours = checker.bytes(header.neighbours_s, header.neighbours_e);
        VertexTupleReader {
            header,
            elements,
            neighbours,
        }
    }
}

impl WithWriter for VertexTuple {
    type Writer<'a> = VertexTupleWriter<'a>;

    fn deserialize_mut(source: &mut [u8]) -> VertexTupleWriter<'_> {
        let mut checker = MutChecker::new(source);
        let header: &mut VertexTupleHeader = checker.prefix(0_u16);
        let elements = checker.bytes(header.elements_s, header.elements_e);
        let neighbours = checker.bytes(header.neighbours_s, header.neighbours_e);
        VertexTupleWriter {
            header,
            elements,
            neighbours,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertexTupleReader<'a> {
    header: &'a VertexTupleHeader,
    elements: &'a [u8],
    neighbours: &'a [Neighbour],
}

impl<'a> VertexTupleReader<'a> {
    pub fn metadata(self) -> [f32; 3] {
        self.header.metadata
    }
    pub fn payload(self) -> Option<NonZero<u64>> {
        self.header.payload
    }
    pub fn elements(self) -> &'a [u8] {
        self.elements
    }
    pub fn neighbours(self) -> &'a [Neighbour] {
        self.neighbours
    }
}

#[derive(Debug)]
pub struct VertexTupleWriter<'a> {
    header: &'a mut VertexTupleHeader,
    #[expect(dead_code)]
    elements: &'a mut [u8],
    neighbours: &'a mut [Neighbour],
}

impl VertexTupleWriter<'_> {
    pub fn payload(&mut self) -> &mut Option<NonZero<u64>> {
        &mut self.header.payload
    }
    pub fn neighbours(&mut self) -> &mut [Neighbour] {
        self.neighbours
    }
}

#[repr(C)]
#[derive(
    Debug,
    Default,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    IntoBytes,
    FromBytes,
    Immutable,
    KnownLayout,
)]
pub struct Start {
    pub x: u32,
    pub y: u16,
    _padding_0: [ZeroU8; 2],
}

impl Start {
    pub const NULL: Self = Self {
        x: u32::MAX,
        y: 0,
        _padding_0: [ZeroU8(None); 2],
    };
    pub fn is_null(self) -> bool {
        self.x == u32::MAX
    }
    pub fn new((x, y): (u32, u16)) -> Self {
        Self {
            x,
            y,
            _padding_0: Default::default(),
        }
    }
    pub fn id(self) -> Option<(u32, u16)> {
        if self.x == u32::MAX {
            return None;
        }
        Some((self.x, self.y))
    }
}

#[repr(C)]
#[derive(
    Debug,
    Default,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    IntoBytes,
    FromBytes,
    Immutable,
    KnownLayout,
)]
pub struct Neighbour {
    pub d: Distance,
    pub x: u32,
    pub y: u16,
    _padding_0: [ZeroU8; 2],
}

impl Neighbour {
    pub const NULL: Self = Self {
        d: Distance::ZERO,
        x: u32::MAX,
        y: 0,
        _padding_0: [ZeroU8(None); 2],
    };
    pub fn new(d: Distance, (x, y): (u32, u16)) -> Self {
        Self {
            d,
            x,
            y,
            _padding_0: Default::default(),
        }
    }
    pub fn distance(self) -> Option<Distance> {
        if self.x == u32::MAX {
            return None;
        }
        Some(self.d)
    }
    pub fn id(self) -> Option<(u32, u16)> {
        if self.x == u32::MAX {
            return None;
        }
        Some((self.x, self.y))
    }
}

#[repr(transparent)]
#[derive(
    Debug,
    Default,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    IntoBytes,
    FromBytes,
    Immutable,
    KnownLayout,
)]
pub struct ZeroU8(Option<NonZero<u8>>);

#[repr(transparent)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    IntoBytes,
    FromBytes,
    Immutable,
    KnownLayout,
)]
pub struct Bool(u8);

impl Bool {
    pub const FALSE: Self = Self(0);
    pub const TRUE: Self = Self(1);
}

impl From<Bool> for bool {
    fn from(value: Bool) -> Self {
        value != Bool::FALSE
    }
}

impl From<bool> for Bool {
    fn from(value: bool) -> Self {
        if value { Self::TRUE } else { Self::FALSE }
    }
}
