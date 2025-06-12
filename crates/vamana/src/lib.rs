mod build;
mod bulkdelete;
mod checker;
mod insert;
mod maintain;
mod prune;
mod search;
mod tuples;
mod visited;

pub mod operator;
pub mod types;

pub use build::build;
pub use bulkdelete::bulkdelete;
pub use insert::insert;
pub use search::search;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct Opaque {}

#[allow(unsafe_code)]
unsafe impl algo::Opaque for Opaque {}

pub type Id = (u32, u16);
