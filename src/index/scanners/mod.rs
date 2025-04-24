mod default;
mod maxsim;

use super::opclass::Opfamily;
use crate::index::lazy_cell::LazyCell;
use algorithm::{RelationPrefetch, RelationReadStream};
use pgrx::pg_sys::Datum;

pub use default::DefaultBuilder;
pub use maxsim::MaxsimBuilder;

#[derive(Debug)]
pub struct SearchOptions {
    pub epsilon: f32,
    pub probes: Vec<u32>,
    pub max_scan_tuples: Option<u32>,
    pub maxsim_refine: u32,
    pub maxsim_threshold: u32,
}

pub trait SearchBuilder: 'static {
    fn new(opfamily: Opfamily) -> Self;

    unsafe fn add(&mut self, strategy: u16, datum: Option<Datum>);

    fn build<'a>(
        self,
        relation: &'a (impl RelationPrefetch + RelationReadStream),
        options: SearchOptions,
        fetcher: impl SearchFetcher + 'a,
        bump: &'a bumpalo::Bump,
    ) -> Box<dyn Iterator<Item = (f32, [u16; 3], bool)> + 'a>;
}

pub trait SearchFetcher {
    fn fetch(&mut self, ctid: [u16; 3]) -> Option<(&[Datum; 32], &[bool; 32])>;
}

impl<T: SearchFetcher, F: FnOnce() -> T> SearchFetcher for LazyCell<T, F> {
    fn fetch(&mut self, key: [u16; 3]) -> Option<(&[Datum; 32], &[bool; 32])> {
        LazyCell::force_mut(self).fetch(key)
    }
}
