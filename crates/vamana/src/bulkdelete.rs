use crate::Opaque;
use crate::operator::Operator;
use algo::{Page, RelationLength, RelationRead, RelationWrite};
use std::num::NonZero;

pub fn bulkdelete<R: RelationRead + RelationWrite + RelationLength, O: Operator>(
    index: &R,
    _check: impl Fn(),
    _callback: impl Fn(NonZero<u64>) -> bool,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let _n = index.len();
    unimplemented!()
}
