use crate::Opaque;
use crate::operator::Operator;
use crate::tuples::{VertexTuple, WithWriter};
use algo::{Page, RelationLength, RelationRead, RelationWrite};
use std::num::NonZero;

#[allow(clippy::collapsible_if)]
pub fn bulkdelete<R: RelationRead + RelationWrite + RelationLength, O: Operator>(
    index: &R,
    check: impl Fn(),
    callback: impl Fn(NonZero<u64>) -> bool,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let n = index.len();
    for id in 1..n {
        check();
        let mut guard = index.write(id, true);
        for i in 1..=guard.len() {
            if let Some(bytes) = guard.get_mut(i) {
                let mut tuple = VertexTuple::deserialize_mut(bytes);
                if let Some(payload) = tuple.payload() {
                    if callback(*payload) {
                        tuple.payload().take();
                    }
                }
            }
        }
    }
}
