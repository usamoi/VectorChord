use crate::Opaque;
use crate::operator::Operator;
use crate::tuples::{VertexTuple, WithReader, WithWriter};
use algo::{Page, RelationLength, RelationRead, RelationWrite};
use std::num::NonZero;

pub fn bulkdelete<R: RelationRead + RelationWrite + RelationLength, O: Operator>(
    index: &R,
    check: impl Fn(),
    callback: impl Fn(NonZero<u64>) -> bool,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let link = meta_guard.get_opaque().link;
    drop(meta_guard);
    let mut current = link;
    while current != u32::MAX {
        check();
        let read = index.read(current);
        let flag = 'flag: {
            for i in 1..=read.len() {
                let bytes = read.get(i).expect("data corruption");
                let tuple = VertexTuple::deserialize_ref(bytes);
                let p = tuple.payload();
                if Some(true) == p.map(&callback) {
                    break 'flag true;
                }
            }
            false
        };
        if flag {
            drop(read);
            let mut write = index.write(current, false);
            for i in 1..=write.len() {
                let bytes = write.get_mut(i).expect("data corruption");
                let mut tuple = VertexTuple::deserialize_mut(bytes);
                let p = tuple.payload();
                if Some(true) == p.map(&callback) {
                    *p = None;
                }
            }
            current = write.get_opaque().next;
        } else {
            current = read.get_opaque().next;
        }
    }
}
