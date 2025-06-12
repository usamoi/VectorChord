use crate::Opaque;
use crate::operator::Operator;
use crate::tuples::{MetaTuple, Pointer, Tuple};
use crate::types::{VamanaIndexOptions, VectorOptions};
use algo::{Page, PageGuard, RelationWrite};

pub fn build<R: RelationWrite, O: Operator>(
    vector_options: VectorOptions,
    index_options: VamanaIndexOptions,
    index: &R,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let mut meta_guard = index.extend(
        Opaque {
            next: u32::MAX,
            link: u32::MAX,
        },
        false,
    );
    assert_eq!(meta_guard.id(), 0);
    let serialized = MetaTuple::serialize(&MetaTuple {
        dims: vector_options.dims,
        rerank_in_heap: true,
        ef_construction: index_options.ef_construction,
        m: index_options.m,
        start: Pointer::NULL,
    });
    let i = meta_guard
        .alloc(&serialized)
        .expect("implementation: a free page cannot accommodate a single tuple");
    assert_eq!(i, 1);
}
