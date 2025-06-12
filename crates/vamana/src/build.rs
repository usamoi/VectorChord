use crate::Opaque;
use crate::operator::Operator;
use crate::tuples::{MetaTuple, OptionPointer, Tuple};
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
            skip: u32::MAX,
            link: 1,
            _padding_0: Default::default(),
        },
        false,
    );
    assert_eq!(meta_guard.id(), 0);
    let vertex_guard = index.extend(
        Opaque {
            next: u32::MAX,
            skip: 1,
            link: 2,
            _padding_0: Default::default(),
        },
        true,
    );
    assert_eq!(vertex_guard.id(), 1);
    drop(vertex_guard);
    let vector_guard = index.extend(
        Opaque {
            next: u32::MAX,
            skip: u32::MAX,
            link: u32::MAX,
            _padding_0: Default::default(),
        },
        false,
    );
    assert_eq!(vector_guard.id(), 2);
    drop(vector_guard);
    let serialized = MetaTuple::serialize(&MetaTuple {
        dims: vector_options.dims,
        m: index_options.m,
        alpha: index_options.alpha,
        ef_construction: index_options.ef_construction,
        beam_construction: index_options.beam_construction,
        start: OptionPointer::NONE,
    });
    let i = meta_guard
        .alloc(&serialized)
        .expect("implementation: a free page cannot accommodate a single tuple");
    assert_eq!(i, 1);
}
