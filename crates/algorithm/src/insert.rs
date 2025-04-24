use crate::linked_vec::LinkedVec;
use crate::operator::*;
use crate::tuples::*;
use crate::vectors::{self};
use crate::{Page, Prefetcher, RelationRead, RelationWrite, tape};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

type Item = (Reverse<Distance>, AlwaysEqual<u32>, AlwaysEqual<u16>);

pub fn insert<
    'b,
    R: RelationRead + RelationWrite,
    O: Operator,
    P: Prefetcher<'b, R = R, T = Item>,
>(
    index: R,
    payload: NonZero<u64>,
    vector: O::Vector,
    bump: &'b bumpalo::Bump,
    mut prefetch: impl FnMut(Vec<(Item, AlwaysEqual<&'b mut [u32]>)>) -> P,
) {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let is_residual = meta_tuple.is_residual();
    let rerank_in_heap = meta_tuple.rerank_in_heap();
    let height_of_root = meta_tuple.height_of_root();
    assert_eq!(dims, vector.as_borrowed().dims(), "unmatched dimensions");
    let root_prefetch = meta_tuple.root_prefetch().to_vec();
    let root_head = meta_tuple.root_head();
    let root_first = meta_tuple.root_first();
    let vectors_first = meta_tuple.vectors_first();
    drop(meta_guard);

    let default_block_lut = if !is_residual {
        Some(O::Vector::block_preprocess(vector.as_borrowed()))
    } else {
        None
    };

    let (list, head) = if !rerank_in_heap {
        vectors::append::<O>(index.clone(), vectors_first, vector.as_borrowed(), payload)
    } else {
        (Vec::new(), 0)
    };

    type State<O> = (u32, Option<<O as Operator>::Vector>);
    let mut state: State<O> = {
        if is_residual {
            let list = root_prefetch.into_iter().map(|id| index.read(id));
            let residual = vectors::read_for_h1_tuple::<R, O, _>(
                root_head,
                list,
                LAccess::new(
                    O::Vector::unpack(vector.as_borrowed()),
                    O::ResidualAccessor::default(),
                ),
            );
            (root_first, Some(residual))
        } else {
            (root_first, None)
        }
    };
    let mut step = |state: State<O>| {
        let mut results = LinkedVec::new();
        {
            let (first, residual) = state;
            let block_lut = if let Some(residual) = residual {
                &O::Vector::block_preprocess(residual.as_borrowed())
            } else if let Some(block_lut) = default_block_lut.as_ref() {
                block_lut
            } else {
                unreachable!()
            };
            tape::read_h1_tape(
                index.clone(),
                first,
                || RAccess::new((&block_lut.1, block_lut.0), O::BlockAccessor::default()),
                |(rough, err), head, first, prefetch| {
                    let lowerbound = Distance::from_f32(rough - err * 1.9);
                    results.push((
                        (Reverse(lowerbound), AlwaysEqual(first), AlwaysEqual(head)),
                        AlwaysEqual(bump.alloc_slice_copy(prefetch)),
                    ));
                },
                |_| (),
            );
        }
        let mut heap = (prefetch)(results.into_vec());
        let mut cache = BinaryHeap::<(Reverse<Distance>, _, _)>::new();
        {
            while let Some(((Reverse(_), AlwaysEqual(first), AlwaysEqual(head)), list)) =
                heap.pop_if(|((d, ..), _)| Some(*d) > cache.peek().map(|(d, ..)| *d))
            {
                if is_residual {
                    let (distance, residual) = vectors::read_for_h1_tuple::<R, O, _>(
                        head,
                        list.into_iter(),
                        LAccess::new(
                            O::Vector::unpack(vector.as_borrowed()),
                            (
                                O::DistanceAccessor::default(),
                                O::ResidualAccessor::default(),
                            ),
                        ),
                    );
                    cache.push((
                        Reverse(distance),
                        AlwaysEqual(first),
                        AlwaysEqual(Some(residual)),
                    ));
                } else {
                    let distance = vectors::read_for_h1_tuple::<R, O, _>(
                        head,
                        list.into_iter(),
                        LAccess::new(
                            O::Vector::unpack(vector.as_borrowed()),
                            O::DistanceAccessor::default(),
                        ),
                    );
                    cache.push((Reverse(distance), AlwaysEqual(first), AlwaysEqual(None)));
                }
            }
            let (_, AlwaysEqual(first), AlwaysEqual(mean)) = cache
                .pop()
                .expect("invariant is violated: tree is not height-balanced");
            (first, mean)
        }
    };
    for _ in (1..height_of_root).rev() {
        state = step(state);
    }

    let (first, residual) = state;
    let code = if let Some(residual) = residual {
        O::Vector::code(residual.as_borrowed())
    } else {
        O::Vector::code(vector.as_borrowed())
    };
    let bytes = AppendableTuple::serialize(&AppendableTuple {
        head,
        dis_u_2: code.dis_u_2,
        factor_ppc: code.factor_ppc,
        factor_ip: code.factor_ip,
        factor_err: code.factor_err,
        payload: Some(payload),
        prefetch: list,
        elements: rabitq::pack_to_u64(&code.signs),
    });

    let jump_guard = index.read(first);
    let jump_bytes = jump_guard.get(1).expect("data corruption");
    let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);

    tape::append(index.clone(), jump_tuple.appendable_first(), &bytes, false);
}
