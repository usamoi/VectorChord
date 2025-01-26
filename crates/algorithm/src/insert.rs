use crate::operator::*;
use crate::pipe::Pipe;
use crate::tape::{access_1, append};
use crate::tuples::*;
use crate::vectors::{self};
use crate::{Page, RelationWrite};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZeroU64;
use vector::{VectorBorrowed, VectorOwned};

pub fn insert<O: Operator>(index: impl RelationWrite, payload: NonZeroU64, vector: O::Vector) {
    let meta_guard = index.read(0);
    let meta_tuple = meta_guard.get(1).unwrap().pipe(read_tuple::<MetaTuple>);
    let dims = meta_tuple.dims();
    let is_residual = meta_tuple.is_residual();
    let height_of_root = meta_tuple.height_of_root();
    assert_eq!(dims, vector.as_borrowed().dims(), "unmatched dimensions");
    let root_mean = meta_tuple.root_mean();
    let root_first = meta_tuple.root_first();
    let root_size = meta_tuple.root_size();
    let vectors_first = meta_tuple.vectors_first();
    drop(meta_guard);

    let default_lut_block = if !is_residual {
        Some(O::Vector::compute_lut_block(vector.as_borrowed()))
    } else {
        None
    };

    let mean = vectors::append::<O>(index.clone(), vectors_first, vector.as_borrowed(), payload);

    struct State<O: Operator> {
        first: u32,
        residual: Option<O::Vector>,
        size: u32,
    }
    let mut state: State<O> = {
        let mean = root_mean;
        if is_residual {
            let residual_u = vectors::access_1::<O, _>(
                index.clone(),
                mean,
                LAccess::new(
                    O::Vector::elements_and_metadata(vector.as_borrowed()),
                    O::ResidualAccessor::default(),
                ),
            );
            State {
                residual: Some(residual_u),
                first: root_first,
                size: root_size,
            }
        } else {
            State {
                residual: None,
                first: root_first,
                size: root_size,
            }
        }
    };
    let step = |state: State<O>| {
        let mut results = Vec::with_capacity(state.size as _);
        {
            let lut = if let Some(residual) = state.residual {
                &O::Vector::compute_lut_block(residual.as_borrowed())
            } else {
                default_lut_block.as_ref().unwrap()
            };
            access_1(
                index.clone(),
                state.first,
                || {
                    RAccess::new(
                        (&lut.4, (lut.0, lut.1, lut.2, lut.3, 1.9f32)),
                        O::Distance::block_accessor(),
                    )
                },
                |lowerbound, mean, first, size| {
                    results.push((
                        Reverse(lowerbound),
                        AlwaysEqual(mean),
                        AlwaysEqual(first),
                        AlwaysEqual(size),
                    ));
                },
            );
        }
        let mut heap = BinaryHeap::from(results);
        let mut cache = BinaryHeap::<(Reverse<Distance>, _)>::new();
        {
            while !heap.is_empty() && heap.peek().map(|x| x.0) > cache.peek().map(|x| x.0) {
                let (_, AlwaysEqual(mean), AlwaysEqual(first), AlwaysEqual(size)) =
                    heap.pop().unwrap();
                if is_residual {
                    let (dis_u, residual_u) = vectors::access_1::<O, _>(
                        index.clone(),
                        mean,
                        LAccess::new(
                            O::Vector::elements_and_metadata(vector.as_borrowed()),
                            (
                                O::DistanceAccessor::default(),
                                O::ResidualAccessor::default(),
                            ),
                        ),
                    );
                    cache.push((
                        Reverse(dis_u),
                        AlwaysEqual(State {
                            residual: Some(residual_u),
                            first,
                            size,
                        }),
                    ));
                } else {
                    let dis_u = vectors::access_1::<O, _>(
                        index.clone(),
                        mean,
                        LAccess::new(
                            O::Vector::elements_and_metadata(vector.as_borrowed()),
                            O::DistanceAccessor::default(),
                        ),
                    );
                    cache.push((
                        Reverse(dis_u),
                        AlwaysEqual(State {
                            residual: None,
                            first,
                            size,
                        }),
                    ));
                }
            }
            let (_, AlwaysEqual(state)) = cache.pop().unwrap();
            state
        }
    };
    for _ in (1..height_of_root).rev() {
        state = step(state);
    }

    let code = if let Some(residual) = state.residual {
        O::Vector::code(residual.as_borrowed())
    } else {
        O::Vector::code(vector.as_borrowed())
    };
    let bytes = serialize(&H0Tuple::_0 {
        mean,
        dis_u_2: code.dis_u_2,
        factor_ppc: code.factor_ppc,
        factor_ip: code.factor_ip,
        factor_err: code.factor_err,
        payload: Some(payload),
        elements: rabitq::pack_to_u64(&code.signs),
    });

    let jump_guard = index.read(state.first);
    let jump_tuple = jump_guard
        .get(1)
        .expect("data corruption")
        .pipe(read_tuple::<JumpTuple>);

    let first = jump_tuple.first();

    append(index.clone(), first, &bytes, false);
}
