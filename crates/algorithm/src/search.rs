use crate::operator::*;
use crate::pipe::Pipe;
use crate::tape::{access_0, access_1};
use crate::tuples::*;
use crate::{Page, RelationRead, vectors};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZeroU64;
use vector::{VectorBorrowed, VectorOwned};

pub fn search<O: Operator>(
    index: impl RelationRead,
    vector: O::Vector,
    probes: Vec<u32>,
    epsilon: f32,
) -> impl Iterator<Item = (Distance, NonZeroU64)> {
    let meta_guard = index.read(0);
    let meta_tuple = meta_guard.get(1).unwrap().pipe(read_tuple::<MetaTuple>);
    let dims = meta_tuple.dims();
    let is_residual = meta_tuple.is_residual();
    let height_of_root = meta_tuple.height_of_root();
    assert_eq!(dims, vector.as_borrowed().dims(), "unmatched dimensions");
    assert_eq!(height_of_root as usize, 1 + probes.len(), "invalid probes");
    let root_mean = meta_tuple.root_mean();
    let root_first = meta_tuple.root_first();
    let root_size = meta_tuple.root_size();
    drop(meta_guard);

    let default_lut = if !is_residual {
        Some(O::Vector::compute_lut(vector.as_borrowed()))
    } else {
        None
    };

    struct State<O: Operator> {
        residual: Option<O::Vector>,
        first: u32,
        size: u32,
    }
    let mut states: Vec<State<O>> = vec![{
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
    }];
    let step = |states: Vec<State<O>>, probes| {
        let mut results = Vec::with_capacity(states.iter().map(|x| x.size).sum::<u32>() as _);
        for state in states {
            let lut = if let Some(residual) = state.residual {
                &O::Vector::compute_lut_block(residual.as_borrowed())
            } else {
                default_lut.as_ref().map(|x| &x.0).unwrap()
            };
            access_1(
                index.clone(),
                state.first,
                || {
                    RAccess::new(
                        (&lut.4, (lut.0, lut.1, lut.2, lut.3, epsilon)),
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
        std::iter::from_fn(|| {
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
            let (_, AlwaysEqual(state)) = cache.pop()?;
            Some(state)
        })
        .take(probes as usize)
        .collect()
    };
    for i in (1..height_of_root).rev() {
        states = step(states, probes[i as usize - 1]);
    }

    let mut results = Vec::new();
    for state in states {
        let lut = if let Some(residual) = state.residual.as_ref().map(|x| x.as_borrowed()) {
            &O::Vector::compute_lut(residual)
        } else {
            default_lut.as_ref().unwrap()
        };
        let jump_guard = index.read(state.first);
        let jump_tuple = jump_guard
            .get(1)
            .expect("data corruption")
            .pipe(read_tuple::<JumpTuple>);
        let first = jump_tuple.first();
        access_0(
            index.clone(),
            first,
            || {
                RAccess::new(
                    (&lut.0.4, (lut.0.0, lut.0.1, lut.0.2, lut.0.3, epsilon)),
                    O::Distance::block_accessor(),
                )
            },
            |code| O::Distance::compute_lowerbound_binary(&lut.1, code, epsilon),
            |lowerbound, mean, payload| {
                results.push((Reverse(lowerbound), AlwaysEqual(mean), AlwaysEqual(payload)));
            },
        );
    }
    let mut heap = BinaryHeap::from(results);
    let mut cache = BinaryHeap::<(Reverse<Distance>, _)>::new();
    std::iter::from_fn(move || {
        while !heap.is_empty() && heap.peek().map(|x| x.0) > cache.peek().map(|x| x.0) {
            let (_, AlwaysEqual(mean), AlwaysEqual(pay_u)) = heap.pop().unwrap();
            if let Some(dis_u) = vectors::access_0::<O, _>(
                index.clone(),
                mean,
                pay_u,
                LAccess::new(
                    O::Vector::elements_and_metadata(vector.as_borrowed()),
                    O::DistanceAccessor::default(),
                ),
            ) {
                cache.push((Reverse(dis_u), AlwaysEqual(pay_u)));
            };
        }
        let (Reverse(dis_u), AlwaysEqual(pay_u)) = cache.pop()?;
        Some((dis_u, pay_u))
    })
}
