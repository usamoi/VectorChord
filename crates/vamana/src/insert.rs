use crate::Opaque;
use crate::operator::{Operator, Vector};
use crate::results::Results;
use crate::tuples::*;
use crate::visited::Visited;
use algo::prefetcher::{Prefetcher, PrefetcherSequenceFamily};
use algo::{Page, PageGuard, RelationRead, RelationWrite};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::iter::{repeat, zip};
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

pub fn insert<'b, R: RelationRead + RelationWrite, O: Operator>(
    index: &'b R,
    vector: <O::Vector as VectorOwned>::Borrowed<'b>,
    payload: NonZero<u64>,
    mut prefetch_vertexs: impl PrefetcherSequenceFamily<'b, R> + 'b,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    assert_eq!(dims, vector.dims(), "unmatched dimensions");
    let start = meta_tuple.start();
    let m = meta_tuple.m();
    let ef = meta_tuple.ef_construction();
    let alpha = 1.0f32;
    drop(meta_guard);
    let vector_bytes = {
        let b8_code = O::Vector::b8_code(vector);
        VectorTuple::serialize(&VectorTuple {
            metadata: b8_code.0.into_array(),
            payload: Some(payload),
            elements: rabitq::b8::binary::pack_code(&b8_code.1),
            neighbours: vec![Neighbour::NULL; m as usize],
        })
    };
    let mut vertex_bytes = {
        let b1_code = O::Vector::b1_code(vector);
        VertexTuple::serialize(&VertexTuple {
            metadata: b1_code.0.into_array(),
            payload: Some(payload),
            elements: rabitq::original::binary::pack_code(&b1_code.1),
            pointer: Pointer::NULL,
        })
    };
    let (vector_pointer, vertex_pointer) =
        if let Some(mut vertex_guard) = index.search(vertex_bytes.len()) {
            let vector_pointer = {
                let mut vector_guard = index.write(vertex_guard.get_opaque().link, false);
                loop {
                    if let Some(i) = vector_guard.alloc(&vector_bytes) {
                        break (vector_guard.id(), i);
                    }
                    if vector_guard.get_opaque().next == u32::MAX {
                        vector_guard = index.extend(
                            Opaque {
                                next: u32::MAX,
                                link: u32::MAX,
                            },
                            false,
                        );
                        break (
                            vector_guard.id(),
                            vector_guard.alloc(&vector_bytes).expect(
                                "implementation: a free page cannot accommodate a single tuple",
                            ),
                        );
                    }
                    vector_guard = index.write(vector_guard.get_opaque().next, false);
                }
            };
            {
                let mut vertex_tuple = VertexTuple::deserialize_mut(&mut vertex_bytes);
                *vertex_tuple.pointer() = Pointer::new(vector_pointer);
            }
            let vertex_pointer = {
                (
                    vertex_guard.id(),
                    vertex_guard
                        .alloc(&vertex_bytes)
                        .expect("implementation: a free page cannot accommodate a single tuple"),
                )
            };
            (vector_pointer, vertex_pointer)
        } else {
            let vector_pointer = {
                let mut vector_guard = index.extend(
                    Opaque {
                        next: u32::MAX,
                        link: u32::MAX,
                    },
                    false,
                );
                (
                    vector_guard.id(),
                    vector_guard
                        .alloc(&vector_bytes)
                        .expect("implementation: a free page cannot accommodate a single tuple"),
                )
            };
            {
                let mut vertex_tuple = VertexTuple::deserialize_mut(&mut vertex_bytes);
                *vertex_tuple.pointer() = Pointer::new(vector_pointer);
            }
            let vertex_pointer = {
                let mut vertex_guard = index.extend(
                    Opaque {
                        next: u32::MAX,
                        link: vector_pointer.0,
                    },
                    true,
                );
                (
                    vertex_guard.id(),
                    vertex_guard
                        .alloc(&vertex_bytes)
                        .expect("implementation: a free page cannot accommodate a single tuple"),
                )
            };
            (vector_pointer, vertex_pointer)
        };
    let start = if start.into_inner().is_none() {
        let mut meta_guard = index.write(0, false);
        let meta_bytes = meta_guard.get_mut(1).expect("data corruption");
        let mut meta_tuple = MetaTuple::deserialize_mut(meta_bytes);
        let dst = meta_tuple.start();
        if dst.into_inner().is_none() {
            *dst = Pointer::new(vertex_pointer);
            return;
        } else {
            *dst
        }
    } else {
        start
    };
    let vector_lut = O::Vector::b8_preprocess(vector);
    let vertex_lut = O::Vector::b1_preprocess(vector);
    let mut visited = Visited::new();
    let mut candidates = BinaryHeap::<(Reverse<Distance>, AlwaysEqual<_>)>::new();
    let Some(s) = start.into_inner() else {
        return;
    };
    {
        visited.insert(s);
        let vertex_guard = index.read(s.0);
        let Some(vertex_bytes) = vertex_guard.get(s.1) else {
            // the link is broken
            return;
        };
        let mut vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
        if let Some(pointer_s) = vertex_tuple.pointer().into_inner() {
            let score_s = O::b1_process(
                (vertex_tuple.metadata(), vertex_tuple.elements()),
                &vertex_lut,
            );
            candidates.push((Reverse(score_s), AlwaysEqual((pointer_s, s))));
        }
    }
    let mut iter = std::iter::from_fn(|| {
        while let Some((_, AlwaysEqual((pointer_u, u)))) = candidates.pop() {
            let vector_guard = index.read(pointer_u.0);
            let Some(vector_bytes) = vector_guard.get(pointer_u.1) else {
                // the link is broken
                continue;
            };
            let vector_tuple = VectorTuple::deserialize_ref(vector_bytes);
            let outs_u = vector_tuple
                .neighbours()
                .iter()
                .flat_map(|neighbour| neighbour.into_inner())
                .map(|(v, _)| v)
                .filter(|&v| !visited.contains(v))
                .collect::<VecDeque<_>>();
            let dis_u = O::b8_process(
                (vector_tuple.metadata(), vector_tuple.elements()),
                &vector_lut,
            );
            drop(vector_guard);
            let mut iterator = prefetch_vertexs.prefetch(outs_u);
            while let Some((v, guards)) = iterator.next() {
                visited.insert(v);
                let vertex_guard = {
                    let mut guards = guards;
                    let guard = guards.pop().expect("should be at least one element");
                    assert!(guards.pop().is_none(), "should be at most one element");
                    guard
                };
                let Some(vertex_bytes) = vertex_guard.get(v.1) else {
                    // the link is broken
                    continue;
                };
                let mut vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                if let Some(pointer_v) = vertex_tuple.pointer().into_inner() {
                    let score_v = O::b1_process(
                        (vertex_tuple.metadata(), vertex_tuple.elements()),
                        &vertex_lut,
                    );
                    candidates.push((Reverse(score_v), AlwaysEqual((pointer_v, v))));
                }
            }
            return Some((Reverse(dis_u), AlwaysEqual((pointer_u, u))));
        }
        None
    });
    let mut results = Results::new(ef as _);
    for element @ (Reverse(dis_c), _) in iter.by_ref() {
        results.push(element);
        if results
            .peek_ef_th()
            .map(|dis_e| dis_e < dis_c)
            .unwrap_or_default()
        {
            break;
        }
    }
    let trace = results
        .into_inner()
        .0
        .into_iter()
        .map(|(dis_v, x)| (Reverse(dis_v), x))
        .collect::<Vec<_>>();
    let outs = crate::prune::robust_prune(
        |x, y| O::b8_distance((x.0, &x.1), (y.0, &y.1)),
        (vector_pointer, vertex_pointer),
        trace.into_iter().flat_map(|item| {
            let (Reverse(dis_u), AlwaysEqual((pointer_u, u))) = item;
            let vector_guard = index.read(pointer_u.0);
            let Some(vector_bytes) = vector_guard.get(pointer_u.1) else {
                // the link is broken
                return None;
            };
            let vector_tuple = VectorTuple::deserialize_ref(vector_bytes);
            let vector = (vector_tuple.metadata(), vector_tuple.elements().to_vec());
            Some(((Reverse(dis_u), AlwaysEqual((pointer_u, u))), vector))
        }),
        m,
        alpha,
    );
    {
        let mut vector_guard = index.write(vector_pointer.0, false);
        let Some(vector_bytes) = vector_guard.get_mut(vector_pointer.1) else {
            // unreachable
            return;
        };
        let mut vector_tuple = VectorTuple::deserialize_mut(vector_bytes);
        let iterator = outs
            .iter()
            .map(|&(Reverse(dis_v), AlwaysEqual((_, v)))| Neighbour::new(v, dis_v))
            .chain(repeat(Neighbour::NULL));
        for (hole, fill) in zip(vector_tuple.neighbours().iter_mut(), iterator) {
            *hole = fill;
        }
    }
    for (Reverse(dis_t), AlwaysEqual((pointer_u, u))) in outs {
        let vector_guard = index.read(pointer_u.0);
        let Some(vector_bytes) = vector_guard.get(pointer_u.1) else {
            // the link is broken
            continue;
        };
        let vector_tuple = VectorTuple::deserialize_ref(vector_bytes);
        let trace = vector_tuple
            .neighbours()
            .iter()
            .flat_map(|neighbour| neighbour.into_inner())
            .chain(std::iter::once((vertex_pointer, dis_t)))
            .map(|(v, dis_v)| (Reverse(dis_v), AlwaysEqual(v)))
            .collect::<Vec<_>>();
        drop(vector_guard); // deadlock
        let outs = crate::prune::robust_prune(
            |x, y| O::b8_distance((x.0, &x.1), (y.0, &y.1)),
            (pointer_u, u),
            trace
                .into_iter()
                .flat_map(|item| {
                    let (Reverse(dis_u), AlwaysEqual(u)) = item;
                    let vertex_guard = index.read(u.0);
                    let Some(vertex_bytes) = vertex_guard.get(u.1) else {
                        // the link is broken
                        return None;
                    };
                    let mut vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                    let Some(pointer_u) = vertex_tuple.pointer().into_inner() else {
                        // the link is broken
                        return None;
                    };
                    Some((Reverse(dis_u), AlwaysEqual((pointer_u, u))))
                })
                .flat_map(|item| {
                    let (Reverse(dis_u), AlwaysEqual((pointer_u, u))) = item;
                    let vector_guard = index.read(pointer_u.0);
                    let Some(vector_bytes) = vector_guard.get(pointer_u.1) else {
                        // the link is broken
                        return None;
                    };
                    let vector_tuple = VectorTuple::deserialize_ref(vector_bytes);
                    let vector = (vector_tuple.metadata(), vector_tuple.elements().to_vec());
                    Some(((Reverse(dis_u), AlwaysEqual((pointer_u, u))), vector))
                }),
            m,
            alpha,
        );
        let mut vector_guard = index.write(pointer_u.0, false);
        let Some(vector_bytes) = vector_guard.get_mut(pointer_u.1) else {
            // the link is broken
            continue;
        };
        let mut vector_tuple = VectorTuple::deserialize_mut(vector_bytes);
        let filling = outs
            .iter()
            .map(|&(Reverse(dis_v), AlwaysEqual((_, v)))| Neighbour::new(v, dis_v))
            .chain(repeat(Neighbour::NULL));
        for (hole, fill) in zip(vector_tuple.neighbours().iter_mut(), filling) {
            *hole = fill;
        }
    }
}
