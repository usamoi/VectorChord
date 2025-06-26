use crate::Opaque;
use crate::candidates::Candidates;
use crate::operator::{Operator, Vector};
use crate::results::Results;
use crate::tuples::{Pointer, *};
use crate::visited::Visited;
use algo::accessor::{Accessor1, LAccess, LTryAccess, TryAccessor1};
use algo::prefetcher::{Prefetcher, PrefetcherSequenceFamily};
use algo::{Bump, Page, PageGuard, RelationRead, RelationWrite};
use always_equal::AlwaysEqual;
use std::cmp::Reverse;
use std::collections::VecDeque;
use std::iter::{repeat, zip};
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

pub fn insert<'b, R: RelationRead + RelationWrite, O: Operator>(
    index: &'b R,
    vector: <O::Vector as VectorOwned>::Borrowed<'b>,
    payload: NonZero<u64>,
    bump: &'b impl Bump,
    mut prefetch_vertexs: impl PrefetcherSequenceFamily<'b, R> + 'b,
    prefetch_vectors: impl PrefetcherSequenceFamily<'b, R> + 'b,
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
        let (mut slices, metadata) = O::Vector::split(vector, m as usize);
        let last = slices.pop().expect("internal: bad split");
        slices
            .iter()
            .enumerate()
            .map(|(index, elements)| {
                VectorTuple::serialize(&VectorTuple::_1 {
                    elements: elements.to_vec(),
                    payload: Some(payload),
                    index: index as u8,
                })
            })
            .chain(std::iter::once(VectorTuple::serialize(&VectorTuple::_0 {
                metadata,
                payload: Some(payload),
                elements: last.to_vec(),
                neighbours: vec![Neighbour::NULL; m as usize],
            })))
            .collect::<Vec<_>>()
    };
    let mut vertex_bytes = {
        let code = O::Vector::code(vector);
        VertexTuple::serialize(&VertexTuple {
            metadata: code.0.into_array(),
            payload: Some(payload),
            elements: rabitq::original::binary::pack_code(&code.1),
            prefetch: vec![Pointer::NULL; vector_bytes.len()],
        })
    };
    let (vector_pointer, vertex_pointer) =
        if let Some(mut vertex_guard) = index.search(vertex_bytes.len()) {
            let vector_pointer = vector_bytes
                .into_iter()
                .map(|vector_bytes| {
                    let mut vector_guard = index.write(vertex_guard.get_opaque().link, false);
                    loop {
                        if let Some(i) = vector_guard.alloc(&vector_bytes) {
                            break (vector_guard.id(), i);
                        }
                        if vector_guard.get_opaque().next == u32::MAX {
                            let mut guard = vector_guard;
                            vector_guard = index.extend(
                                Opaque {
                                    next: u32::MAX,
                                    link: u32::MAX,
                                },
                                false,
                            );
                            guard.get_opaque_mut().next = vector_guard.id();
                            drop(guard);
                            break (
                                vector_guard.id(),
                                vector_guard.alloc(&vector_bytes).expect(
                                    "implementation: a free page cannot accommodate a single tuple",
                                ),
                            );
                        }
                        vector_guard = index.write(vector_guard.get_opaque().next, false);
                    }
                })
                .collect::<Vec<_>>();
            {
                let mut vertex_tuple = VertexTuple::deserialize_mut(&mut vertex_bytes);
                vertex_tuple.prefetch().copy_from_slice(
                    &vector_pointer
                        .iter()
                        .map(|&p| Pointer::new(p))
                        .collect::<Vec<_>>(),
                );
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
            let mut vertex_guard = {
                let vector_guard = index.extend(
                    Opaque {
                        next: u32::MAX,
                        link: u32::MAX,
                    },
                    false,
                );
                let id = vector_guard.id();
                drop(vector_guard);
                index.extend(
                    Opaque {
                        next: u32::MAX,
                        link: id,
                    },
                    true,
                )
            };
            let vector_pointer = vector_bytes
                .into_iter()
                .map(|vector_bytes| {
                    let mut vector_guard = index.write(vertex_guard.get_opaque().link, false);
                    loop {
                        if let Some(i) = vector_guard.alloc(&vector_bytes) {
                            break (vector_guard.id(), i);
                        }
                        if vector_guard.get_opaque().next == u32::MAX {
                            let mut guard = vector_guard;
                            vector_guard = index.extend(
                                Opaque {
                                    next: u32::MAX,
                                    link: u32::MAX,
                                },
                                false,
                            );
                            guard.get_opaque_mut().next = vector_guard.id();
                            drop(guard);
                            break (
                                vector_guard.id(),
                                vector_guard.alloc(&vector_bytes).expect(
                                    "implementation: a free page cannot accommodate a single tuple",
                                ),
                            );
                        }
                        vector_guard = index.write(vector_guard.get_opaque().next, false);
                    }
                })
                .collect::<Vec<_>>();
            {
                let mut vertex_tuple = VertexTuple::deserialize_mut(&mut vertex_bytes);
                vertex_tuple.prefetch().copy_from_slice(
                    &vector_pointer
                        .iter()
                        .map(|&p| Pointer::new(p))
                        .collect::<Vec<_>>(),
                );
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
    let lut = O::Vector::preprocess(vector);
    let mut visited = Visited::new();
    let mut candidates = Candidates::new(1, prefetch_vectors);
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
        let pointer_s = {
            let buffer = bump.alloc_slice_default(vertex_tuple.prefetch().len());
            for (i, x) in vertex_tuple.prefetch().iter().enumerate() {
                buffer[i] = x.into_inner_unchecked();
            }
            buffer
        };
        let score_s = O::process((vertex_tuple.metadata(), vertex_tuple.elements()), &lut);
        candidates.push((Reverse(score_s), AlwaysEqual(pointer_s)));
    }
    let mut iter = std::iter::from_fn(move || {
        while let Some(((_, AlwaysEqual(pointers_u)), guards_u)) = candidates.pop() {
            let Some((outs_u, payload_u, dis_u)) = read::<R, O, _>(
                &visited,
                guards_u.into_iter(),
                pointers_u,
                LAccess::new(O::Vector::unpack(vector), O::DistanceAccessor::default()),
            ) else {
                // the link is broken
                continue;
            };
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
                let pointer_v = {
                    let buffer = bump.alloc_slice_default(vertex_tuple.prefetch().len());
                    for (i, x) in vertex_tuple.prefetch().iter().enumerate() {
                        buffer[i] = x.into_inner_unchecked();
                    }
                    buffer
                };
                let score_v = O::process((vertex_tuple.metadata(), vertex_tuple.elements()), &lut);
                candidates.push((Reverse(score_v), AlwaysEqual(pointer_v)));
            }
            return Some((Reverse(dis_u), AlwaysEqual(payload_u)));
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
            |x, y| O::distance((x.0, &x.1), (y.0, &y.1)),
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
                    let pointer_u = {
                        let buffer = bump.alloc_slice_default(vertex_tuple.prefetch().len());
                        for (i, x) in vertex_tuple.prefetch().iter().enumerate() {
                            buffer[i] = x.into_inner_unchecked();
                        }
                        buffer
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

fn read<
    'r,
    R: RelationRead + 'r,
    O: Operator,
    A: Accessor1<<O::Vector as Vector>::Element, <O::Vector as Vector>::Metadata>,
>(
    visited: &Visited,
    mut guards_u: impl Iterator<Item = R::ReadGuard<'r>>,
    pointers: &[(u32, u16)],
    accessor: A,
) -> Option<(VecDeque<(u32, u16)>, Option<NonZero<u64>>, A::Output)> {
    assert!(pointers.len() >= 1);
    let f = pointers.len() - 1;
    let mut result = accessor;
    for (index, pointer) in pointers[..f].iter().enumerate() {
        let vector_guard = guards_u.next().unwrap();
        let Some(vector_bytes) = vector_guard.get(pointer.1) else {
            // the link is broken
            return None;
        };
        let vector_tuple = VectorTuple::<O::Vector>::deserialize_ref(vector_bytes);
        match vector_tuple {
            VectorTupleReader::_0(_) => {
                // link is broken
                return None;
            }
            VectorTupleReader::_1(vector_tuple) => {
                if vector_tuple.index() != index as u8 {
                    return None;
                }
                result.push(vector_tuple.elements());
            }
        }
    }
    let outs_u;
    let dis_u;
    let payload_u;
    {
        let vector_guard = guards_u.next().unwrap();
        let Some(vector_bytes) = vector_guard.get(pointers[f].1) else {
            // the link is broken
            return None;
        };
        let vector_tuple = VectorTuple::<O::Vector>::deserialize_ref(vector_bytes);
        match vector_tuple {
            VectorTupleReader::_0(vector_tuple) => {
                outs_u = vector_tuple
                    .neighbours()
                    .iter()
                    .flat_map(|neighbour| neighbour.into_inner())
                    .map(|(v, _)| v)
                    .filter(|&v| !visited.contains(v))
                    .collect::<VecDeque<_>>();
                result.push(vector_tuple.elements());
                dis_u = result.finish(*vector_tuple.metadata());
                payload_u = vector_tuple.payload();
            }
            VectorTupleReader::_1(_) => {
                // link is broken
                return None;
            }
        }
    }
    Some((outs_u, payload_u, dis_u))
}
