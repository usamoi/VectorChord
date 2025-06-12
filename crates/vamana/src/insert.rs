use crate::checker::Checker;
use crate::operator::{Operator, Vector};
use crate::tuples::{MetaTuple, Neighbour, Start, Tuple, VertexTuple, WithReader, WithWriter};
use crate::visited::Visited;
use crate::{Id, Opaque};
use algo::prefetcher::{Prefetcher, PrefetcherSequenceFamily};
use algo::{Page, PageGuard, RelationRead, RelationWrite};
use always_equal::AlwaysEqual;
use distance::Distance;
use rabitq8::CodeMetadata;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::iter::{repeat, zip};
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

pub fn insert<'r, 'b: 'r, R: RelationRead + RelationWrite, O: Operator>(
    index: &'r R,
    vector: <O::Vector as VectorOwned>::Borrowed<'b>,
    payload: NonZero<u64>,
    mut prefetch_vertexs: impl PrefetcherSequenceFamily<'r, R> + 'b,
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
    let target;
    {
        let code = O::Vector::code(vector);
        let serialized = VertexTuple::serialize(&VertexTuple {
            metadata: [code.0.dis_u_2, code.0.norm_of_lattice, code.0.sum_of_code],
            payload: Some(payload),
            elements: code.1,
            neighbours: vec![Neighbour::NULL; m as usize],
        });
        let mut guard = index
            .search(serialized.len())
            .unwrap_or_else(|| index.extend(Opaque {}, true));
        let i = guard
            .alloc(&serialized)
            .expect("implementation: a free page cannot accommodate a single tuple");
        target = (guard.id(), i);
        drop(guard);
    }
    let start = if start.is_null() {
        let mut meta_guard = index.write(0, false);
        let meta_bytes = meta_guard.get_mut(1).expect("data corruption");
        let mut meta_tuple = MetaTuple::deserialize_mut(meta_bytes);
        let dst = meta_tuple.start();
        if dst.is_null() {
            *dst = Start::new(target);
            return;
        } else {
            *dst
        }
    } else {
        start
    };
    let lut = O::Vector::binary_preprocess(vector);
    let binary_process =
        move |value, code, lut| Distance::from_f32(O::binary_process(dims, value, code, lut));
    let mut visited = Visited::new();
    let mut candidates = BinaryHeap::<(Reverse<Distance>, AlwaysEqual<(Id, Vec<Id>)>)>::new();
    let mut checker = Checker::new(ef as _);
    let Some(s) = start.id() else {
        return;
    };
    {
        visited.insert(s);
        let vertex_guard = index.read(s.0);
        let Some(vertex_bytes) = vertex_guard.get(s.1) else {
            return;
        };
        let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
        let score_s = O::binary_access(&lut, binary_process)(
            vertex_tuple.metadata(),
            vertex_tuple.elements(),
        );
        let outs_s = vertex_tuple
            .neighbours()
            .iter()
            .flat_map(|t| t.id())
            .collect();
        candidates.push((Reverse(score_s), AlwaysEqual((s, outs_s))));
    }
    let mut iter = std::iter::from_fn(|| {
        while let Some((Reverse(dis_u), AlwaysEqual((u, outs_u)))) = candidates.pop() {
            let mut filtered = prefetch_vertexs.prefetch(
                outs_u
                    .into_iter()
                    .filter(|&v| !visited.contains(v))
                    .collect::<VecDeque<_>>(),
            );
            while let Some((v, guards)) = filtered.next() {
                visited.insert(v);
                let vertex_guard = {
                    let mut guards = guards;
                    let guard = guards.pop().expect("should be at least one element");
                    assert!(guards.pop().is_none(), "should be at most one element");
                    guard
                };
                let Some(vertex_bytes) = vertex_guard.get(v.1) else {
                    continue;
                };
                let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                let score_v = O::binary_access(&lut, binary_process)(
                    vertex_tuple.metadata(),
                    vertex_tuple.elements(),
                );
                let outs_v = vertex_tuple
                    .neighbours()
                    .iter()
                    .flat_map(|t| t.id())
                    .collect();
                candidates.push((Reverse(score_v), AlwaysEqual((v, outs_v))));
            }
            return Some((Reverse(dis_u), AlwaysEqual(u)));
        }
        None
    });
    let mut leading = Vec::new();
    for element @ (Reverse(dis), _) in iter.by_ref() {
        leading.push(element);
        if checker.check(dis) {
            checker.push(element);
        } else {
            break;
        }
    }
    leading.sort_unstable();
    let outs = crate::prune::robust_prune(
        |id| {
            let vertex_guard = index.read(id.0);
            let Some(vertex_bytes) = vertex_guard.get(id.1) else {
                return None;
            };
            let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
            let metadata = vertex_tuple.metadata();
            Some((
                CodeMetadata {
                    dis_u_2: metadata[0],
                    norm_of_lattice: metadata[1],
                    sum_of_code: metadata[2],
                },
                vertex_tuple.elements().to_vec(),
            ))
        },
        |lhs, rhs| {
            let value = simd::u8::reduce_sum_of_x_as_u32_y_as_u32(&lhs.1, &rhs.1);
            O::binary_process(dims, value, lhs.0, rhs.0).into()
        },
        target,
        leading.into_iter(),
        m,
        alpha,
    );
    {
        let mut vertex_guard = index.write(target.0, true);
        let Some(vertex_bytes) = vertex_guard.get_mut(target.1) else {
            return;
        };
        let mut vertex_tuple = VertexTuple::deserialize_mut(vertex_bytes);
        for (hole, fill) in zip(
            vertex_tuple.neighbours().iter_mut(),
            outs.iter()
                .map(|&(Reverse(dis), AlwaysEqual(id))| Neighbour::new(dis, id))
                .chain(repeat(Neighbour::NULL)),
        ) {
            *hole = fill;
        }
    }
    for (Reverse(dis), AlwaysEqual(id)) in outs {
        let mut vertex_guard = index.write(id.0, true);
        let Some(vertex_bytes) = vertex_guard.get_mut(id.1) else {
            continue;
        };
        let mut vertex_tuple = VertexTuple::deserialize_mut(vertex_bytes);
        let merged_outs = vertex_tuple
            .neighbours()
            .iter()
            .flat_map(|neighbour| {
                let distance = Reverse(neighbour.distance()?);
                let id = AlwaysEqual(neighbour.id()?);
                Some((distance, id))
            })
            .chain(std::iter::once((Reverse(dis), AlwaysEqual(target))))
            .collect::<Vec<_>>();
        drop(vertex_guard);
        let new_outs = crate::prune::robust_prune(
            |id| {
                let vertex_guard = index.read(id.0);
                let Some(vertex_bytes) = vertex_guard.get(id.1) else {
                    return None;
                };
                let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                let metadata = vertex_tuple.metadata();
                Some((
                    CodeMetadata {
                        dis_u_2: metadata[0],
                        norm_of_lattice: metadata[1],
                        sum_of_code: metadata[2],
                    },
                    vertex_tuple.elements().to_vec(),
                ))
            },
            |lhs, rhs| {
                let value = simd::u8::reduce_sum_of_x_as_u32_y_as_u32(&lhs.1, &rhs.1);
                O::binary_process(dims, value, lhs.0, rhs.0).into()
            },
            id,
            merged_outs.clone().into_iter(),
            m,
            alpha,
        );
        let mut vertex_guard = index.write(id.0, true);
        let Some(vertex_bytes) = vertex_guard.get_mut(id.1) else {
            continue;
        };
        let mut vertex_tuple = VertexTuple::deserialize_mut(vertex_bytes);
        for (hole, fill) in zip(
            vertex_tuple.neighbours().iter_mut(),
            new_outs
                .iter()
                .map(|&(Reverse(dis), AlwaysEqual(id))| Neighbour::new(dis, id))
                .chain(repeat(Neighbour::NULL)),
        ) {
            *hole = fill;
        }
    }
}
