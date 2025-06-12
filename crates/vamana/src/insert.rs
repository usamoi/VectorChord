use crate::operator::{Accessor2, Operator, Vector};
use crate::results::Results;
use crate::tuples::{MetaTuple, Neighbour, Start, Tuple, VertexTuple, WithReader, WithWriter};
use crate::visited::Visited;
use crate::{Id, Opaque};
use algo::prefetcher::{Prefetcher, PrefetcherSequenceFamily};
use algo::{Page, PageGuard, RelationRead, RelationWrite};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::iter::{repeat, zip};
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

pub fn insert<'r, 'b: 'r, R: RelationRead + RelationWrite, O: Operator>(
    index: &'r R,
    vector: <O::Vector as VectorOwned>::Borrowed<'b>,
    raw: <O::Vector as VectorOwned>::Borrowed<'b>,
    payload: NonZero<u64>,
    mut fetch: impl FnMut(NonZero<u64>) -> Option<O::Vector> + 'b,
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
    let _epsilon = 1.9f32;
    let alpha = 1.0f32;
    drop(meta_guard);
    let t;
    {
        let code = O::Vector::code(vector);
        let serialized = VertexTuple::serialize(&VertexTuple {
            metadata: code.0.into_array(),
            payload: Some(payload),
            elements: rabitq::b2::binary::pack_code(&code.1),
            neighbours: vec![Neighbour::NULL; m as usize],
        });
        let mut guard = index
            .search(serialized.len())
            .unwrap_or_else(|| index.extend(Opaque {}, true));
        let i = guard
            .alloc(&serialized)
            .expect("implementation: a free page cannot accommodate a single tuple");
        t = (guard.id(), i);
        drop(guard);
    }
    let start = if start.into_inner().is_none() {
        let mut meta_guard = index.write(0, false);
        let meta_bytes = meta_guard.get_mut(1).expect("data corruption");
        let mut meta_tuple = MetaTuple::deserialize_mut(meta_bytes);
        let dst = meta_tuple.start();
        if dst.into_inner().is_none() {
            *dst = Start::new(t);
            return;
        } else {
            *dst
        }
    } else {
        start
    };
    let lut = O::Vector::binary_preprocess(vector);
    let binary_process = move |value, code, lut| {
        let (rough,) = O::binary_process(dims, value, code, lut);
        Distance::from_f32(rough)
    };
    let mut visited = Visited::new();
    let mut candidates = BinaryHeap::<(
        Reverse<Distance>,
        AlwaysEqual<(Id, Option<NonZero<u64>>, Vec<Id>)>,
    )>::new();
    let Some(s) = start.into_inner() else {
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
        let payload_s = vertex_tuple.payload();
        let outs_s = vertex_tuple
            .neighbours()
            .iter()
            .flat_map(|neighbour| neighbour.into_inner())
            .map(|(_, v)| v)
            .collect();
        candidates.push((Reverse(score_s), AlwaysEqual((s, payload_s, outs_s))));
    }
    let mut iter = std::iter::from_fn(|| {
        while let Some((_, AlwaysEqual((u, payload_u, outs_u)))) = candidates.pop() {
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
                let payload_v = vertex_tuple.payload();
                let score_v = if let Some(payload_v) = payload_v {
                    fetch(payload_v).map(|raw_u| {
                        let raw = O::Vector::unpack(raw);
                        let raw_u = O::Vector::unpack(raw_u.as_borrowed());
                        let mut accessor = O::DistanceAccessor::default();
                        accessor.push(raw.0, raw_u.0);
                        accessor.finish(raw.1, raw_u.1)
                    })
                } else {
                    None
                }
                .unwrap_or_else(|| {
                    O::binary_access(&lut, binary_process)(
                        vertex_tuple.metadata(),
                        vertex_tuple.elements(),
                    )
                });
                let outs_v = vertex_tuple
                    .neighbours()
                    .iter()
                    .flat_map(|neighbour| neighbour.into_inner())
                    .map(|(_, v)| v)
                    .collect();
                candidates.push((Reverse(score_v), AlwaysEqual((v, payload_v, outs_v))));
            }
            let Some(payload_u) = payload_u else {
                continue;
            };
            let Some(raw_u) = fetch(payload_u) else {
                continue;
            };
            let dis_u = {
                let raw = O::Vector::unpack(raw);
                let raw_u = O::Vector::unpack(raw_u.as_borrowed());
                let mut accessor = O::DistanceAccessor::default();
                accessor.push(raw.0, raw_u.0);
                accessor.finish(raw.1, raw_u.1)
            };
            return Some((Reverse(dis_u), AlwaysEqual((u, payload_u))));
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
        |x, y| {
            let raw_x = O::Vector::unpack(x.as_borrowed());
            let raw_y = O::Vector::unpack(y.as_borrowed());
            let mut accessor = O::DistanceAccessor::default();
            accessor.push(raw_x.0, raw_y.0);
            accessor.finish(raw_x.1, raw_y.1)
        },
        (t, payload),
        trace.into_iter().flat_map(|item| {
            let (_, AlwaysEqual((_, payload_u))) = item;
            let vector_u = fetch(payload_u)?;
            Some((item, vector_u))
        }),
        m,
        alpha,
    );
    {
        let mut vertex_guard = index.write(t.0, true);
        let Some(vertex_bytes) = vertex_guard.get_mut(t.1) else {
            return;
        };
        let mut vertex_tuple = VertexTuple::deserialize_mut(vertex_bytes);
        let filling = outs
            .iter()
            .map(|&(Reverse(dis_v), AlwaysEqual((v, _)))| Neighbour::new(dis_v, v))
            .chain(repeat(Neighbour::NULL));
        for (hole, fill) in zip(vertex_tuple.neighbours().iter_mut(), filling) {
            *hole = fill;
        }
    }
    for (Reverse(dis_t), AlwaysEqual((u, _))) in outs {
        let vertex_guard = index.read(u.0);
        let Some(vertex_bytes) = vertex_guard.get(u.1) else {
            continue;
        };
        let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
        let trace = vertex_tuple
            .neighbours()
            .iter()
            .flat_map(|neighbour| neighbour.into_inner())
            .chain(std::iter::once((dis_t, t)))
            .map(|(dis_v, v)| (Reverse(dis_v), AlwaysEqual(v)))
            .collect::<Vec<_>>();
        drop(vertex_guard); // deadlock
        let outs = crate::prune::robust_prune(
            |x, y| {
                let raw_x = O::Vector::unpack(x.as_borrowed());
                let raw_y = O::Vector::unpack(y.as_borrowed());
                let mut accessor = O::DistanceAccessor::default();
                accessor.push(raw_x.0, raw_y.0);
                accessor.finish(raw_x.1, raw_y.1)
            },
            u,
            trace.into_iter().flat_map(|item| {
                let (_, AlwaysEqual(u)) = item;
                let vertex_guard = index.read(u.0);
                let vertex_bytes = vertex_guard.get(u.1)?;
                let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                let payload_u = vertex_tuple.payload()?;
                let vector_u = fetch(payload_u)?;
                Some((item, vector_u))
            }),
            m,
            alpha,
        );
        let mut vertex_guard = index.write(u.0, true);
        let Some(vertex_bytes) = vertex_guard.get_mut(u.1) else {
            continue;
        };
        let mut vertex_tuple = VertexTuple::deserialize_mut(vertex_bytes);
        let filling = outs
            .iter()
            .map(|&(Reverse(dis_v), AlwaysEqual(v))| Neighbour::new(dis_v, v))
            .chain(repeat(Neighbour::NULL));
        for (hole, fill) in zip(vertex_tuple.neighbours().iter_mut(), filling) {
            *hole = fill;
        }
    }
}
