use crate::operator::{Accessor2, Operator, Vector};
use crate::results::Results;
use crate::tuples::{MetaTuple, VertexTuple, WithReader};
use crate::visited::Visited;
use crate::{Id, Opaque};
use algo::prefetcher::{Prefetcher, PrefetcherSequenceFamily};
use algo::{Page, RelationRead};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

#[allow(clippy::type_complexity)]
pub fn search<'r, 'b: 'r, R: RelationRead, O: Operator>(
    index: &'r R,
    vector: <O::Vector as VectorOwned>::Borrowed<'b>,
    raw: <O::Vector as VectorOwned>::Borrowed<'b>,
    _epsilon: f32,
    ef_search: u32,
    mut fetch: impl FnMut(NonZero<u64>) -> Option<O::Vector> + 'b,
    mut prefetch_vertexs: impl PrefetcherSequenceFamily<'r, R> + 'b,
) -> Box<dyn Iterator<Item = (Distance, AlwaysEqual<NonZero<u64>>)> + 'b>
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let start = meta_tuple.start();
    assert_eq!(dims, vector.dims(), "unmatched dimensions");
    let ef = ef_search;
    drop(meta_guard);
    let lut = O::Vector::binary_preprocess(vector);
    let binary_process = move |value, code, lut| {
        let (rough,) = O::binary_process(dims, value, code, lut);
        Distance::from_f32(rough)
    };
    let mut visited = Visited::new();
    let mut candidates = BinaryHeap::<(
        Reverse<Distance>,
        AlwaysEqual<(Option<NonZero<u64>>, Vec<Id>)>,
    )>::new();
    let Some(s) = start.into_inner() else {
        return Box::new(std::iter::empty());
    };
    {
        visited.insert(s);
        let vertex_guard = index.read(s.0);
        let Some(vertex_bytes) = vertex_guard.get(s.1) else {
            return Box::new(std::iter::empty());
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
        candidates.push((Reverse(score_s), AlwaysEqual((payload_s, outs_s))));
    }
    let mut iter = std::iter::from_fn(move || {
        while let Some((_, AlwaysEqual((payload_u, outs_u)))) = candidates.pop() {
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
                let payload_v = vertex_tuple.payload();
                let outs_v = vertex_tuple
                    .neighbours()
                    .iter()
                    .flat_map(|neighbour| neighbour.into_inner())
                    .map(|(_, v)| v)
                    .collect();
                candidates.push((Reverse(score_v), AlwaysEqual((payload_v, outs_v))));
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
            return Some((Reverse(dis_u), AlwaysEqual(payload_u)));
        }
        None
    });
    let mut results = Results::new(ef as _);
    let search = std::iter::from_fn(move || {
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
        results.pop_min()
    });
    Box::new(search)
}
