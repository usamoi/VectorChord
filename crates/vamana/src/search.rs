use crate::checker::Checker;
use crate::operator::{Operator, Vector};
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
    _epsilon: f32,
    ef_search: u32,
    mut prefetch_vertexs: impl PrefetcherSequenceFamily<'r, R> + 'b,
) -> Box<dyn Iterator<Item = (Reverse<Distance>, AlwaysEqual<NonZero<u64>>)> + 'b>
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
    let binary_process =
        move |value, code, lut| Distance::from_f32(O::binary_process(dims, value, code, lut));
    let mut visited = Visited::new();
    let mut candidates = BinaryHeap::<(
        Reverse<Distance>,
        AlwaysEqual<(Option<NonZero<u64>>, Vec<Id>)>,
    )>::new();
    let mut checker = Checker::new(ef as _);
    let Some(s) = start.id() else {
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
            .flat_map(|t| t.id())
            .collect();
        candidates.push((Reverse(score_s), AlwaysEqual((payload_s, outs_s))));
    }
    let mut iter = std::iter::from_fn(move || {
        while let Some((Reverse(dis_u), AlwaysEqual((payload_u, outs_u)))) = candidates.pop() {
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
                    .flat_map(|t| t.id())
                    .collect();
                candidates.push((Reverse(score_v), AlwaysEqual((payload_v, outs_v))));
            }
            let Some(payload_u) = payload_u else {
                continue;
            };
            return Some((Reverse(dis_u), AlwaysEqual(payload_u)));
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
    let mut iter = iter.peekable();
    Box::new(std::iter::from_fn(move || {
        if leading.last() > iter.peek() {
            leading.pop()
        } else {
            iter.next()
        }
    }))
}
