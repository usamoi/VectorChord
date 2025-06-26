use crate::Opaque;
use crate::candidates::Candidates;
use crate::operator::{Operator, Vector};
use crate::results::Results;
use crate::tuples::*;
use crate::visited::Visited;
use algo::accessor::{Accessor1, LAccess};
use algo::prefetcher::{Prefetcher, PrefetcherSequenceFamily};
use algo::{Bump, Page, RelationRead};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::VecDeque;
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

pub fn search<'b, R: RelationRead, O: Operator>(
    index: &'b R,
    vector: <O::Vector as VectorOwned>::Borrowed<'b>,
    ef_search: u32,
    bump: &'b impl Bump,
    mut prefetch_vertexs: impl PrefetcherSequenceFamily<'b, R> + 'b,
    prefetch_vectors: impl PrefetcherSequenceFamily<'b, R> + 'b,
) -> Box<dyn Iterator<Item = (Distance, NonZero<u64>)> + 'b>
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
    let lut = O::Vector::preprocess(vector);
    let mut visited = Visited::new();
    let mut candidates = Candidates::new(1, prefetch_vectors);
    let Some(s) = start.into_inner() else {
        return Box::new(std::iter::empty());
    };
    {
        visited.insert(s);
        let vertex_guard = index.read(s.0);
        let Some(vertex_bytes) = vertex_guard.get(s.1) else {
            // the link is broken
            return Box::new(std::iter::empty());
        };
        let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
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
                let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
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
    Box::new(search.filter_map(|(dis_u, AlwaysEqual(payload_u))| Some((dis_u, payload_u?))))
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
