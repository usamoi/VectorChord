use std::cmp::Reverse;

use crate::Opaque;
use crate::operator::Operator;
use crate::tuples::{MetaTuple, VertexTuple, WithReader, WithWriter};
use algo::{Page, RelationLength, RelationRead, RelationWrite};
use always_equal::AlwaysEqual;
use rabitq8::CodeMetadata;

pub fn maintain<R: RelationRead + RelationWrite + RelationLength, O: Operator>(
    index: &R,
    check: impl Fn(),
) where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let start = meta_tuple.start();
    let m = meta_tuple.m();
    let alpha = 1.0f32;
    drop(meta_guard);
    if start.is_null() {
        return;
    }
    let n = index.len();
    for id in 1..n {
        check();
        let guard = index.read(id);
        let mut input = Vec::new();
        for i in 1..=guard.len() {
            if let Some(bytes) = guard.get(i) {
                let tuple = VertexTuple::deserialize_ref(bytes);
                if let Some(payload) = tuple.payload() {
                    input.push((
                        i,
                        tuple.neighbours().to_vec(),
                        payload,
                        tuple.metadata(),
                        tuple.elements().to_vec(),
                    ));
                }
            }
        }
        if input.is_empty() {
            continue;
        }
        let mut output = Vec::new();
        for (i, neighbours, payload, metadata, elements) in input.iter() {
            let neighbours = neighbours.iter().flat_map(|neighbour| {
                let distance = neighbour.distance()?;
                let id = neighbour.id()?;
                Some((distance, id))
            });
            let mut living = Vec::new();
            let mut dead = Vec::new();
            for (distance, id) in neighbours {
                let vertex_guard = index.read(id.0);
                let Some(vertex_bytes) = vertex_guard.get(id.1) else {
                    continue;
                };
                let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                let metadata = vertex_tuple.metadata();
                if vertex_tuple.payload().is_some() {
                    living.push((Reverse(distance), AlwaysEqual(id)));
                } else {
                    dead.push(vertex_tuple.neighbours().to_vec());
                }
            }
            let mut add = living.clone();
            for (_, id) in dead.into_iter().flatten().flat_map(|neighbour| {
                let distance = neighbour.distance()?;
                let id = neighbour.id()?;
                Some((distance, id))
            }) {
                let vertex_guard = index.read(id.0);
                let Some(vertex_bytes) = vertex_guard.get(id.1) else {
                    continue;
                };
                let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                if vertex_tuple.payload().is_some() {
                    add.push((Reverse(), AlwaysEqual(id)));
                }
            }
            crate::prune::robust_prune(
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
                (id, *i),
                add.into_iter(),
                m,
                alpha,
            );
        }
    }
}
