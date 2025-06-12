use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::fmt::Debug;

pub fn robust_prune<T: Copy + Ord + Debug, V>(
    mut d: impl FnMut(&V, &V) -> Distance,
    u: T,
    outs: impl Iterator<Item = ((Reverse<Distance>, AlwaysEqual<T>), V)>,
    m: u32,
    _alpha: f32,
) -> Vec<(Reverse<Distance>, AlwaysEqual<T>)> {
    // V ← (V ∪ Nout(p)) \ {p}
    let mut trace = outs.collect::<Vec<_>>();
    trace.sort_by_key(|((_, AlwaysEqual(v)), _)| *v);
    trace.dedup_by_key(|((_, AlwaysEqual(v)), _)| *v);
    trace.retain(|((_, AlwaysEqual(v)), _)| *v != u);
    trace.sort_by_key(|&((Reverse(d), _), _)| d);
    // Nout(p) ← ∅
    let mut result = Vec::new();
    let mut pruned = Vec::new();
    for ((Reverse(dis_u), AlwaysEqual(u)), vector_u) in trace {
        if result.len() == m as usize {
            break;
        }
        let check = result
            .iter()
            .map(|(_, vector_v)| d(&vector_u, vector_v))
            .all(|dis| dis_u.to_f32() < dis.to_f32());
        if check {
            result.push(((Reverse(dis_u), AlwaysEqual(u)), vector_u));
        } else {
            pruned.push(((Reverse(dis_u), AlwaysEqual(u)), vector_u));
        }
    }
    result
        .into_iter()
        .chain(pruned.into_iter())
        .map(|(x, _)| x)
        .take(m as _)
        .collect()
}
