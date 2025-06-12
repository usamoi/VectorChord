use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::fmt::Debug;

pub fn robust_prune<T: Copy + Ord + Debug, V>(
    mut fetch: impl FnMut(T) -> Option<V>,
    mut d: impl FnMut(&V, &V) -> Distance,
    u: T,
    outs: impl Iterator<Item = (Reverse<Distance>, AlwaysEqual<T>)>,
    capacity: u32,
    _alpha: f32,
) -> Vec<(Reverse<Distance>, AlwaysEqual<T>)> {
    // V ← (V ∪ Nout(p)) \ {p}
    let mut trace = outs.collect::<Vec<_>>();
    trace.sort_by_key(|(_, AlwaysEqual(v))| *v);
    trace.dedup_by_key(|(_, AlwaysEqual(v))| *v);
    trace.retain(|(_, AlwaysEqual(v))| *v != u);
    trace.sort();
    // Nout(p) ← ∅
    let mut result = Vec::new();
    for (Reverse(dis_u), AlwaysEqual(u)) in trace.into_iter().rev() {
        if result.len() == capacity as usize {
            break;
        }
        let Some(f_u) = fetch(u) else { continue };
        let check = result
            .iter()
            .flat_map(|(_, f_v)| Some(d(&f_u, f_v)))
            .all(|dis| dis_u.to_f32() < dis.to_f32());
        if check {
            result.push(((Reverse(dis_u), AlwaysEqual(u)), f_u));
        }
    }
    result.into_iter().map(|(x, _)| x).collect()
}
