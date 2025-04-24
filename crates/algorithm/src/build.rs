use crate::operator::{Accessor2, Operator, Vector};
use crate::tape::TapeWriter;
use crate::tuples::*;
use crate::types::*;
use crate::{Branch, RelationWrite};
use simd::fast_scan::{any_pack, padding_pack};
use vector::VectorOwned;

pub fn build<O: Operator>(
    vector_options: VectorOptions,
    vchordrq_options: VchordrqIndexOptions,
    index: impl RelationWrite,
    structures: Vec<Structure<O::Vector>>,
) {
    if vchordrq_options.residual_quantization && !O::SUPPORTS_RESIDUAL {
        panic!("residual_quantization can be enabled only if distance type is L2");
    }
    let dims = vector_options.dims;
    let is_residual = vchordrq_options.residual_quantization;
    let mut meta = TapeWriter::<_, MetaTuple>::create(&index, false);
    assert_eq!(meta.first(), 0);
    let freepage = TapeWriter::<_, FreepageTuple>::create(&index, false);
    let mut vectors = TapeWriter::<_, VectorTuple<O::Vector>>::create(&index, true);
    let mut pointer_of_means = Vec::<Vec<(Vec<u32>, u16)>>::new();
    for i in 0..structures.len() {
        let mut level = Vec::new();
        for j in 0..structures[i].len() {
            let vector = structures[i].means[j].as_borrowed();
            let (slices, metadata) = O::Vector::split(vector);
            let mut chain = Ok(metadata);
            let mut prefetch = Vec::new();
            for i in (0..slices.len()).rev() {
                let (id, head) = vectors.push(match chain {
                    Ok(metadata) => VectorTuple::_0 {
                        payload: None,
                        elements: slices[i].to_vec(),
                        metadata,
                    },
                    Err(head) => VectorTuple::_1 {
                        payload: None,
                        elements: slices[i].to_vec(),
                        head,
                    },
                });
                chain = Err(head);
                prefetch.push(id);
            }
            prefetch.reverse();
            level.push((
                prefetch,
                chain.expect_err("internal error: 0-dimensional vector"),
            ));
        }
        pointer_of_means.push(level);
    }
    let mut pointer_of_firsts = Vec::<Vec<u32>>::new();
    for i in 0..structures.len() {
        let mut level = Vec::new();
        for j in 0..structures[i].len() {
            if i == 0 {
                let frozen_tape = TapeWriter::<_, FrozenTuple>::create(&index, false);
                let appendable_tape = TapeWriter::<_, AppendableTuple>::create(&index, false);
                let mut jump = TapeWriter::<_, JumpTuple>::create(&index, false);
                jump.push(JumpTuple {
                    frozen_first: frozen_tape.first(),
                    appendable_first: appendable_tape.first(),
                    tuples: 0,
                });
                level.push(jump.first());
            } else {
                let mut tape = H1TapeWriter::create(&index, O::Vector::count(dims as _), false);
                let h2_mean = structures[i].means[j].as_borrowed();
                let h2_children = structures[i].children[j].as_slice();
                for child in h2_children.iter().copied() {
                    let h1_mean = structures[i - 1].means[child as usize].as_borrowed();
                    let code = if is_residual {
                        let mut residual_accessor = O::ResidualAccessor::default();
                        residual_accessor
                            .push(O::Vector::unpack(h1_mean).0, O::Vector::unpack(h2_mean).0);
                        let residual = residual_accessor
                            .finish(O::Vector::unpack(h1_mean).1, O::Vector::unpack(h2_mean).1);
                        O::Vector::code(residual.as_borrowed())
                    } else {
                        O::Vector::code(h1_mean)
                    };
                    tape.push(Branch {
                        head: pointer_of_means[i - 1][child as usize].1.clone(),
                        dis_u_2: code.dis_u_2,
                        factor_ppc: code.factor_ppc,
                        factor_ip: code.factor_ip,
                        factor_err: code.factor_err,
                        signs: code.signs,
                        prefetch: pointer_of_means[i - 1][child as usize].0.clone(),
                        extra: pointer_of_firsts[i - 1][child as usize],
                    });
                }
                let (mut tape, chunk) = tape.into_inner();
                if !chunk.is_empty() {
                    let mut remain =
                        padding_pack(chunk.iter().map(|x| rabitq::pack_to_u4(&x.signs)));
                    loop {
                        let freespace = tape.freespace();
                        if H1Tuple::estimate_size_0(O::Vector::count(dims as _), remain.len())
                            <= freespace as usize
                        {
                            tape.tape_put(H1Tuple::_0 {
                                head: any_pack(chunk.iter().map(|x| x.head)),
                                dis_u_2: any_pack(chunk.iter().map(|x| x.dis_u_2)),
                                factor_ppc: any_pack(chunk.iter().map(|x| x.factor_ppc)),
                                factor_ip: any_pack(chunk.iter().map(|x| x.factor_ip)),
                                factor_err: any_pack(chunk.iter().map(|x| x.factor_err)),
                                first: any_pack(chunk.iter().map(|x| x.extra)),
                                prefetch: fix(chunk.iter().map(|x| x.prefetch.as_slice())),
                                len: chunk.len() as _,
                                elements: remain,
                            });
                            break;
                        }
                        if let Some(w) = H1Tuple::fit_1(O::Vector::count(dims as _), freespace) {
                            let (left, right) = remain.split_at(std::cmp::min(w, remain.len()));
                            tape.tape_put(H1Tuple::_1 {
                                elements: left.to_vec(),
                            });
                            remain = right.to_vec();
                        } else {
                            tape.tape_move();
                        }
                    }
                }
                level.push(tape.first());
            }
        }
        pointer_of_firsts.push(level);
    }
    meta.push(MetaTuple {
        dims,
        height_of_root: structures.len() as u32,
        is_residual,
        rerank_in_heap: vchordrq_options.rerank_in_table,
        vectors_first: vectors.first(),
        root_prefetch: pointer_of_means
            .last()
            .expect("internal error: empty structure")[0]
            .0
            .clone(),
        root_head: pointer_of_means
            .last()
            .expect("internal error: empty structure")[0]
            .1,
        root_first: pointer_of_firsts
            .last()
            .expect("internal error: empty structure")[0],
        freepage_first: freepage.first(),
        cells: structures.iter().map(|s| s.len() as _).collect(),
    });
}

pub struct H1TapeWriter<'a, R>
where
    R: RelationWrite + 'a,
{
    tape: TapeWriter<'a, R, H1Tuple>,
    branches: Vec<Branch<u32>>,
    prefetch: usize,
}

impl<'a, R> H1TapeWriter<'a, R>
where
    R: RelationWrite + 'a,
{
    fn create(index: &'a R, prefetch: usize, tracking_freespace: bool) -> Self {
        Self {
            tape: TapeWriter::create(index, tracking_freespace),
            branches: Vec::new(),
            prefetch,
        }
    }
    fn push(&mut self, branch: Branch<u32>) {
        self.branches.push(branch);
        if let Ok(chunk) = <&[_; 32]>::try_from(self.branches.as_slice()) {
            let mut remain = padding_pack(chunk.iter().map(|x| rabitq::pack_to_u4(&x.signs)));
            loop {
                let freespace = self.tape.freespace();
                if H1Tuple::estimate_size_0(self.prefetch, remain.len()) <= freespace as usize {
                    self.tape.tape_put(H1Tuple::_0 {
                        head: chunk.each_ref().map(|x| x.head),
                        dis_u_2: chunk.each_ref().map(|x| x.dis_u_2),
                        factor_ppc: chunk.each_ref().map(|x| x.factor_ppc),
                        factor_ip: chunk.each_ref().map(|x| x.factor_ip),
                        factor_err: chunk.each_ref().map(|x| x.factor_err),
                        first: chunk.each_ref().map(|x| x.extra),
                        prefetch: fix(chunk.each_ref().map(|x| x.prefetch.as_slice())),
                        len: chunk.len() as _,
                        elements: remain,
                    });
                    break;
                }
                if let Some(w) = H1Tuple::fit_1(self.prefetch, freespace) {
                    let (left, right) = remain.split_at(std::cmp::min(w, remain.len()));
                    self.tape.tape_put(H1Tuple::_1 {
                        elements: left.to_vec(),
                    });
                    remain = right.to_vec();
                } else {
                    self.tape.tape_move();
                }
            }
            self.branches.clear();
        }
    }
    fn into_inner(self) -> (TapeWriter<'a, R, H1Tuple>, Vec<Branch<u32>>) {
        (self.tape, self.branches)
    }
}

fn fix<'a>(into_iter: impl IntoIterator<Item = &'a [u32]>) -> Vec<[u32; 32]> {
    use std::array::from_fn;
    let mut iter = into_iter.into_iter();
    let mut array: [_; 32] = from_fn(|_| iter.next().map(<[u32]>::to_vec).unwrap_or_default());
    if iter.next().is_some() {
        panic!("too many slices");
    }
    let step = array.iter().map(Vec::len).max().unwrap_or_default();
    array.iter_mut().for_each(|x| x.resize(step, u32::MAX));
    let flat = array.into_iter().flatten().collect::<Vec<_>>();
    (0..step).map(|i| from_fn(|j| flat[i * 32 + j])).collect()
}
