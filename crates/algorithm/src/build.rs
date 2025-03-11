use crate::operator::{Accessor2, Operator, Vector};
use crate::tape::TapeWriter;
use crate::tuples::*;
use crate::types::*;
use crate::{Branch, DerefMut, IndexPointer, Page, PageGuard, RelationWrite};
use simd::fast_scan::{any_pack, padding_pack};
use vector::VectorOwned;

pub fn build<O: Operator>(
    vector_options: VectorOptions,
    vchordrq_options: VchordrqIndexOptions,
    index: impl RelationWrite,
    structures: Vec<Structure<O::Vector>>,
) {
    let dims = vector_options.dims;
    let is_residual = vchordrq_options.residual_quantization && O::SUPPORTS_RESIDUAL;
    let mut meta = TapeWriter::<_, _, MetaTuple>::create(|| index.extend(false));
    assert_eq!(meta.first(), 0);
    let freepage = TapeWriter::<_, _, FreepageTuple>::create(|| index.extend(false));
    let mut vectors = TapeWriter::<_, _, VectorTuple<O::Vector>>::create(|| index.extend(true));
    let mut pointer_of_means = Vec::<Vec<IndexPointer>>::new();
    for i in 0..structures.len() {
        let mut level = Vec::new();
        for j in 0..structures[i].len() {
            let vector = structures[i].means[j].as_borrowed();
            let (metadata, slices) = O::Vector::vector_split(vector);
            let mut chain = Ok(metadata);
            for i in (0..slices.len()).rev() {
                chain = Err(vectors.push(match chain {
                    Ok(metadata) => VectorTuple::_0 {
                        payload: None,
                        elements: slices[i].to_vec(),
                        metadata,
                    },
                    Err(pointer) => VectorTuple::_1 {
                        payload: None,
                        elements: slices[i].to_vec(),
                        pointer,
                    },
                }));
            }
            level.push(chain.err().unwrap());
        }
        pointer_of_means.push(level);
    }
    let mut pointer_of_firsts = Vec::<Vec<u32>>::new();
    for i in 0..structures.len() {
        let mut level = Vec::new();
        for j in 0..structures[i].len() {
            if i == 0 {
                let frozen_tape = TapeWriter::<_, _, FrozenTuple>::create(|| index.extend(false));
                let appendable_tape =
                    TapeWriter::<_, _, AppendableTuple>::create(|| index.extend(false));
                let mut jump = TapeWriter::<_, _, JumpTuple>::create(|| index.extend(false));
                jump.push(JumpTuple {
                    frozen_first: frozen_tape.first(),
                    appendable_first: appendable_tape.first(),
                    tuples: 0,
                });
                level.push(jump.first());
            } else {
                let mut tape = H1TapeWriter::<_, _>::create(|| index.extend(false));
                let h2_mean = structures[i].means[j].as_borrowed();
                let h2_children = structures[i].children[j].as_slice();
                for child in h2_children.iter().copied() {
                    let h1_mean = structures[i - 1].means[child as usize].as_borrowed();
                    let code = if is_residual {
                        let mut residual_accessor = O::ResidualAccessor::default();
                        residual_accessor.push(
                            O::Vector::elements_and_metadata(h1_mean).0,
                            O::Vector::elements_and_metadata(h2_mean).0,
                        );
                        let residual = residual_accessor.finish(
                            O::Vector::elements_and_metadata(h1_mean).1,
                            O::Vector::elements_and_metadata(h2_mean).1,
                        );
                        O::Vector::code(residual.as_borrowed())
                    } else {
                        O::Vector::code(h1_mean)
                    };
                    tape.push(Branch {
                        mean: pointer_of_means[i - 1][child as usize],
                        dis_u_2: code.dis_u_2,
                        factor_ppc: code.factor_ppc,
                        factor_ip: code.factor_ip,
                        factor_err: code.factor_err,
                        signs: code.signs,
                        extra: pointer_of_firsts[i - 1][child as usize],
                    });
                }
                let (mut tape, branches) = tape.into_inner();
                if !branches.is_empty() {
                    let mut remain =
                        padding_pack(branches.iter().map(|x| rabitq::pack_to_u4(&x.signs)));
                    loop {
                        let freespace = tape.freespace();
                        if H1Tuple::estimate_size_0(remain.len()) <= freespace as usize {
                            tape.tape_put(H1Tuple::_0 {
                                mean: any_pack(branches.iter().map(|x| x.mean)),
                                dis_u_2: any_pack(branches.iter().map(|x| x.dis_u_2)),
                                factor_ppc: any_pack(branches.iter().map(|x| x.factor_ppc)),
                                factor_ip: any_pack(branches.iter().map(|x| x.factor_ip)),
                                factor_err: any_pack(branches.iter().map(|x| x.factor_err)),
                                first: any_pack(branches.iter().map(|x| x.extra)),
                                len: branches.len() as _,
                                elements: remain,
                            });
                            break;
                        }
                        if let Some(w) = H1Tuple::fit_1(freespace) {
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
        root_mean: pointer_of_means.last().unwrap()[0],
        root_first: pointer_of_firsts.last().unwrap()[0],
        freepage_first: freepage.first(),
    });
}

pub struct H1TapeWriter<G, E> {
    tape: TapeWriter<G, E, H1Tuple>,
    branches: Vec<Branch<u32>>,
}

impl<G, E> H1TapeWriter<G, E>
where
    G: PageGuard + DerefMut,
    G::Target: Page,
    E: Fn() -> G,
{
    fn create(extend: E) -> Self {
        Self {
            tape: TapeWriter::create(extend),
            branches: Vec::new(),
        }
    }
    fn push(&mut self, branch: Branch<u32>) {
        self.branches.push(branch);
        if self.branches.len() == 32 {
            let chunk = std::array::from_fn::<_, 32, _>(|_| self.branches.pop().unwrap());
            let mut remain = padding_pack(chunk.iter().map(|x| rabitq::pack_to_u4(&x.signs)));
            loop {
                let freespace = self.tape.freespace();
                if H1Tuple::estimate_size_0(remain.len()) <= freespace as usize {
                    self.tape.tape_put(H1Tuple::_0 {
                        mean: chunk.each_ref().map(|x| x.mean),
                        dis_u_2: chunk.each_ref().map(|x| x.dis_u_2),
                        factor_ppc: chunk.each_ref().map(|x| x.factor_ppc),
                        factor_ip: chunk.each_ref().map(|x| x.factor_ip),
                        factor_err: chunk.each_ref().map(|x| x.factor_err),
                        first: chunk.each_ref().map(|x| x.extra),
                        len: chunk.len() as _,
                        elements: remain,
                    });
                    break;
                }
                if let Some(w) = H1Tuple::fit_1(freespace) {
                    let (left, right) = remain.split_at(std::cmp::min(w, remain.len()));
                    self.tape.tape_put(H1Tuple::_1 {
                        elements: left.to_vec(),
                    });
                    remain = right.to_vec();
                } else {
                    self.tape.tape_move();
                }
            }
        }
    }
    fn into_inner(self) -> (TapeWriter<G, E, H1Tuple>, Vec<Branch<u32>>) {
        (self.tape, self.branches)
    }
}
