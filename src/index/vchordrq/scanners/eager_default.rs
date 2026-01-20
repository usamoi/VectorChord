// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025 TensorChord Inc.

use crate::index::fetcher::*;
use crate::index::opclass::Sphere;
use crate::index::scanners::{Io, SearchBuilder};
use crate::index::vchordrq::dispatch::*;
use crate::index::vchordrq::opclass::Opfamily;
use crate::index::vchordrq::scanners::{SearchOptions, SearchProbes};
use crate::recorder::{Recorder, text};
use always_equal::AlwaysEqual;
use dary_heap::QuaternaryHeap as Heap;
use index::accessor::L2S;
use index::bump::Bump;
use index::prefetcher::*;
use index::relation::{Page, RelationPrefetch, RelationRead, RelationReadStream};
use std::cmp::Reverse;
use std::num::NonZero;
use vchordrq::types::{DistanceKind, OwnedVector, VectorKind};
use vchordrq::{RerankMethod, eager_default_search, how, rerank_index};
use vector::VectorOwned;
use vector::vect::VectOwned;

pub struct EagerDefaultBuilder {
    opfamily: Opfamily,
    orderbys: Vec<Option<OwnedVector>>,
    spheres: Vec<Option<Sphere<OwnedVector>>>,
}

impl SearchBuilder for EagerDefaultBuilder {
    type Options = SearchOptions;

    type Opfamily = Opfamily;

    type Opaque = vchordrq::Opaque;

    fn new(opfamily: Opfamily) -> Self {
        assert!(matches!(
            opfamily,
            Opfamily::HalfvecCosine
                | Opfamily::HalfvecIp
                | Opfamily::HalfvecL2
                | Opfamily::VectorCosine
                | Opfamily::VectorIp
                | Opfamily::VectorL2
        ));
        Self {
            opfamily,
            orderbys: Vec::new(),
            spheres: Vec::new(),
        }
    }

    unsafe fn add(&mut self, strategy: u16, datum: Option<pgrx::pg_sys::Datum>) {
        match strategy {
            1 => {
                let x = unsafe { datum.and_then(|x| self.opfamily.input_vector(x)) };
                self.orderbys.push(x);
            }
            2 => {
                let x = unsafe { datum.and_then(|x| self.opfamily.input_sphere(x)) };
                self.spheres.push(x);
            }
            _ => unreachable!(),
        }
    }

    fn build<'b, R>(
        self,
        index: &'b R,
        options: SearchOptions,
        _fetcher: impl Fetcher + 'b,
        bump: &'b impl Bump,
        recorder: impl Recorder,
    ) -> Box<dyn Iterator<Item = (f32, [u16; 3], bool)> + 'b>
    where
        R: RelationRead + RelationPrefetch + RelationReadStream,
        R::Page: Page<Opaque = vchordrq::Opaque>,
    {
        let SearchProbes::Eager((partitions, recalls, target_number, target_recall)) =
            options.probes
        else {
            unreachable!();
        };
        let mut vector = None;
        let mut threshold = None;
        let mut recheck = false;
        for orderby_vector in self.orderbys.into_iter().flatten() {
            if vector.is_none() {
                vector = Some(orderby_vector);
            } else {
                pgrx::error!("vector search with multiple vectors is not supported");
            }
        }
        for Sphere { center, radius } in self.spheres.into_iter().flatten() {
            if vector.is_none() {
                (vector, threshold) = (Some(center), Some(radius));
            } else {
                recheck = true;
            }
        }
        let opfamily = self.opfamily;
        let Some(vector) = vector else {
            return Box::new(std::iter::empty()) as Box<dyn Iterator<Item = (f32, [u16; 3], bool)>>;
        };
        let make_h1_plain_prefetcher = MakeH1PlainPrefetcher { index };
        let make_h0_plain_prefetcher = MakeH0PlainPrefetcher { index };
        let f =
            move |(Reverse(distance), AlwaysEqual(payload))| (opfamily.output(distance), payload);
        let method = how(index);
        let iter: Box<dyn Iterator<Item = (f32, NonZero<u64>)>> =
            match (opfamily.vector_kind(), opfamily.distance_kind()) {
                (VectorKind::Vecf32, DistanceKind::L2S) => {
                    type Op = vchordrq::operator::Op<VectOwned<f32>, L2S>;
                    let unprojected = if let OwnedVector::Vecf32(vector) = vector.clone() {
                        vector
                    } else {
                        unreachable!()
                    };
                    let projected = RandomProject::project(unprojected.as_borrowed());
                    let results = match (
                        options.io_search,
                        method,
                        options.io_rerank,
                        options.prefilter,
                    ) {
                        (Io::Plain, RerankMethod::Index, Io::Plain, false) => {
                            eager_default_search::<_, Op, _, _>(
                                index,
                                projected.as_borrowed(),
                                partitions,
                                recalls,
                                target_number,
                                target_recall,
                                options.epsilon,
                                bump,
                                make_h1_plain_prefetcher,
                                make_h0_plain_prefetcher,
                                |results| {
                                    let sequence = Heap::from(results);
                                    let prefetcher = PlainPrefetcher::new(index, sequence);
                                    rerank_index::<Op, _, _, _>(unprojected.clone(), prefetcher)
                                },
                            )
                        }
                        _ => todo!(),
                    };
                    Box::new(results.into_iter().map(f))
                }
                _ => todo!(),
            };
        let iter = if let Some(threshold) = threshold {
            Box::new(iter.take_while(move |(x, _)| *x < threshold))
        } else {
            iter
        };
        let iter = if let Some(max_scan_tuples) = options.max_scan_tuples {
            Box::new(iter.take(max_scan_tuples as _))
        } else {
            iter
        };
        if recorder.is_enabled() {
            match &vector {
                OwnedVector::Vecf32(v) => {
                    recorder.send(&text::vector_out(v.as_borrowed()));
                }
                OwnedVector::Vecf16(v) => {
                    recorder.send(&text::halfvec_out(v.as_borrowed()));
                }
                OwnedVector::Rabitq8(v) => {
                    recorder.send(&text::rabitq8_out(v.as_borrowed()));
                }
                OwnedVector::Rabitq4(v) => {
                    recorder.send(&text::rabitq4_out(v.as_borrowed()));
                }
            }
        }
        Box::new(iter.map(move |(distance, pointer)| {
            let (key, _) = pointer_to_kv(pointer);
            (distance, key, recheck)
        }))
    }
}
