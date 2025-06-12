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

use super::{Fetcher, SearchBuilder, SearchOptions};
use crate::index::fetcher::{Tuple, pointer_to_kv};
use crate::index::vamana::algo::*;
use crate::index::vamana::opclass::{Opfamily, Sphere};
use algo::*;
use always_equal::AlwaysEqual;
use distance::Distance;
use half::f16;
use std::num::NonZero;
use vamana::operator::{self, L2};
use vamana::types::{DistanceKind, OwnedVector, VectorKind};
use vamana::*;
use vector::VectorOwned;
use vector::vect::VectOwned;

pub struct DefaultBuilder {
    opfamily: Opfamily,
    orderbys: Vec<Option<OwnedVector>>,
    spheres: Vec<Option<Sphere<OwnedVector>>>,
}

impl SearchBuilder for DefaultBuilder {
    type Opaque = vamana::Opaque;

    fn new(opfamily: Opfamily) -> Self {
        assert!(matches!(
            opfamily,
            Opfamily::HalfvecCosine
                | Opfamily::HalfvecL2
                | Opfamily::VectorCosine
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

    fn build<'a, R>(
        self,
        index: &'a R,
        options: SearchOptions,
        mut fetcher: impl Fetcher + 'a,
        bump: &'a impl Bump,
    ) -> Box<dyn Iterator<Item = (f32, [u16; 3], bool)> + 'a>
    where
        R: RelationRead + RelationPrefetch + RelationReadStream,
        R::Page: Page<Opaque = vamana::Opaque>,
    {
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
        let make_vertex_stream_prefetcher = MakeVertexStreamPrefetcher {
            index,
            hints: Hints::default().full(true),
        };
        let iter: Box<dyn Iterator<Item = (Distance, AlwaysEqual<NonZero<u64>>)>> =
            match (opfamily.vector_kind(), opfamily.distance_kind()) {
                (VectorKind::Vecf32, DistanceKind::L2) => {
                    type Op = operator::Op<VectOwned<f32>, L2>;
                    let unprojected = if let OwnedVector::Vecf32(vector) = vector {
                        bump.alloc(vector)
                    } else {
                        unreachable!()
                    };
                    let projected = bump.alloc(RandomProject::project(unprojected.as_borrowed()));
                    let fetch = move |payload| {
                        let (key, _) = pointer_to_kv(payload);
                        let mut tuple = fetcher.fetch(key)?;
                        let (datums, is_nulls) = tuple.build();
                        let datum = (!is_nulls[0]).then_some(datums[0]);
                        let maybe_vector = unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                        let raw = if let OwnedVector::Vecf32(vector) = maybe_vector.unwrap() {
                            vector
                        } else {
                            unreachable!()
                        };
                        Some(raw)
                    };
                    search::<_, Op>(
                        index,
                        projected.as_borrowed(),
                        unprojected.as_borrowed(),
                        options.epsilon,
                        options.ef_search,
                        fetch,
                        make_vertex_stream_prefetcher,
                    )
                }
                (VectorKind::Vecf16, DistanceKind::L2) => {
                    type Op = operator::Op<VectOwned<f16>, L2>;
                    let unprojected = if let OwnedVector::Vecf16(vector) = vector {
                        bump.alloc(vector)
                    } else {
                        unreachable!()
                    };
                    let projected = bump.alloc(RandomProject::project(unprojected.as_borrowed()));
                    let fetch = move |payload| {
                        let (key, _) = pointer_to_kv(payload);
                        let mut tuple = fetcher.fetch(key)?;
                        let (datums, is_nulls) = tuple.build();
                        let datum = (!is_nulls[0]).then_some(datums[0]);
                        let maybe_vector = unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                        let raw = if let OwnedVector::Vecf16(vector) = maybe_vector.unwrap() {
                            vector
                        } else {
                            unreachable!()
                        };
                        Some(raw)
                    };
                    search::<_, Op>(
                        index,
                        projected.as_borrowed(),
                        unprojected.as_borrowed(),
                        options.epsilon,
                        options.ef_search,
                        fetch,
                        make_vertex_stream_prefetcher,
                    )
                }
            };
        let iter = if let Some(threshold) = threshold {
            Box::new(iter.take_while(move |(distance, _)| distance.to_f32() < threshold))
        } else {
            iter
        };
        let iter = if let Some(max_scan_tuples) = options.max_scan_tuples {
            Box::new(iter.take(max_scan_tuples as _))
        } else {
            iter
        };
        Box::new(iter.map(move |(distance, AlwaysEqual(pointer))| {
            let (key, _) = pointer_to_kv(pointer);
            (opfamily.output(distance), key, recheck)
        }))
    }
}
