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

use crate::index::vamana::opclass::Opfamily;
use algo::prefetcher::*;
use algo::*;
use half::f16;
use std::num::NonZero;
use vamana::Opaque;
use vamana::operator::{L2, Op};
use vamana::types::*;
use vector::VectorOwned;
use vector::vect::{VectBorrowed, VectOwned};

pub fn bulkdelete<R>(
    opfamily: Opfamily,
    index: &R,
    check: impl Fn(),
    callback: impl Fn(NonZero<u64>) -> bool,
) where
    R: RelationRead + RelationWrite + RelationLength,
    R::Page: Page<Opaque = Opaque>,
{
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => {
            vamana::bulkdelete::<_, Op<VectOwned<f32>, L2>>(index, &check, &callback);
        }
        (VectorKind::Vecf16, DistanceKind::L2) => {
            vamana::bulkdelete::<_, Op<VectOwned<f16>, L2>>(index, &check, &callback);
        }
    }
}

pub fn maintain<R>(opfamily: Opfamily, index: &R, check: impl Fn())
where
    R: RelationRead + RelationWrite,
    R::Page: Page<Opaque = Opaque>,
{
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => {
            vamana::maintain::<_, Op<VectOwned<f32>, L2>>(index, &check);
        }
        (VectorKind::Vecf16, DistanceKind::L2) => {
            vamana::maintain::<_, Op<VectOwned<f16>, L2>>(index, &check);
        }
    }
}

pub fn build<R>(vector_options: VectorOptions, vamana_options: VamanaIndexOptions, index: &R)
where
    R: RelationRead + RelationWrite,
    R::Page: Page<Opaque = Opaque>,
{
    match (vector_options.v, vector_options.d) {
        (VectorKind::Vecf32, DistanceKind::L2) => {
            vamana::build::<_, Op<VectOwned<f32>, L2>>(vector_options, vamana_options, index)
        }
        (VectorKind::Vecf16, DistanceKind::L2) => {
            vamana::build::<_, Op<VectOwned<f16>, L2>>(vector_options, vamana_options, index)
        }
    }
}

pub fn insert<R>(opfamily: Opfamily, index: &R, payload: NonZero<u64>, vector: OwnedVector)
where
    R: RelationRead + RelationWrite + RelationReadStream,
    R::Page: Page<Opaque = Opaque>,
{
    let make_vertex_stream_prefetcher = MakeStreamPrefetcher {
        index,
        hints: Hints::default().full(true),
    };
    let make_vector_stream_prefetcher = MakeStreamPrefetcher {
        index,
        hints: Hints::default().full(true),
    };
    match (vector, opfamily.distance_kind()) {
        (OwnedVector::Vecf32(unprojected), DistanceKind::L2) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf32);
            let projected = RandomProject::project(unprojected.as_borrowed());
            vamana::insert::<_, Op<VectOwned<f32>, L2>>(
                index,
                projected.as_borrowed(),
                payload,
                make_vertex_stream_prefetcher,
                make_vector_stream_prefetcher,
            )
        }
        (OwnedVector::Vecf16(unprojected), DistanceKind::L2) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf16);
            let projected = RandomProject::project(unprojected.as_borrowed());
            vamana::insert::<_, Op<VectOwned<f16>, L2>>(
                index,
                projected.as_borrowed(),
                payload,
                make_vertex_stream_prefetcher,
                make_vector_stream_prefetcher,
            )
        }
    }
}

pub trait RandomProject {
    type Output;
    fn project(self) -> Self::Output;
}

impl RandomProject for VectBorrowed<'_, f32> {
    type Output = VectOwned<f32>;
    fn project(self) -> VectOwned<f32> {
        use rabitq::rotate::rotate;
        let input = self.slice();
        VectOwned::new(rotate(input))
    }
}

impl RandomProject for VectBorrowed<'_, f16> {
    type Output = VectOwned<f16>;
    fn project(self) -> VectOwned<f16> {
        use rabitq::rotate::rotate;
        use simd::Floating;
        let input = f16::vector_to_f32(self.slice());
        VectOwned::new(f16::vector_from_f32(&rotate(&input)))
    }
}

#[derive(Debug)]
pub struct MakeStreamPrefetcher<'r, R> {
    pub index: &'r R,
    pub hints: Hints,
}

impl<'r, R> Clone for MakeStreamPrefetcher<'r, R> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            hints: self.hints.clone(),
        }
    }
}

impl<'r, R: RelationRead + RelationReadStream> PrefetcherSequenceFamily<'r, R>
    for MakeStreamPrefetcher<'r, R>
{
    type P<S: Sequence>
        = StreamPrefetcher<'r, R, S>
    where
        S::Item: Fetch;

    fn prefetch<S: Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: Fetch,
    {
        StreamPrefetcher::new(self.index, seq, self.hints.clone())
    }
}
