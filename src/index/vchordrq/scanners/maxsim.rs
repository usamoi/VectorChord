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

#![allow(dead_code)]
#![allow(unused)]

use super::{SearchBuilder, SearchOptions};
use crate::index::fetcher::*;
use crate::index::vchordrq::algo::*;
use crate::index::vchordrq::opclass::Opfamily;
use crate::index::vchordrq::scanners::{Io, filter};
use algo::accessor::Dot;
use algo::prefetcher::*;
use algo::*;
use always_equal::AlwaysEqual;
use distance::Distance;
use half::f16;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vchordrq::types::{DistanceKind, OwnedVector, VectorKind};
use vchordrq::*;
use vector::VectorOwned;
use vector::vect::VectOwned;

pub struct MaxsimBuilder {
    opfamily: Opfamily,
    orderbys: Vec<Option<Vec<OwnedVector>>>,
}

impl SearchBuilder for MaxsimBuilder {
    type Opaque = vchordrq::Opaque;

    fn new(opfamily: Opfamily) -> Self {
        assert!(matches!(
            opfamily,
            Opfamily::VectorMaxsim | Opfamily::HalfvecMaxsim
        ));
        Self {
            opfamily,
            orderbys: Vec::new(),
        }
    }

    unsafe fn add(&mut self, strategy: u16, datum: Option<pgrx::pg_sys::Datum>) {
        match strategy {
            3 => {
                let x = unsafe { datum.and_then(|x| self.opfamily.input_vectors(x)) };
                self.orderbys.push(x);
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
        R::Page: Page<Opaque = vchordrq::Opaque>,
    {
        unimplemented!()
    }
}

// Emulate unstable library feature `binary_heap_into_iter_sorted`.
// See https://github.com/rust-lang/rust/issues/59278.

pub trait IntoIterSortedPolyfill<T> {
    fn into_iter_sorted_polyfill(self) -> IntoIterSorted<T>;
}

impl<T> IntoIterSortedPolyfill<T> for BinaryHeap<T> {
    fn into_iter_sorted_polyfill(self) -> IntoIterSorted<T> {
        IntoIterSorted { inner: self }
    }
}

#[derive(Clone, Debug)]
pub struct IntoIterSorted<T> {
    inner: BinaryHeap<T>,
}

impl<T: Ord> Iterator for IntoIterSorted<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner.pop()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.inner.len();
        (exact, Some(exact))
    }
}

impl<T: Ord> ExactSizeIterator for IntoIterSorted<T> {}

impl<T: Ord> std::iter::FusedIterator for IntoIterSorted<T> {}

#[inline(always)]
pub fn id_0<F, A: ?Sized, B: ?Sized, C: ?Sized, D: ?Sized, R: ?Sized>(f: F) -> F
where
    F: for<'a> FnMut(&(A, AlwaysEqual<&'a mut (B, C, &'a mut D)>)) -> R,
{
    f
}
