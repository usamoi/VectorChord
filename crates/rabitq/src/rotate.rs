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

const BITS: &[u8; 262144] = include_bytes!(concat!(env!("OUT_DIR"), "/bits"));

const _: () = assert!(BITS[0] == 246);
const _: () = assert!(BITS[1] == 133);
const _: () = assert!(BITS[2] == 163);
const _: () = assert!(BITS[3] == 106);
const _: () = assert!(BITS[4] == 54);
const _: () = assert!(BITS[5] == 126);
const _: () = assert!(BITS[6] == 9);
const _: () = assert!(BITS[7] == 115);

static BITS_0: [u64; 1024] = zerocopy::transmute!(BITS.as_chunks::<8192>().0[0]);
static BITS_1: [u64; 1024] = zerocopy::transmute!(BITS.as_chunks::<8192>().0[1]);
static BITS_2: [u64; 1024] = zerocopy::transmute!(BITS.as_chunks::<8192>().0[2]);
static BITS_3: [u64; 1024] = zerocopy::transmute!(BITS.as_chunks::<8192>().0[3]);

fn kacs_walk(result: &mut [f32]) {
    let n = result.len();
    let m = n / 2;
    let (l, t) = result.split_at_mut(m);
    let (_, r) = t.split_at_mut(n - 2 * m);
    simd::rotate::givens(l, r);
}

#[expect(dead_code)]
fn rotate_1(vector: &[f32]) -> Vec<f32> {
    use simd::Floating;
    use std::ops::Bound::{Excluded, Included, Unbounded};

    let mut result = vector.to_vec();
    let n = vector.len();
    let base = n.ilog2();
    let scale = 1.0 / ((1_usize << base) as f32).sqrt();

    let l = (Unbounded, Excluded(1_usize << base));
    let r = (Included(n - (1_usize << base)), Unbounded);

    simd::rotate::flip(&BITS_0, &mut result);
    simd::fht::fht(&mut result[l]);
    f32::vector_mul_scalar_inplace(&mut result[l], scale);
    if n != (1_usize << base) {
        kacs_walk(&mut result);
    }

    simd::rotate::flip(&BITS_1, &mut result);
    simd::fht::fht(&mut result[r]);
    f32::vector_mul_scalar_inplace(&mut result[r], scale);
    if n != (1_usize << base) {
        kacs_walk(&mut result);
    }

    simd::rotate::flip(&BITS_2, &mut result);
    simd::fht::fht(&mut result[l]);
    f32::vector_mul_scalar_inplace(&mut result[l], scale);
    if n != (1_usize << base) {
        kacs_walk(&mut result);
    }

    simd::rotate::flip(&BITS_3, &mut result);
    simd::fht::fht(&mut result[r]);
    f32::vector_mul_scalar_inplace(&mut result[r], scale);
    if n != (1_usize << base) {
        kacs_walk(&mut result);
    }

    result
}

pub fn rotate(vector: &[f32]) -> Vec<f32> {
    fn random_full_rank_matrix(n: usize) -> nalgebra::DMatrix<f32> {
        use nalgebra::DMatrix;
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha12Rng;
        use rand_distr::StandardNormal;
        let mut rng = ChaCha12Rng::from_seed([7; 32]);
        DMatrix::from_fn(n, n, |_, _| rng.sample(StandardNormal))
    }

    fn random_orthogonal_matrix(n: usize) -> Vec<Vec<f32>> {
        use nalgebra::QR;
        let matrix = random_full_rank_matrix(n);
        // QR decomposition is unique if the matrix is full rank
        let qr = QR::new(matrix);
        let q = qr.q();
        let mut projection = Vec::new();
        for row in q.row_iter() {
            projection.push(row.iter().copied().collect::<Vec<f32>>());
        }
        projection
    }

    use simd::Floating;
    use std::sync::LazyLock;
    match vector.len() {
        768 => {
            static MATRIX: LazyLock<Vec<Vec<f32>>> =
                LazyLock::new(|| random_orthogonal_matrix(768));
            #[ctor::ctor]
            fn init() {
                LazyLock::force(&MATRIX);
            }
            (0..768)
                .map(|i| f32::reduce_sum_of_xy(vector, &MATRIX[i]))
                .collect()
        }
        1024 => {
            static MATRIX: LazyLock<Vec<Vec<f32>>> =
                LazyLock::new(|| random_orthogonal_matrix(1024));
            #[ctor::ctor]
            fn init() {
                LazyLock::force(&MATRIX);
            }
            (0..1024)
                .map(|i| f32::reduce_sum_of_xy(vector, &MATRIX[i]))
                .collect()
        }
        _ => unimplemented!(),
    }
}
