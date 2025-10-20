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

pub mod givens {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn givens(lhs: &mut [f32], rhs: &mut [f32]) {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let scale = 1.0 / (2.0_f32).sqrt();
        for i in 0..n {
            (lhs[i], rhs[i]) = ((lhs[i] + rhs[i]) * scale, (lhs[i] - rhs[i]) * scale);
        }
    }
}

pub fn givens(lhs: &mut [f32], rhs: &mut [f32]) {
    givens::givens(lhs, rhs)
}

pub mod flip {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn flip(bits: &[u8; 8192], result: &mut [f32]) {
        let result: &mut [u32] = zerocopy::transmute_mut!(result);
        let (arrays, remainder) = result.as_chunks_mut::<8>();
        let n = arrays.len();
        assert!(n <= 8192);
        for i in 0..n {
            arrays[i][0] ^= (bits[i] as u32) << (31 - 0);
            arrays[i][1] ^= (bits[i] as u32) << (31 - 1);
            arrays[i][2] ^= (bits[i] as u32) << (31 - 2);
            arrays[i][3] ^= (bits[i] as u32) << (31 - 3);
            arrays[i][4] ^= (bits[i] as u32) << (31 - 4);
            arrays[i][5] ^= (bits[i] as u32) << (31 - 5);
            arrays[i][6] ^= (bits[i] as u32) << (31 - 6);
            arrays[i][7] ^= (bits[i] as u32) << (31 - 7);
        }
        if remainder.len() >= 1 {
            remainder[0] ^= (bits[n] as u32) << (31 - 0);
        }
        if remainder.len() >= 2 {
            remainder[1] ^= (bits[n] as u32) << (31 - 1);
        }
        if remainder.len() >= 3 {
            remainder[2] ^= (bits[n] as u32) << (31 - 2);
        }
        if remainder.len() >= 4 {
            remainder[3] ^= (bits[n] as u32) << (31 - 3);
        }
        if remainder.len() >= 5 {
            remainder[4] ^= (bits[n] as u32) << (31 - 4);
        }
        if remainder.len() >= 6 {
            remainder[5] ^= (bits[n] as u32) << (31 - 5);
        }
        if remainder.len() >= 7 {
            remainder[6] ^= (bits[n] as u32) << (31 - 6);
        }
        if remainder.len() >= 8 {
            remainder[7] ^= (bits[n] as u32) << (31 - 7);
        }
    }
}

pub fn flip(bits: &[u8; 8192], result: &mut [f32]) {
    flip::flip(bits, result)
}
