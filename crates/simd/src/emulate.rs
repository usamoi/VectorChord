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

// VP2INTERSECT emulation.
// Díez-Cañas, G. (2021). Faster-Than-Native Alternatives for x86 VP2INTERSECT
// Instructions. arXiv preprint arXiv:2112.06342.
#[inline]
#[cfg(target_arch = "x86_64")]
#[crate::target_cpu(enable = "v4")]
pub fn emulate_mm512_2intersect_epi32(
    a: std::arch::x86_64::__m512i,
    b: std::arch::x86_64::__m512i,
) -> (std::arch::x86_64::__mmask16, std::arch::x86_64::__mmask16) {
    use std::arch::x86_64::*;

    let a1 = _mm512_alignr_epi32(a, a, 4);
    let a2 = _mm512_alignr_epi32(a, a, 8);
    let a3 = _mm512_alignr_epi32(a, a, 12);
    let b1 = _mm512_shuffle_epi32(b, _MM_PERM_ADCB);
    let b2 = _mm512_shuffle_epi32(b, _MM_PERM_BADC);
    let b3 = _mm512_shuffle_epi32(b, _MM_PERM_CBAD);
    let m00 = _mm512_cmpeq_epi32_mask(a, b);
    let m01 = _mm512_cmpeq_epi32_mask(a, b1);
    let m02 = _mm512_cmpeq_epi32_mask(a, b2);
    let m03 = _mm512_cmpeq_epi32_mask(a, b3);
    let m10 = _mm512_cmpeq_epi32_mask(a1, b);
    let m11 = _mm512_cmpeq_epi32_mask(a1, b1);
    let m12 = _mm512_cmpeq_epi32_mask(a1, b2);
    let m13 = _mm512_cmpeq_epi32_mask(a1, b3);
    let m20 = _mm512_cmpeq_epi32_mask(a2, b);
    let m21 = _mm512_cmpeq_epi32_mask(a2, b1);
    let m22 = _mm512_cmpeq_epi32_mask(a2, b2);
    let m23 = _mm512_cmpeq_epi32_mask(a2, b3);
    let m30 = _mm512_cmpeq_epi32_mask(a3, b);
    let m31 = _mm512_cmpeq_epi32_mask(a3, b1);
    let m32 = _mm512_cmpeq_epi32_mask(a3, b2);
    let m33 = _mm512_cmpeq_epi32_mask(a3, b3);

    let m0 = m00 | m10 | m20 | m30;
    let m1 = m01 | m11 | m21 | m31;
    let m2 = m02 | m12 | m22 | m32;
    let m3 = m03 | m13 | m23 | m33;

    let res_a = m00
        | m01
        | m02
        | m03
        | (m10 | m11 | m12 | m13).rotate_left(4)
        | (m20 | m21 | m22 | m23).rotate_left(8)
        | (m30 | m31 | m32 | m33).rotate_right(4);

    let res_b = m0
        | ((0x7777 & m1) << 1)
        | ((m1 >> 3) & 0x1111)
        | ((0x3333 & m2) << 2)
        | ((m2 >> 2) & 0x3333)
        | ((0x1111 & m3) << 3)
        | ((m3 >> 1) & 0x7777);
    (res_a, res_b)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[crate::target_cpu(enable = "v3")]
pub fn emulate_mm256_reduce_add_ps(mut x: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    x = _mm256_add_ps(x, _mm256_permute2f128_ps(x, x, 1));
    x = _mm256_hadd_ps(x, x);
    x = _mm256_hadd_ps(x, x);
    _mm256_cvtss_f32(x)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[crate::target_cpu(enable = "v2")]
pub fn emulate_mm_reduce_add_ps(mut x: std::arch::x86_64::__m128) -> f32 {
    use std::arch::x86_64::*;
    x = _mm_hadd_ps(x, x);
    x = _mm_hadd_ps(x, x);
    _mm_cvtss_f32(x)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[crate::target_cpu(enable = "v4")]
pub fn emulate_mm512_reduce_add_epi16(x: std::arch::x86_64::__m512i) -> i16 {
    use std::arch::x86_64::*;
    i16::wrapping_add(
        _mm256_reduce_add_epi16(_mm512_castsi512_si256(x)),
        _mm256_reduce_add_epi16(_mm512_extracti32x8_epi32(x, 1)),
    )
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[crate::target_cpu(enable = "v3")]
pub fn emulate_mm256_reduce_add_epi16(mut x: std::arch::x86_64::__m256i) -> i16 {
    use std::arch::x86_64::*;
    x = _mm256_add_epi16(x, _mm256_permute2f128_si256(x, x, 1));
    x = _mm256_hadd_epi16(x, x);
    x = _mm256_hadd_epi16(x, x);
    let x = _mm256_cvtsi256_si32(x);
    i16::wrapping_add(x as i16, (x >> 16) as i16)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[crate::target_cpu(enable = "v2")]
pub fn emulate_mm_reduce_add_epi16(mut x: std::arch::x86_64::__m128i) -> i16 {
    use std::arch::x86_64::*;
    x = _mm_hadd_epi16(x, x);
    x = _mm_hadd_epi16(x, x);
    let x = _mm_cvtsi128_si32(x);
    i16::wrapping_add(x as i16, (x >> 16) as i16)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[crate::target_cpu(enable = "v3")]
pub fn emulate_mm256_reduce_add_epi32(mut x: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    x = _mm256_add_epi32(x, _mm256_permute2f128_si256(x, x, 1));
    x = _mm256_hadd_epi32(x, x);
    x = _mm256_hadd_epi32(x, x);
    _mm256_cvtsi256_si32(x)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[crate::target_cpu(enable = "v2")]
pub fn emulate_mm_reduce_add_epi32(mut x: std::arch::x86_64::__m128i) -> i32 {
    use std::arch::x86_64::*;
    x = _mm_hadd_epi32(x, x);
    x = _mm_hadd_epi32(x, x);
    _mm_cvtsi128_si32(x)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[crate::target_cpu(enable = "v3")]
pub fn emulate_mm256_reduce_min_ps(x: std::arch::x86_64::__m256) -> f32 {
    use crate::aligned::Aligned16;
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(x);
    let hi = _mm256_extractf128_ps(x, 1);
    let min = _mm_min_ps(lo, hi);
    let mut x = Aligned16([0.0f32; 4]);
    unsafe {
        _mm_store_ps(x.0.as_mut_ptr(), min);
    }
    f32::min(f32::min(x.0[0], x.0[1]), f32::min(x.0[2], x.0[3]))
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[crate::target_cpu(enable = "v2")]
pub fn emulate_mm_reduce_min_ps(x: std::arch::x86_64::__m128) -> f32 {
    use crate::aligned::Aligned16;
    use std::arch::x86_64::*;
    let min = x;
    let mut x = Aligned16([0.0f32; 4]);
    unsafe {
        _mm_store_ps(x.0.as_mut_ptr(), min);
    }
    f32::min(f32::min(x.0[0], x.0[1]), f32::min(x.0[2], x.0[3]))
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[crate::target_cpu(enable = "v3")]
pub fn emulate_mm256_reduce_max_ps(x: std::arch::x86_64::__m256) -> f32 {
    use crate::aligned::Aligned16;
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(x);
    let hi = _mm256_extractf128_ps(x, 1);
    let max = _mm_max_ps(lo, hi);
    let mut x = Aligned16([0.0f32; 4]);
    unsafe {
        _mm_store_ps(x.0.as_mut_ptr(), max);
    }
    f32::max(f32::max(x.0[0], x.0[1]), f32::max(x.0[2], x.0[3]))
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[crate::target_cpu(enable = "v2")]
pub fn emulate_mm_reduce_max_ps(x: std::arch::x86_64::__m128) -> f32 {
    use crate::aligned::Aligned16;
    use std::arch::x86_64::*;
    let max = x;
    let mut x = Aligned16([0.0f32; 4]);
    unsafe {
        _mm_store_ps(x.0.as_mut_ptr(), max);
    }
    f32::max(f32::max(x.0[0], x.0[1]), f32::max(x.0[2], x.0[3]))
}
