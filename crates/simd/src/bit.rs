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

#[inline(always)]
pub fn reduce_sum_of_and(lhs: &[u64], rhs: &[u64]) -> u32 {
    reduce_sum_of_and::reduce_sum_of_and(lhs, rhs)
}

mod reduce_sum_of_and {
    // FIXME: add manually-implemented SIMD version for AVX512 and AVX2

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    #[target_feature(enable = "avx512vpopcntdq")]
    fn reduce_sum_of_and_v4_avx512vpopcntdq(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert!(lhs.len() == rhs.len());
        use std::arch::x86_64::*;
        let mut and = _mm512_setzero_si512();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut n = lhs.len();
        while n >= 8 {
            let x = unsafe { _mm512_loadu_si512(a.cast()) };
            let y = unsafe { _mm512_loadu_si512(b.cast()) };
            a = unsafe { a.add(8) };
            b = unsafe { b.add(8) };
            n -= 8;
            and = _mm512_add_epi64(and, _mm512_popcnt_epi64(_mm512_and_si512(x, y)));
        }
        if n > 0 {
            let mask = _bzhi_u32(0xff, n as u32) as u8;
            let x = unsafe { _mm512_maskz_loadu_epi64(mask, a.cast()) };
            let y = unsafe { _mm512_maskz_loadu_epi64(mask, b.cast()) };
            and = _mm512_add_epi64(and, _mm512_popcnt_epi64(_mm512_and_si512(x, y)));
        }
        _mm512_reduce_add_epi64(and) as u32
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_and_v4_avx512vpopcntdq_test() {
        if !crate::is_cpu_detected!("v4") || !crate::is_feature_detected!("avx512vpopcntdq") {
            println!("test {} ... skipped (v4:avx512vpopcntdq)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let rhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_and_v4_avx512vpopcntdq(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert_eq!(specialized, fallback);
        }
    }

    #[crate::multiversion(@"v4:avx512vpopcntdq", "v4", "v3", "v2", "a2")]
    pub fn reduce_sum_of_and(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut and = 0;
        for i in 0..n {
            and += (lhs[i] & rhs[i]).count_ones();
        }
        and
    }
}

#[inline(always)]
pub fn reduce_sum_of_or(lhs: &[u64], rhs: &[u64]) -> u32 {
    reduce_sum_of_or::reduce_sum_of_or(lhs, rhs)
}

mod reduce_sum_of_or {
    // FIXME: add manually-implemented SIMD version for AVX512 and AVX2

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    #[target_feature(enable = "avx512vpopcntdq")]
    fn reduce_sum_of_or_v4_avx512vpopcntdq(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert!(lhs.len() == rhs.len());
        use std::arch::x86_64::*;
        let mut or = _mm512_setzero_si512();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut n = lhs.len();
        while n >= 8 {
            let x = unsafe { _mm512_loadu_si512(a.cast()) };
            let y = unsafe { _mm512_loadu_si512(b.cast()) };
            a = unsafe { a.add(8) };
            b = unsafe { b.add(8) };
            n -= 8;
            or = _mm512_add_epi64(or, _mm512_popcnt_epi64(_mm512_or_si512(x, y)));
        }
        if n > 0 {
            let mask = _bzhi_u32(0xff, n as u32) as u8;
            let x = unsafe { _mm512_maskz_loadu_epi64(mask, a.cast()) };
            let y = unsafe { _mm512_maskz_loadu_epi64(mask, b.cast()) };
            or = _mm512_add_epi64(or, _mm512_popcnt_epi64(_mm512_or_si512(x, y)));
        }
        _mm512_reduce_add_epi64(or) as u32
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_or_v4_avx512vpopcntdq_test() {
        if !crate::is_cpu_detected!("v4") || !crate::is_feature_detected!("avx512vpopcntdq") {
            println!("test {} ... skipped (v4:avx512vpopcntdq)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let rhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_or_v4_avx512vpopcntdq(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert_eq!(specialized, fallback);
        }
    }

    #[crate::multiversion(@"v4:avx512vpopcntdq", "v4", "v3", "v2", "a2")]
    pub fn reduce_sum_of_or(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut or = 0;
        for i in 0..n {
            or += (lhs[i] | rhs[i]).count_ones();
        }
        or
    }
}

#[inline(always)]
pub fn reduce_sum_of_xor(lhs: &[u64], rhs: &[u64]) -> u32 {
    reduce_sum_of_xor::reduce_sum_of_xor(lhs, rhs)
}

mod reduce_sum_of_xor {
    // FIXME: add manually-implemented SIMD version for AVX512 and AVX2

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    #[target_feature(enable = "avx512vpopcntdq")]
    fn reduce_sum_of_xor_v4_avx512vpopcntdq(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert!(lhs.len() == rhs.len());
        use std::arch::x86_64::*;
        let mut xor = _mm512_setzero_si512();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut n = lhs.len();
        while n >= 8 {
            let x = unsafe { _mm512_loadu_si512(a.cast()) };
            let y = unsafe { _mm512_loadu_si512(b.cast()) };
            a = unsafe { a.add(8) };
            b = unsafe { b.add(8) };
            n -= 8;
            xor = _mm512_add_epi64(xor, _mm512_popcnt_epi64(_mm512_xor_si512(x, y)));
        }
        if n > 0 {
            let mask = _bzhi_u32(0xff, n as u32) as u8;
            let x = unsafe { _mm512_maskz_loadu_epi64(mask, a.cast()) };
            let y = unsafe { _mm512_maskz_loadu_epi64(mask, b.cast()) };
            xor = _mm512_add_epi64(xor, _mm512_popcnt_epi64(_mm512_xor_si512(x, y)));
        }
        _mm512_reduce_add_epi64(xor) as u32
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_xor_v4_avx512vpopcntdq_test() {
        if !crate::is_cpu_detected!("v4") || !crate::is_feature_detected!("avx512vpopcntdq") {
            println!("test {} ... skipped (v4:avx512vpopcntdq)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let rhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_xor_v4_avx512vpopcntdq(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert_eq!(specialized, fallback);
        }
    }

    #[crate::multiversion(@"v4:avx512vpopcntdq", "v4", "v3", "v2", "a2")]
    pub fn reduce_sum_of_xor(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut xor = 0;
        for i in 0..n {
            xor += (lhs[i] ^ rhs[i]).count_ones();
        }
        xor
    }
}

#[inline(always)]
pub fn reduce_sum_of_and_or(lhs: &[u64], rhs: &[u64]) -> (u32, u32) {
    reduce_sum_of_and_or::reduce_sum_of_and_or(lhs, rhs)
}

mod reduce_sum_of_and_or {
    // FIXME: add manually-implemented SIMD version for AVX512 and AVX2

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    #[target_feature(enable = "avx512vpopcntdq")]
    fn reduce_sum_of_and_or_v4_avx512vpopcntdq(lhs: &[u64], rhs: &[u64]) -> (u32, u32) {
        assert!(lhs.len() == rhs.len());
        use std::arch::x86_64::*;
        let mut and = _mm512_setzero_si512();
        let mut or = _mm512_setzero_si512();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut n = lhs.len();
        while n >= 8 {
            let x = unsafe { _mm512_loadu_si512(a.cast()) };
            let y = unsafe { _mm512_loadu_si512(b.cast()) };
            a = unsafe { a.add(8) };
            b = unsafe { b.add(8) };
            n -= 8;
            and = _mm512_add_epi64(and, _mm512_popcnt_epi64(_mm512_and_si512(x, y)));
            or = _mm512_add_epi64(or, _mm512_popcnt_epi64(_mm512_or_si512(x, y)));
        }
        if n > 0 {
            let mask = _bzhi_u32(0xff, n as u32) as u8;
            let x = unsafe { _mm512_maskz_loadu_epi64(mask, a.cast()) };
            let y = unsafe { _mm512_maskz_loadu_epi64(mask, b.cast()) };
            and = _mm512_add_epi64(and, _mm512_popcnt_epi64(_mm512_and_si512(x, y)));
            or = _mm512_add_epi64(or, _mm512_popcnt_epi64(_mm512_or_si512(x, y)));
        }
        (
            _mm512_reduce_add_epi64(and) as u32,
            _mm512_reduce_add_epi64(or) as u32,
        )
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_xor_v4_avx512vpopcntdq_test() {
        if !crate::is_cpu_detected!("v4") || !crate::is_feature_detected!("avx512vpopcntdq") {
            println!("test {} ... skipped (v4:avx512vpopcntdq)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let rhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_and_or_v4_avx512vpopcntdq(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert_eq!(specialized, fallback);
        }
    }

    #[crate::multiversion(@"v4:avx512vpopcntdq", "v4", "v3", "v2", "a2")]
    pub fn reduce_sum_of_and_or(lhs: &[u64], rhs: &[u64]) -> (u32, u32) {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut and = 0;
        let mut or = 0;
        for i in 0..n {
            and += (lhs[i] & rhs[i]).count_ones();
            or += (lhs[i] | rhs[i]).count_ones();
        }
        (and, or)
    }
}

#[inline(always)]
pub fn reduce_sum_of_x(this: &[u64]) -> u32 {
    reduce_sum_of_x::reduce_sum_of_x(this)
}

mod reduce_sum_of_x {
    // FIXME: add manually-implemented SIMD version for AVX512 and AVX2

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    #[target_feature(enable = "avx512vpopcntdq")]
    fn reduce_sum_of_x_v4_avx512vpopcntdq(this: &[u64]) -> u32 {
        use std::arch::x86_64::*;
        let mut and = _mm512_setzero_si512();
        let mut a = this.as_ptr();
        let mut n = this.len();
        while n >= 8 {
            let x = unsafe { _mm512_loadu_si512(a.cast()) };
            a = unsafe { a.add(8) };
            n -= 8;
            and = _mm512_add_epi64(and, _mm512_popcnt_epi64(x));
        }
        if n > 0 {
            let mask = _bzhi_u32(0xff, n as u32) as u8;
            let x = unsafe { _mm512_maskz_loadu_epi64(mask, a.cast()) };
            and = _mm512_add_epi64(and, _mm512_popcnt_epi64(x));
        }
        _mm512_reduce_add_epi64(and) as u32
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_x_v4_avx512vpopcntdq_test() {
        if !crate::is_cpu_detected!("v4") || !crate::is_feature_detected!("avx512vpopcntdq") {
            println!("test {} ... skipped (v4:avx512vpopcntdq)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let this = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_x_v4_avx512vpopcntdq(&this) };
            let fallback = fallback(&this);
            assert_eq!(specialized, fallback);
        }
    }

    #[crate::multiversion(@"v4:avx512vpopcntdq", "v4", "v3", "v2", "a2")]
    pub fn reduce_sum_of_x(this: &[u64]) -> u32 {
        let n = this.len();
        let mut and = 0;
        for i in 0..n {
            and += this[i].count_ones();
        }
        and
    }
}

#[inline(always)]
pub fn vector_and(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
    vector_and::vector_and(lhs, rhs)
}

mod vector_and {
    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn vector_and(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<u64>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(lhs[i] & rhs[i]);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

#[inline(always)]
pub fn vector_or(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
    vector_or::vector_or(lhs, rhs)
}

mod vector_or {
    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn vector_or(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<u64>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(lhs[i] | rhs[i]);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

#[inline(always)]
pub fn vector_xor(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
    vector_xor::vector_xor(lhs, rhs)
}

mod vector_xor {
    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn vector_xor(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<u64>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(lhs[i] ^ rhs[i]);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}
