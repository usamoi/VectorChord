mod reduce_sum_of_xy {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_xy_v4(s: &[u8], t: &[u8]) -> u32 {
        use core::arch::x86_64::*;
        assert_eq!(s.len(), t.len());
        let n = s.len();
        let lo = _mm512_set1_epi16(0x00ff_u16 as i16);
        let hi = _mm512_set1_epi16(0xff00_u16 as i16);
        let mut _0 = _mm512_setzero_si512();
        let mut _1 = _mm512_setzero_si512();
        let mut _2 = _mm512_setzero_si512();
        let mut _3 = _mm512_setzero_si512();
        for i in (0..n).step_by(64) {
            let mask = _bzhi_u64(u64::MAX, n as u32);
            let s = unsafe { _mm512_maskz_loadu_epi8(mask, s.as_ptr().add(i).cast()) };
            let t = unsafe { _mm512_maskz_loadu_epi8(mask, t.as_ptr().add(i).cast()) };
            let s_l = _mm512_and_si512(s, lo);
            let s_h = _mm512_and_si512(_mm512_srli_epi16(_mm512_and_si512(s, hi), 8), lo);
            let t_l = _mm512_and_si512(t, lo);
            let t_h = _mm512_and_si512(_mm512_srli_epi16(_mm512_and_si512(t, hi), 8), lo);
            let l = _mm512_mullo_epi16(s_l, t_l);
            let h = _mm512_mullo_epi16(s_h, t_h);
            _0 = _mm512_add_epi32(_0, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(l, 0)));
            _1 = _mm512_add_epi32(_1, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(l, 1)));
            _2 = _mm512_add_epi32(_2, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(h, 0)));
            _3 = _mm512_add_epi32(_3, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(h, 1)));
        }
        let _5 = _mm512_add_epi32(_0, _1);
        let _6 = _mm512_add_epi32(_2, _3);
        _mm512_reduce_add_epi32(_mm512_add_epi32(_5, _6)) as u32
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_xy_v4_test() {
        use rand::Rng;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            let rhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_v4(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    specialized == fallback,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_xy_v3(s: &[u8], t: &[u8]) -> u32 {
        use crate::emulate::emulate_mm256_reduce_add_epi32;
        use core::arch::x86_64::*;
        assert_eq!(s.len(), t.len());
        let n = s.len();
        let lo = _mm256_set1_epi16(0x00ff_u16 as i16);
        let hi = _mm256_set1_epi16(0xff00_u16 as i16);
        let mut _0 = _mm256_setzero_si256();
        let mut _1 = _mm256_setzero_si256();
        let mut _2 = _mm256_setzero_si256();
        let mut _3 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            let s = unsafe { _mm256_loadu_si256(s.as_ptr().add(i).cast()) };
            let t = unsafe { _mm256_loadu_si256(t.as_ptr().add(i).cast()) };
            let s_l = _mm256_and_si256(s, lo);
            let s_h = _mm256_and_si256(_mm256_srli_epi16(_mm256_and_si256(s, hi), 8), lo);
            let t_l = _mm256_and_si256(t, lo);
            let t_h = _mm256_and_si256(_mm256_srli_epi16(_mm256_and_si256(t, hi), 8), lo);
            let l = _mm256_mullo_epi16(s_l, t_l);
            let h = _mm256_mullo_epi16(s_h, t_h);
            _0 = _mm256_add_epi32(_0, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(l, 0)));
            _1 = _mm256_add_epi32(_1, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(l, 1)));
            _2 = _mm256_add_epi32(_2, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(h, 0)));
            _3 = _mm256_add_epi32(_3, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(h, 1)));
            i += 32;
        }
        let _5 = _mm256_add_epi32(_0, _1);
        let _6 = _mm256_add_epi32(_2, _3);
        let mut result = emulate_mm256_reduce_add_epi32(_mm256_add_epi32(_5, _6)) as u32;
        if i < n {
            while i < n {
                result += s[i] as u32 * t[i] as u32;
                i += 1;
            }
        }
        result
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_xy_v3_test() {
        use rand::Rng;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            let rhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_v3(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    specialized == fallback,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    fn reduce_sum_of_xy_v2(s: &[u8], t: &[u8]) -> u32 {
        use crate::emulate::emulate_mm_reduce_add_epi32;
        use core::arch::x86_64::*;
        assert_eq!(s.len(), t.len());
        let n = s.len();
        let lo = _mm_set1_epi16(0x00ff_u16 as i16);
        let hi = _mm_set1_epi16(0xff00_u16 as i16);
        let mut _0 = _mm_setzero_si128();
        let mut _1 = _mm_setzero_si128();
        let mut _2 = _mm_setzero_si128();
        let mut _3 = _mm_setzero_si128();
        let mut i = 0;
        while i + 16 <= n {
            let s = unsafe { _mm_loadu_si128(s.as_ptr().add(i).cast()) };
            let t = unsafe { _mm_loadu_si128(t.as_ptr().add(i).cast()) };
            let s_l = _mm_and_si128(s, lo);
            let s_h = _mm_and_si128(_mm_srli_epi16(_mm_and_si128(s, hi), 8), lo);
            let t_l = _mm_and_si128(t, lo);
            let t_h = _mm_and_si128(_mm_srli_epi16(_mm_and_si128(t, hi), 8), lo);
            let l = _mm_mullo_epi16(s_l, t_l);
            let h = _mm_mullo_epi16(s_h, t_h);
            _0 = _mm_add_epi32(_0, _mm_cvtepu16_epi32(l));
            _1 = _mm_add_epi32(_1, _mm_cvtepu16_epi32(_mm_unpackhi_epi64(l, l)));
            _2 = _mm_add_epi32(_2, _mm_cvtepu16_epi32(h));
            _3 = _mm_add_epi32(_3, _mm_cvtepu16_epi32(_mm_unpackhi_epi64(h, h)));
            i += 16;
        }
        let _5 = _mm_add_epi32(_0, _1);
        let _6 = _mm_add_epi32(_2, _3);
        let mut result = emulate_mm_reduce_add_epi32(_mm_add_epi32(_5, _6)) as u32;
        if i < n {
            while i < n {
                result += s[i] as u32 * t[i] as u32;
                i += 1;
            }
        }
        result
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_xy_v2_test() {
        use rand::Rng;
        if !crate::is_cpu_detected!("v2") {
            println!("test {} ... skipped (v2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            let rhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_v2(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    specialized == fallback,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_xy_a2(s: &[u8], t: &[u8]) -> u32 {
        use core::arch::aarch64::*;
        assert_eq!(s.len(), t.len());
        let n = s.len();
        let lo = vdupq_n_u16(0x00ff_u16);
        let mut _0 = vdupq_n_u32(0);
        let mut _1 = vdupq_n_u32(0);
        let mut _2 = vdupq_n_u32(0);
        let mut _3 = vdupq_n_u32(0);
        let mut i = 0;
        while i + 16 <= n {
            let s = vreinterpretq_u16_u8(unsafe { vld1q_u8(s.as_ptr().add(i).cast()) });
            let t = vreinterpretq_u16_u8(unsafe { vld1q_u8(t.as_ptr().add(i).cast()) });
            let s_l = vandq_u16(s, lo);
            let s_h = vshrq_n_u16(s, 8);
            let t_l = vandq_u16(t, lo);
            let t_h = vshrq_n_u16(t, 8);
            let l = vmulq_u16(s_l, t_l);
            let h = vmulq_u16(s_h, t_h);
            _0 = vaddq_u32(_0, vmovl_u16(vget_low_u16(l)));
            _1 = vaddq_u32(_1, vmovl_u16(vget_high_u16(l)));
            _2 = vaddq_u32(_2, vmovl_u16(vget_low_u16(h)));
            _3 = vaddq_u32(_3, vmovl_u16(vget_high_u16(h)));
            i += 16;
        }
        let _5 = vaddq_u32(_0, _1);
        let _6 = vaddq_u32(_2, _3);
        let mut result = vaddvq_u32(vaddq_u32(_5, _6));
        if i < n {
            while i < n {
                result += s[i] as u32 * t[i] as u32;
                i += 1;
            }
        }
        result
    }

    #[cfg(all(target_arch = "aarch64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_xy_a2_test() {
        use rand::Rng;
        if !crate::is_cpu_detected!("a2") {
            println!("test {} ... skipped (a2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            let rhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_a2(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    specialized == fallback,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[crate::multiversion(@"v4", @"v3", @"v2", @"a2")]
    pub fn reduce_sum_of_xy(s: &[u8], t: &[u8]) -> u32 {
        assert_eq!(s.len(), t.len());
        let n = s.len();
        let mut result = 0;
        for i in 0..n {
            result += (s[i] as u32) * (t[i] as u32);
        }
        result
    }
}

#[inline(always)]
pub fn reduce_sum_of_xy(s: &[u8], t: &[u8]) -> u32 {
    reduce_sum_of_xy::reduce_sum_of_xy(s, t)
}

mod reduce_sum_of_x_as_u16 {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_x_as_u16_v4(this: &[u8]) -> u16 {
        use crate::emulate::emulate_mm512_reduce_add_epi16;
        use std::arch::x86_64::*;
        let us = _mm512_set1_epi16(255);
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm512_setzero_si512();
        while n >= 32 {
            let x = unsafe { _mm256_loadu_si256(a.cast()) };
            a = unsafe { a.add(32) };
            n -= 32;
            sum = _mm512_add_epi16(_mm512_and_si512(us, _mm512_cvtepi8_epi16(x)), sum);
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffffffff, n as u32);
            let x = unsafe { _mm256_maskz_loadu_epi8(mask, a.cast()) };
            sum = _mm512_add_epi16(_mm512_and_si512(us, _mm512_cvtepi8_epi16(x)), sum);
        }
        emulate_mm512_reduce_add_epi16(sum) as u16
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_x_as_u16_v4_test() {
        use rand::Rng;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n).map(|_| rng.random_range(0..16)).collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x_as_u16_v4(this) };
                let fallback = fallback(this);
                assert_eq!(specialized, fallback);
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_x_as_u16_v3(this: &[u8]) -> u16 {
        use crate::emulate::emulate_mm256_reduce_add_epi16;
        use std::arch::x86_64::*;
        let us = _mm256_set1_epi16(255);
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm256_setzero_si256();
        while n >= 16 {
            let x = unsafe { _mm_loadu_si128(a.cast()) };
            a = unsafe { a.add(16) };
            n -= 16;
            sum = _mm256_add_epi16(_mm256_and_si256(us, _mm256_cvtepi8_epi16(x)), sum);
        }
        let mut sum = emulate_mm256_reduce_add_epi16(sum) as u16;
        // this hint is used to disable loop unrolling
        while std::hint::black_box(n) > 0 {
            let x = unsafe { a.read() };
            a = unsafe { a.add(1) };
            n -= 1;
            sum += x as u16;
        }
        sum
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_x_as_u16_v3_test() {
        use rand::Rng;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n).map(|_| rng.random_range(0..16)).collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x_as_u16_v3(this) };
                let fallback = fallback(this);
                assert_eq!(specialized, fallback);
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    fn reduce_sum_of_x_as_u16_v2(this: &[u8]) -> u16 {
        use crate::emulate::emulate_mm_reduce_add_epi16;
        use std::arch::x86_64::*;
        let us = _mm_set1_epi16(255);
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm_setzero_si128();
        while n >= 8 {
            let x = unsafe { _mm_loadu_si64(a.cast()) };
            a = unsafe { a.add(8) };
            n -= 8;
            sum = _mm_add_epi16(_mm_and_si128(us, _mm_cvtepi8_epi16(x)), sum);
        }
        let mut sum = emulate_mm_reduce_add_epi16(sum) as u16;
        // this hint is used to disable loop unrolling
        while std::hint::black_box(n) > 0 {
            let x = unsafe { a.read() };
            a = unsafe { a.add(1) };
            n -= 1;
            sum += x as u16;
        }
        sum
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_x_as_u16_v2_test() {
        use rand::Rng;
        if !crate::is_cpu_detected!("v2") {
            println!("test {} ... skipped (v2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n).map(|_| rng.random_range(0..16)).collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x_as_u16_v2(this) };
                let fallback = fallback(this);
                assert_eq!(specialized, fallback);
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_x_as_u16_a2(this: &[u8]) -> u16 {
        use std::arch::aarch64::*;
        let us = vdupq_n_u16(255);
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = vdupq_n_u16(0);
        while n >= 8 {
            let x = unsafe { vld1_u8(a) };
            a = unsafe { a.add(8) };
            n -= 8;
            sum = vaddq_u16(vandq_u16(us, vmovl_u8(x)), sum);
        }
        let mut sum = vaddvq_u16(sum);
        // this hint is used to disable loop unrolling
        while std::hint::black_box(n) > 0 {
            let x = unsafe { a.read() };
            a = unsafe { a.add(1) };
            n -= 1;
            sum += x as u16;
        }
        sum
    }

    #[cfg(all(target_arch = "aarch64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_x_as_u16_a2_test() {
        use rand::Rng;
        if !crate::is_cpu_detected!("a2") {
            println!("test {} ... skipped (a2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n).map(|_| rng.random_range(0..16)).collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x_as_u16_a2(this) };
                let fallback = fallback(this);
                assert_eq!(specialized, fallback);
            }
        }
    }

    #[crate::multiversion(@"v4", @"v3", @"v2", @"a2")]
    pub fn reduce_sum_of_x_as_u16(this: &[u8]) -> u16 {
        let n = this.len();
        let mut sum = 0;
        for i in 0..n {
            sum += this[i] as u16;
        }
        sum
    }
}

#[inline(always)]
pub fn reduce_sum_of_x_as_u16(vector: &[u8]) -> u16 {
    reduce_sum_of_x_as_u16::reduce_sum_of_x_as_u16(vector)
}

mod reduce_sum_of_x {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_x_v4(this: &[u8]) -> u32 {
        use std::arch::x86_64::*;
        let us = _mm512_set1_epi32(255);
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm512_setzero_si512();
        while n >= 16 {
            let x = unsafe { _mm_loadu_epi8(a.cast()) };
            a = unsafe { a.add(16) };
            n -= 16;
            sum = _mm512_add_epi32(_mm512_and_si512(us, _mm512_cvtepi8_epi32(x)), sum);
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm_maskz_loadu_epi8(mask, a.cast()) };
            sum = _mm512_add_epi32(_mm512_and_si512(us, _mm512_cvtepi8_epi32(x)), sum);
        }
        _mm512_reduce_add_epi32(sum) as u32
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_x_v4_test() {
        use rand::Rng;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n).map(|_| rng.random_range(0..16)).collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x_v4(this) };
                let fallback = fallback(this);
                assert_eq!(specialized, fallback);
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_x_v3(this: &[u8]) -> u32 {
        use crate::emulate::emulate_mm256_reduce_add_epi32;
        use std::arch::x86_64::*;
        let us = _mm256_set1_epi32(255);
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm256_setzero_si256();
        while n >= 8 {
            let x = unsafe { _mm_loadl_epi64(a.cast()) };
            a = unsafe { a.add(8) };
            n -= 8;
            sum = _mm256_add_epi32(_mm256_and_si256(us, _mm256_cvtepi8_epi32(x)), sum);
        }
        let mut sum = emulate_mm256_reduce_add_epi32(sum) as u32;
        // this hint is used to disable loop unrolling
        while std::hint::black_box(n) > 0 {
            let x = unsafe { a.read() };
            a = unsafe { a.add(1) };
            n -= 1;
            sum += x as u32;
        }
        sum
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_x_v3_test() {
        use rand::Rng;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n).map(|_| rng.random_range(0..16)).collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x_v3(this) };
                let fallback = fallback(this);
                assert_eq!(specialized, fallback);
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    fn reduce_sum_of_x_v2(this: &[u8]) -> u32 {
        use crate::emulate::emulate_mm_reduce_add_epi32;
        use std::arch::x86_64::*;
        let us = _mm_set1_epi32(255);
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm_setzero_si128();
        while n >= 4 {
            let x = unsafe { _mm_cvtsi32_si128(a.cast::<i32>().read_unaligned()) };
            a = unsafe { a.add(4) };
            n -= 4;
            sum = _mm_add_epi32(_mm_and_si128(us, _mm_cvtepi8_epi32(x)), sum);
        }
        let mut sum = emulate_mm_reduce_add_epi32(sum) as u32;
        // this hint is used to disable loop unrolling
        while std::hint::black_box(n) > 0 {
            let x = unsafe { a.read() };
            a = unsafe { a.add(1) };
            n -= 1;
            sum += x as u32;
        }
        sum
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_x_v2_test() {
        use rand::Rng;
        if !crate::is_cpu_detected!("v2") {
            println!("test {} ... skipped (v2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n).map(|_| rng.random_range(0..16)).collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x_v2(this) };
                let fallback = fallback(this);
                assert_eq!(specialized, fallback);
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_x_a2(this: &[u8]) -> u32 {
        use std::arch::aarch64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum_0 = vdupq_n_u32(0);
        let mut sum_1 = vdupq_n_u32(0);
        while n >= 8 {
            let x = unsafe { vmovl_u8(vld1_u8(a.cast())) };
            a = unsafe { a.add(8) };
            n -= 8;
            sum_0 = vaddq_u32(vmovl_u16(vget_low_u16(x)), sum_0);
            sum_1 = vaddq_u32(vmovl_u16(vget_high_u16(x)), sum_1);
        }
        let mut sum = vaddvq_u32(vaddq_u32(sum_0, sum_1));
        // this hint is used to disable loop unrolling
        while std::hint::black_box(n) > 0 {
            let x = unsafe { a.read() };
            a = unsafe { a.add(1) };
            n -= 1;
            sum += x as u32;
        }
        sum
    }

    #[cfg(all(target_arch = "aarch64", test))]
    #[test]
    fn reduce_sum_of_x_a2_test() {
        use rand::Rng;
        if !crate::is_cpu_detected!("a2") {
            println!("test {} ... skipped (a2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n).map(|_| rng.random_range(0..16)).collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x_a2(this) };
                let fallback = fallback(this);
                assert_eq!(specialized, fallback);
            }
        }
    }

    #[crate::multiversion(@"v4", @"v3", @"v2", @"a2")]
    pub fn reduce_sum_of_x(this: &[u8]) -> u32 {
        let n = this.len();
        let mut sum = 0;
        for i in 0..n {
            sum += this[i] as u32;
        }
        sum
    }
}

#[inline(always)]
pub fn reduce_sum_of_x(vector: &[u8]) -> u32 {
    reduce_sum_of_x::reduce_sum_of_x(vector)
}
