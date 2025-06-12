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

use simd::Floating;

#[derive(Debug, Clone, Copy)]
pub struct CodeMetadata {
    pub dis_u_2: f32,
    pub norm_of_lattice: f32,
    pub sum_of_code: f32,
}

pub type Code = (CodeMetadata, Vec<u8>);

pub fn code(vector: &[f32]) -> Code {
    const B: usize = 8;
    let n = vector.len();
    let dis_u_2 = f32::reduce_sum_of_x2(vector);
    let code = {
        const MIN: i32 = -(1 << (B - 1));
        const MAX: i32 = (1 << (B - 1)) - 1;
        let normalized_vector = {
            let mut vector = vector.to_vec();
            f32::vector_mul_scalar_inplace(&mut vector, 1.0 / dis_u_2.sqrt());
            vector
        };
        let scale = {
            let mut o = normalized_vector.clone();
            f32::vector_abs_inplace(&mut o);
            find_scale(&o) as f32 + f32::EPSILON
        };
        let mut code = Vec::with_capacity(n as _);
        for i in 0..n {
            let v = scale * normalized_vector[i as usize];
            let c = v.floor().clamp(MIN as f32, MAX as f32) as i32;
            code.push((c + (1 << (B - 1))) as _);
        }
        code
    };
    let norm_of_lattice = {
        const BASE: f32 = -0.5 * ((1 << B) - 1) as f32;
        let mut y = 0.0;
        for i in 0..n {
            let x = BASE + code[i as usize] as f32;
            y += x * x;
        }
        y.sqrt()
    };
    let sum_of_code = {
        let mut y = 0;
        for i in 0..n {
            let x = code[i as usize] as u32;
            y += x;
        }
        y as f32
    };
    (
        CodeMetadata {
            dis_u_2,
            norm_of_lattice,
            sum_of_code,
        },
        code,
    )
}

pub fn process_l2(lhs: (CodeMetadata, &[u8]), rhs: (CodeMetadata, &[u8])) -> f32 {
    const B: usize = 8;
    const C: f32 = ((1 << B) - 1) as f32 * 0.5;

    assert!(lhs.1.len() == rhs.1.len());
    let n = lhs.1.len();

    let value = simd::u8::reduce_sum_of_x_as_u32_y_as_u32(lhs.1, rhs.1);
    let ip = value as f32 - C * (lhs.0.sum_of_code + rhs.0.sum_of_code) + n as f32 * C * C;
    lhs.0.dis_u_2 + rhs.0.dis_u_2
        - 2.0
            * ip
            * (lhs.0.dis_u_2.sqrt() / lhs.0.norm_of_lattice)
            * (rhs.0.dis_u_2.sqrt() / rhs.0.norm_of_lattice)
}

pub fn half_process_l2(n: u32, value: u32, lhs: CodeMetadata, rhs: CodeMetadata) -> f32 {
    const B: usize = 8;
    const C: f32 = ((1 << B) - 1) as f32 * 0.5;

    let ip = value as f32 - C * (lhs.sum_of_code + rhs.sum_of_code) + n as f32 * C * C;
    lhs.dis_u_2 + rhs.dis_u_2
        - 2.0
            * ip
            * (lhs.dis_u_2.sqrt() / lhs.norm_of_lattice)
            * (rhs.dis_u_2.sqrt() / rhs.norm_of_lattice)
}

pub fn process_dot(lhs: (CodeMetadata, &[u8]), rhs: (CodeMetadata, &[u8])) -> f32 {
    const B: usize = 8;
    const C: f32 = ((1 << B) - 1) as f32 * 0.5;

    assert!(lhs.1.len() == rhs.1.len());
    let n = lhs.1.len();

    let value = simd::u8::reduce_sum_of_x_as_u32_y_as_u32(lhs.1, rhs.1);
    let ip = value as f32 - C * (lhs.0.sum_of_code + rhs.0.sum_of_code) + n as f32 * C * C;
    -ip * (lhs.0.dis_u_2.sqrt() / lhs.0.norm_of_lattice)
        * (rhs.0.dis_u_2.sqrt() / rhs.0.norm_of_lattice)
}

pub fn half_process_dot(n: u32, value: u32, lhs: CodeMetadata, rhs: CodeMetadata) -> f32 {
    const B: usize = 8;
    const C: f32 = ((1 << B) - 1) as f32 * 0.5;

    let ip = value as f32 - C * (lhs.sum_of_code + rhs.sum_of_code) + n as f32 * C * C;
    -ip * (lhs.dis_u_2.sqrt() / lhs.norm_of_lattice) * (rhs.dis_u_2.sqrt() / rhs.norm_of_lattice)
}

fn find_scale(o: &[f32]) -> f64 {
    const B: usize = 8;

    assert!((1..=8).contains(&B));

    let mask = (1_u32 << (B - 1)) - 1;
    let dims = o.len();

    let mut code = Vec::<u8>::with_capacity(dims);
    let mut numerator = 0.0f64;
    let mut sqr_denominator = 0.0f64;

    let (mut y_m, mut x_m);

    for i in 0..dims {
        code.push(0);
        numerator += 0.5 * o[i] as f64;
        sqr_denominator += 0.5 * 0.5;
    }
    {
        let x = 0.0;
        let y = numerator / sqr_denominator.sqrt();
        (y_m, x_m) = (y, x);
    }

    let mut events = Vec::<(f64, usize)>::new();
    for i in 0..dims {
        for c in 1..=mask {
            let x = (c as f64) / o[i] as f64;
            events.push((x, i));
        }
    }
    events.sort_unstable_by(|(x, _), (y, _)| f64::total_cmp(x, y));
    for (x, i) in events.into_iter() {
        code[i] += 1;
        numerator += o[i] as f64;
        sqr_denominator += 2.0 * code[i] as f64;

        let y = numerator / sqr_denominator.sqrt();
        if y > y_m {
            (y_m, x_m) = (y, x);
        }
    }

    x_m
}
