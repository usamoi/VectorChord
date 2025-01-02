use criterion::{Criterion, criterion_group, criterion_main};
use rand::Rng;
use simd::Floating;
use std::array::from_fn;
use std::hint::black_box;

const D: usize = 768;
const DATA: usize = 32 * 10000;
const QUERIES: usize = 1;

fn criterion_benchmark(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    {
        println!(
            "x86_64:is_cpu_detected(v3) = {}",
            simd::is_cpu_detected!("v3")
        );
        println!(
            "x86_64:is_cpu_detected(v4) = {}",
            simd::is_cpu_detected!("v4")
        );
        println!(
            "x86_64:is_feature_detected(avx512vpopcntdq) = {}",
            simd::is_feature_detected!("avx512vpopcntdq")
        );
    }

    let mut rng = rand::thread_rng();
    let mut make = || Box::new(from_fn::<_, D, _>(|_| rng.gen_range(-1.0..=1.0_f32)));
    let data = (0..DATA).map(|_| make()).collect::<Vec<_>>();
    let binary_code = (0..DATA)
        .map(|i| {
            let code = rabitq::binary::code(D as _, data[i].as_ref());
            (
                code.dis_u_2,
                code.factor_ppc,
                code.factor_ip,
                code.factor_err,
                code.t(),
            )
        })
        .collect::<Vec<_>>();
    let block_code = (0..DATA / 32)
        .map(|i| {
            let code = rabitq::block::pack_codes(
                D as _,
                from_fn(|j| rabitq::block::code(D as _, data[i * 32 + j].as_ref())),
            );
            (
                code.dis_u_2,
                code.factor_ppc,
                code.factor_ip,
                code.factor_err,
                code.t,
            )
        })
        .collect::<Vec<_>>();
    let queries = (0..QUERIES).map(|_| make()).collect::<Vec<_>>();
    let binary_lut = (0..QUERIES)
        .map(|i| rabitq::binary::preprocess(queries[i].as_ref()))
        .collect::<Vec<_>>();
    let block_lut = (0..QUERIES)
        .map(|i| rabitq::block::fscan_preprocess(queries[i].as_ref()))
        .collect::<Vec<_>>();

    c.bench_function("dot", |b| {
        b.iter_batched(
            || (data.clone(), queries.clone()),
            |(data, queries)| {
                for query in queries.iter() {
                    for data in data.iter() {
                        black_box(f32::reduce_sum_of_xy(query.as_slice(), data.as_slice()));
                    }
                }
                (data, queries)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    c.bench_function("binary", |b| {
        b.iter_batched(
            || (binary_code.clone(), binary_lut.clone()),
            |(binary_code, binary_lut)| {
                for query in binary_lut.iter() {
                    for data in binary_code.iter() {
                        black_box(rabitq::binary::process_lowerbound_dot(
                            black_box(D as _),
                            black_box(query),
                            black_box((data.0, data.1, data.2, data.3, &data.4)),
                            black_box(1.9),
                        ));
                    }
                }
                (binary_code, binary_lut)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    c.bench_function("block", |b| {
        b.iter_batched(
            || (block_code.clone(), block_lut.clone()),
            |(block_code, block_lut)| {
                for query in block_lut.iter() {
                    for data in block_code.iter() {
                        black_box(rabitq::block::fscan_process_lowerbound_dot(
                            black_box(D as _),
                            black_box(query),
                            black_box((&data.0, &data.1, &data.2, &data.3, &data.4)),
                            black_box(1.9),
                        ));
                    }
                }
                (block_code, block_lut)
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
