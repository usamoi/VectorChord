mod mock;

use crate::mock::{MakePlainPrefetcher, MockRelation};
use algo::accessor::L2S;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{Bound, PyResult, Python, pyclass, pymethods, pymodule};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::num::NonZero;
use vchordg::operator::Op;
use vchordg::types::{DistanceKind, VamanaIndexOptions, VectorKind, VectorOptions};
use vector::vect::{VectBorrowed, VectOwned};

type O = Op<VectOwned<f32>, L2S>;

#[pymodule]
fn vchord_vchordg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vchordg>()?;
    Ok(())
}

#[pyclass]
pub struct Vchordg {
    index: &'static MockRelation,
}

#[pymethods]
impl Vchordg {
    #[new]
    fn new(dim: u32, m: u32, max_alpha: f32, ef_construction: u32) -> Self {
        let index = MockRelation::new();
        vchordg::build::<_, O>(
            VectorOptions {
                dims: dim,
                v: VectorKind::Vecf32,
                d: DistanceKind::L2S,
            },
            VamanaIndexOptions {
                m,
                max_alpha,
                ef_construction,
                beam_construction: 1,
            },
            &index,
        );
        Vchordg { index }
    }

    fn insert(&self, payload: NonZero<u64>, vector: PyReadonlyArray1<f32>) {
        let vector = VectBorrowed::new(vector.as_slice().unwrap());
        let bump = bumpalo::Bump::new();
        let make_vertex_plain_prefetcher = MakePlainPrefetcher { index: &self.index };
        let make_vector_plain_prefetcher = MakePlainPrefetcher { index: &self.index };
        vchordg::insert::<_, O>(
            &self.index,
            vector,
            payload,
            &bump,
            make_vertex_plain_prefetcher,
            make_vector_plain_prefetcher,
        );
    }

    fn insert_many(
        &self,
        num_threads: usize,
        payload: PyReadonlyArray1<u64>,
        vector: PyReadonlyArray2<f32>,
    ) {
        let left_n = payload.dims()[0];
        let right_n = vector.dims()[0];
        let right_d = vector.dims()[1];
        assert_eq!(left_n, right_n);
        let (n, _) = (left_n, right_d);
        let payload_as_array = payload.as_array();
        let vector_as_array = vector.as_array();
        () = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_scoped(
                |t| t.run(),
                |pool| {
                    pool.install(|| {
                        (0..n).into_par_iter().for_each(|i| {
                            let payload =
                                NonZero::new(payload_as_array.get(i).copied().unwrap()).unwrap();
                            let as_row = vector_as_array.row(i);
                            let vector = VectBorrowed::new(as_row.as_slice().unwrap());
                            let bump = bumpalo::Bump::new();
                            let make_vertex_plain_prefetcher =
                                MakePlainPrefetcher { index: &self.index };
                            let make_vector_plain_prefetcher =
                                MakePlainPrefetcher { index: &self.index };
                            vchordg::insert::<_, O>(
                                &self.index,
                                vector,
                                payload,
                                &bump,
                                make_vertex_plain_prefetcher,
                                make_vector_plain_prefetcher,
                            );
                        });
                    })
                },
            )
            .unwrap();
    }

    fn search<'py>(
        &self,
        py: Python<'py>,
        ef_search: u32,
        vector: PyReadonlyArray1<f32>,
        k: usize,
    ) -> Bound<'py, PyArray1<u64>> {
        let vector_as_array = vector.as_array();
        let vector = VectBorrowed::new(vector_as_array.as_slice().unwrap());
        let bump = bumpalo::Bump::new();
        let make_vertex_plain_prefetcher = MakePlainPrefetcher { index: &self.index };
        let make_vector_plain_prefetcher = MakePlainPrefetcher { index: &self.index };
        let iterator = vchordg::search::<_, O>(
            &self.index,
            vector,
            ef_search,
            1,
            &bump,
            make_vertex_plain_prefetcher,
            make_vector_plain_prefetcher,
        );
        let result = iterator.take(k).map(|(_, x)| x.get()).collect::<Vec<_>>();
        PyArray1::from_vec(py, result)
    }
}
