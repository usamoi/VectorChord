use std::marker::PhantomData;
use vchordrq::cuda::ffi;

pub struct Server {
    p: *mut ffi::server_t,
}

unsafe impl Send for Server {}
unsafe impl Sync for Server {}

impl Server {
    pub fn new(op: ffi::op_t::Type, d: usize, n: usize, centroids: &[u8]) -> Option<Self> {
        if centroids.len() != n * d * ffi::size(op) {
            return None;
        }
        let p = unsafe {
            ffi::vchordrq_assign_server_alloc(op, d, n, centroids.as_ptr().cast_mut().cast())
        };
        if p.is_null() {
            return None;
        }
        Some(Self { p })
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        unsafe {
            ffi::vchordrq_assign_server_free(self.p);
        }
    }
}

pub struct Client<'a> {
    p: *mut ffi::client_t,
    results: Vec<u32>,
    _phantom: PhantomData<&'a ()>,
}

unsafe impl Send for Client<'_> {}
unsafe impl Sync for Client<'_> {}

impl<'a> Client<'a> {
    pub fn new(server: &'a Server, m: usize) -> Option<Self> {
        let p = unsafe {
            ffi::vchordrq_assign_client_alloc(
                (*server.p).op,
                (*server.p).d,
                (*server.p).n,
                (*server.p).centroids,
                m,
            )
        };
        if p.is_null() {
            return None;
        }
        let results = vec![u32::MAX; m];
        Some(Self {
            p,
            results,
            _phantom: PhantomData,
        })
    }
    pub fn query(&mut self, k: usize, vectors: &[u8]) -> Option<&mut [u32]> {
        unsafe {
            if k > (*self.p).m {
                return None;
            }
            if vectors.len() != k * (*self.p).d * ffi::size((*self.p).op) {
                return None;
            }
            let r = ffi::vchordrq_assign_client_query(
                self.p,
                k,
                vectors.as_ptr().cast_mut().cast(),
                self.results.as_mut_ptr(),
            );
            if r != 0 {
                return None;
            }
            Some(&mut self.results[..k])
        }
    }
}

impl<'a> Drop for Client<'a> {
    fn drop(&mut self) {
        unsafe {
            ffi::vchordrq_assign_client_free(self.p);
        }
    }
}
