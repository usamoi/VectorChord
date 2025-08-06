use std::mem::MaybeUninit;

#[derive(Clone)]
pub struct StackIntoIter<T: Copy, const N: usize> {
    start: u16,
    end: u16,
    buffer: [MaybeUninit<T>; N],
}

impl<T: Copy, const N: usize> StackIntoIter<T, N> {
    #[inline(always)]
    pub(crate) fn from_slice(slice: &[T]) -> Self {
        assert!(slice.len() <= N && N <= 65535);
        Self {
            start: 0,
            end: slice.len() as u16,
            buffer: {
                let mut buffer = [const { MaybeUninit::uninit() }; N];
                for i in 0..slice.len() {
                    buffer[i] = MaybeUninit::new(slice[i]);
                }
                buffer
            },
        }
    }
}

impl<T: Copy, const N: usize> Iterator for StackIntoIter<T, N> {
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            return None;
        }
        #[allow(unsafe_code)]
        let result = unsafe { self.buffer[self.start as usize].assume_init() };
        self.start += 1;
        Some(result)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = (self.end - self.start) as usize;
        (size, Some(size))
    }
}
