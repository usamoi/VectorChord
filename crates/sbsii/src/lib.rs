// sbsii, small boxed slice into iter

use std::mem::MaybeUninit;

#[derive(Clone)]
pub struct StackIntoIter<T: Copy, const N: usize> {
    start: u16,
    end: u16,
    buffer: [MaybeUninit<T>; N],
}

impl<T: Copy, const N: usize> StackIntoIter<T, N> {
    #[inline(always)]
    fn from_slice(slice: &[T]) -> Self {
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

#[derive(Clone)]
pub enum IntoIter<T: Copy, const N: usize> {
    Stack(StackIntoIter<T, N>),
    Heap(std::vec::IntoIter<T>),
}

impl<T: Copy, const N: usize> IntoIter<T, N> {
    #[inline(always)]
    pub fn from_slice(slice: &[T]) -> Self {
        if slice.len() <= N && N <= 65535 {
            IntoIter::Stack(StackIntoIter::from_slice(slice))
        } else {
            IntoIter::Heap(Box::<[T]>::from(slice).into_iter())
        }
    }
}

impl<T: Copy, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IntoIter::Stack(iter) => iter.next(),
            IntoIter::Heap(iter) => iter.next(),
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            IntoIter::Stack(iter) => iter.size_hint(),
            IntoIter::Heap(iter) => iter.size_hint(),
        }
    }
}

impl<T: Copy, const N: usize> ExactSizeIterator for IntoIter<T, N> {}

#[test]
fn tests() {
    for x in IntoIter::<u32, 5>::from_slice(&[1; 0]) {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 1]) {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 2]) {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 3]) {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 4]) {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 5]) {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 6]) {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 7]) {
        assert_eq!(x, 1);
    }
}
