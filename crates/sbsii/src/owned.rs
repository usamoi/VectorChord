use crate::stack::StackIntoIter;
use std::ptr::NonNull;

pub struct HeapIntoIter<T: Copy> {
    pointer: NonNull<T>,
    off: u16,
    len: u16,
}

impl<T: Copy> HeapIntoIter<T> {
    #[inline(always)]
    pub(crate) fn from_slice(slice: &[T]) -> Self {
        assert!(slice.len() <= 65535_usize);
        #[allow(unsafe_code)]
        unsafe {
            let c = NonNull::new_unchecked(Box::<[T]>::into_raw(Box::from(slice)).cast::<T>());
            Self {
                pointer: c,
                off: 0,
                len: slice.len() as u16,
            }
        }
    }
}

impl<T: Copy> Iterator for HeapIntoIter<T> {
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        #[allow(unsafe_code)]
        unsafe {
            if self.off < self.len {
                let r = self.pointer.as_ptr().add(self.off as _).read();
                self.off += 1;
                Some(r)
            } else {
                None
            }
        }
    }
}

impl<T: Copy> Clone for HeapIntoIter<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        #[allow(unsafe_code)]
        unsafe {
            let p = std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.len as _);
            let c = NonNull::new_unchecked(Box::<[T]>::into_raw(Box::from(p)).cast::<T>());
            Self {
                pointer: c,
                off: self.off,
                len: self.len,
            }
        }
    }
}

impl<T: Copy> Drop for HeapIntoIter<T> {
    #[inline(always)]
    fn drop(&mut self) {
        #[allow(unsafe_code)]
        unsafe {
            let p = std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.len as _);
            let _ = Box::<[T]>::from_raw(p);
        }
    }
}

#[derive(Clone)]
pub enum IntoIter<T: Copy, const N: usize> {
    Stack(StackIntoIter<T, N>),
    Heap(HeapIntoIter<T>),
}

impl<T: Copy, const N: usize> IntoIter<T, N> {
    #[inline(always)]
    pub fn from_slice(slice: &[T]) -> Self {
        assert!(slice.len() <= 65535);
        if slice.len() <= N && N <= 65535 {
            IntoIter::Stack(StackIntoIter::from_slice(slice))
        } else {
            IntoIter::Heap(HeapIntoIter::from_slice(slice))
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

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(size_of::<IntoIter::<u32, 4>>() == 24);
};

#[test]
fn tests() {
    for x in IntoIter::<u32, 5>::from_slice(&[1; 0]).clone() {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 1]).clone() {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 2]).clone() {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 3]).clone() {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 4]).clone() {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 5]).clone() {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 6]).clone() {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 7]).clone() {
        assert_eq!(x, 1);
    }
}
