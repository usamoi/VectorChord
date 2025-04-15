use std::collections::{BinaryHeap, VecDeque};

const WIDTH: usize = 32;

pub struct WindowHeap<T, F> {
    window: VecDeque<T>,
    heap: BinaryHeap<T>,
    f: F,
}

impl<T: Ord, F> WindowHeap<T, F> {
    pub fn new(vec: Vec<T>, f: F) -> Self {
        Self {
            window: VecDeque::with_capacity(WIDTH + 1),
            heap: BinaryHeap::from(vec),
            f,
        }
    }
}

impl<T, F> WindowHeap<T, F> {
    pub fn into_iter(self) -> impl Iterator<Item = T> {
        self.window.into_iter().chain(self.heap.into_iter())
    }
}

impl<T: Ord, F: FnMut(&mut T)> WindowHeap<T, F> {
    pub fn pop_if(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T> {
        while self.window.len() < WIDTH
            && let Some(mut e) = self.heap.pop()
        {
            (self.f)(&mut e);
            self.window.push_back(e);
        }
        self.window.pop_front_if(predicate)
    }
}
