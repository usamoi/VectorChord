use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub struct Checker<T> {
    size: usize,
    heap: BinaryHeap<(Distance, AlwaysEqual<T>)>,
}

impl<T> Checker<T> {
    pub fn new(size: usize) -> Self {
        assert_ne!(size, 0, "size cannot be zero");
        Checker {
            size,
            heap: BinaryHeap::with_capacity(size + 1),
        }
    }
    pub fn push(&mut self, (Reverse(dis), t): (Reverse<Distance>, AlwaysEqual<T>)) {
        self.heap.push((dis, t));
        if self.heap.len() > self.size {
            self.heap.pop();
        }
    }
    pub fn check(&self, value: Distance) -> bool {
        if self.heap.len() < self.size {
            true
        } else {
            Some(value) < self.heap.peek().map(|(x, _)| *x)
        }
    }
}
