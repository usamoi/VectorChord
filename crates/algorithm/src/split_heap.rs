use crate::linked_vec::{LinkedVec, LinkedVecAccessor};
use always_equal::AlwaysEqual;
use std::collections::BinaryHeap;

pub struct Pusher<K, V> {
    keys: LinkedVec<(K, AlwaysEqual<u32>)>,
    values: LinkedVec<V>,
}

impl<K, V> Pusher<K, V> {
    pub fn new() -> Self {
        Self {
            keys: LinkedVec::new(),
            values: LinkedVec::new(),
        }
    }
    pub fn push(&mut self, k: K, v: V) {
        self.keys.push((k, AlwaysEqual(self.values.len())));
        self.values.push(v);
    }
}

impl<K, V> Pusher<K, V>
where
    K: Ord,
{
    pub fn build(self) -> Popper<K, V> {
        Popper {
            keys: BinaryHeap::from(self.keys.into_vec()),
            values: self.values.into_accessor(),
        }
    }
}

pub struct Popper<K, V> {
    keys: BinaryHeap<(K, AlwaysEqual<u32>)>,
    values: LinkedVecAccessor<V>,
}

impl<K, V> Popper<K, V> {
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

impl<K, V> Popper<K, V>
where
    K: Clone,
    V: Clone,
{
    pub fn peek(&self) -> Option<K> {
        self.keys.peek().map(|(x, _)| x.clone())
    }
}

impl<K, V> Popper<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    pub fn pop(&mut self) -> Option<V> {
        let (_, AlwaysEqual(index)) = self.keys.pop()?;
        Some(self.values.get(index).clone())
    }
}
