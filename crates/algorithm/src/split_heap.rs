use always_equal::AlwaysEqual;
use std::collections::BinaryHeap;

pub struct Pusher<K, V> {
    keys: Vec<(K, AlwaysEqual<u32>)>,
    values: Vec<V>,
}

impl<K, V> Pusher<K, V> {
    pub fn new() -> Self {
        Self {
            keys: Vec::with_capacity(200_000),
            values: Vec::with_capacity(200_000),
        }
    }
    pub fn push(&mut self, k: K, v: V) {
        self.keys.push((k, AlwaysEqual(self.values.len() as u32)));
        self.values.push(v);
    }
}

impl<K, V> Pusher<K, V>
where
    K: Ord,
{
    pub fn build(self) -> Popper<K, V> {
        Popper {
            keys: BinaryHeap::from(self.keys),
            values: self.values,
        }
    }
}

pub struct Popper<K, V> {
    keys: BinaryHeap<(K, AlwaysEqual<u32>)>,
    values: Vec<V>,
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
        Some(self.values[index as usize].clone())
    }
}
