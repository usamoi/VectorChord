use crate::Id;
use std::collections::HashSet;

pub struct Visited {
    inner: HashSet<Id>,
}

impl Visited {
    pub fn new() -> Self {
        Self {
            inner: HashSet::new(),
        }
    }
    pub fn insert(&mut self, x: Id) {
        self.inner.insert(x);
    }
    pub fn contains(& self, x: Id) -> bool {
        self.inner.contains(&x)
    }
}
