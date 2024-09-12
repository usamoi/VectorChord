use std::cell::Cell;

pub struct PgCell<T>(Cell<T>);

unsafe impl<T: Send> Send for PgCell<T> {}
unsafe impl<T: Sync> Sync for PgCell<T> {}

impl<T> PgCell<T> {
    pub const unsafe fn new(x: T) -> Self {
        Self(Cell::new(x))
    }
}

impl<T: Copy> PgCell<T> {
    pub fn get(&self) -> T {
        self.0.get()
    }
    pub fn set(&self, value: T) {
        self.0.set(value);
    }
}
