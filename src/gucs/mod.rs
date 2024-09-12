pub mod executing;

pub unsafe fn init() {
    unsafe {
        executing::init();
    }
}
