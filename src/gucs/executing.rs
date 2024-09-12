use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};

static NPROBE: GucSetting<i32> = GucSetting::<i32>::new(10);

pub unsafe fn init() {
    GucRegistry::define_int_guc(
        "rabbithole.nprobe",
        "`nprobe` argument of rabbithole.",
        "`nprobe` argument of rabbithole.",
        &NPROBE,
        1,
        u16::MAX as _,
        GucContext::Userset,
        GucFlags::default(),
    );
}

pub fn nprobe() -> u32 {
    NPROBE.get() as u32
}
