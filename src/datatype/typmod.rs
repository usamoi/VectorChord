use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Typmod {
    Any,
    Dims(NonZeroU32),
}

impl Typmod {
    pub fn parse_from_i32(x: i32) -> Option<Self> {
        use Typmod::*;
        if x == -1 {
            Some(Any)
        } else if x >= 1 {
            Some(Dims(NonZeroU32::new(x as u32).unwrap()))
        } else {
            None
        }
    }
    #[allow(dead_code)]
    pub fn into_option_string(self) -> Option<String> {
        use Typmod::*;
        match self {
            Any => None,
            Dims(x) => Some(x.get().to_string()),
        }
    }
    #[allow(dead_code)]
    pub fn into_i32(self) -> i32 {
        use Typmod::*;
        match self {
            Any => -1,
            Dims(x) => x.get() as i32,
        }
    }
    pub fn dims(self) -> Option<NonZeroU32> {
        use Typmod::*;
        match self {
            Any => None,
            Dims(dims) => Some(dims),
        }
    }
}
