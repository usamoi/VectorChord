use serde::{Deserialize, Serialize};
use std::num::NonZero;
use vchordg::types::VchordgIndexOptions;
use vchordg::types::VectorOptions;

#[derive(Debug, Serialize, Deserialize)]
pub enum ClientPacket {
    Build {
        vector_options: VectorOptions,
        index_options: VchordgIndexOptions,
    },
    Insert {
        vector: vchordg::types::OwnedVector,
        payload: NonZero<u64>,
    },
    Finish {},
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ServerPacket {
    Error {
        reason: String,
    },
    Flush {
        id: u32,
        pd_lower: u16,
        pd_upper: u16,
        pd_special: u16,
        content: Vec<u8>,
    },
    Finish {},
}
