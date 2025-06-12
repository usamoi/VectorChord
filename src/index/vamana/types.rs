// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025 TensorChord Inc.

use serde::{Deserialize, Serialize};
use validator::Validate;
use vamana::types::VamanaIndexOptions;

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct VamanaInternalBuildOptions {
    #[serde(default = "VamanaInternalBuildOptions::default_ef_construction")]
    #[validate(range(min = 1, max = 65535))]
    pub ef_construction: u32,
}

impl VamanaInternalBuildOptions {
    fn default_ef_construction() -> u32 {
        256
    }
}

impl Default for VamanaInternalBuildOptions {
    fn default() -> Self {
        Self {
            ef_construction: Self::default_ef_construction(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct VamanaIndexingOptions {
    #[serde(flatten)]
    pub index: VamanaIndexOptions,
}
