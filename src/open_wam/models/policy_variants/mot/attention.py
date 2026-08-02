"""Compatibility imports for historical MoT attention layouts.

Maintained code imports the unpacked, packed, or cached role owner directly.
"""

from __future__ import annotations

import torch as torch

from open_wam.configs import (
    CurrentBlockCoupling as CurrentBlockCoupling,
    MoTConditionMode as MoTConditionMode,
)
from open_wam.models.common.attention_contracts import (
    PreparedAttentionProfile as PreparedAttentionProfile,
)
from open_wam.models.common.coupling_profiles import (
    build_exact_packed_video_action_coupling_profile as build_exact_packed_video_action_coupling_profile,
)

from .attention_cached import (
    build_mot_inference_action_attention_mask as build_mot_inference_action_attention_mask,
)
from .attention_packed import (
    build_mot_packed_coupling_attention_mask as build_mot_packed_coupling_attention_mask,
    build_mot_packed_coupling_attention_profile as build_mot_packed_coupling_attention_profile,
    build_packed_action_attention_mask as build_packed_action_attention_mask,
)
from .attention_unpacked import (
    build_chunk_causal_video_mask as build_chunk_causal_video_mask,
    build_mot_attention_mask as build_mot_attention_mask,
)

# Preserve the established direct and wildcard surface without adding a global.
(
    CurrentBlockCoupling,
    MoTConditionMode,
    PreparedAttentionProfile,
    build_chunk_causal_video_mask,
    build_exact_packed_video_action_coupling_profile,
    build_mot_attention_mask,
    build_mot_inference_action_attention_mask,
    build_mot_packed_coupling_attention_mask,
    build_mot_packed_coupling_attention_profile,
    build_packed_action_attention_mask,
    torch,
)
