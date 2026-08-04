from __future__ import annotations

from open_wam.models.action_decoders.video_conditioned_expert import (
    ActionExpertPreprocessOutput as DualExpertActionPreprocessOutput,
)
from open_wam.models.action_decoders.video_conditioned_expert import (
    ConditionedActionTransformerBlock as DualExpertActionTransformerBlock,
)
from open_wam.models.action_decoders.video_conditioned_expert import (
    VideoConditionedActionExpert as DualExpertActionExpert,
)
from open_wam.models.action_decoders.video_conditioned_expert import (
    init_conditioned_action_expert_from_video_core,
)

_COMPATIBILITY_EXPORTS = (
    DualExpertActionPreprocessOutput,
    DualExpertActionTransformerBlock,
)


def init_action_expert_from_video_core(
    *,
    action_expert: DualExpertActionExpert,
    video_core,
    mode: str = "video_weight_copy",
) -> None:
    """Compatibility wrapper for the shared action-expert warm-start helper."""

    init_conditioned_action_expert_from_video_core(
        action_expert=action_expert,
        video_core=video_core,
        mode=mode,
    )
