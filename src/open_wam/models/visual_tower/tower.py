from __future__ import annotations

from torch import nn

from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig

from .contracts import VisualCoreInput, VisualStageOutputs
from .core import LingbotVisualCore
from .decoder import VisualFeatureDecoder
from .frontend import LingbotVisualFrontend
from .grid_ids import build_video_grid_ids
from .replica_core import LingbotReplicaVisualCore


class VisualTower(nn.Module):
    """Stage-aware visual tower used by all policy variants."""

    def __init__(self, config: LingbotCompatibleVideoBackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config or LingbotCompatibleVideoBackboneConfig()
        self.frontend = LingbotVisualFrontend(self.config)
        if self.config.implementation == "lingbot_replica":
            self.core = LingbotReplicaVisualCore(self.config)
        elif self.config.implementation == "dummy":
            self.core = LingbotVisualCore(self.config)
        else:
            raise ValueError(
                f"Unsupported backbone implementation '{self.config.implementation}'. "
                "Expected 'dummy' or 'lingbot_replica'."
            )
        self.decoder = VisualFeatureDecoder(self.config.hidden_size)

    def run_frontend(self, canonical_video):
        return self.frontend(canonical_video)

    def run_core(self, core_input: VisualCoreInput):
        return self.core(core_input)

    def run_default_core(self, frontend_output):
        batch_size, seq_len, _ = frontend_output.video_tokens.shape
        return self.run_core(
            VisualCoreInput(
                tokens=frontend_output.video_tokens,
                token_layout=frontend_output.token_grid,
                grid_ids=build_video_grid_ids(
                    frontend_output.token_grid,
                    device=frontend_output.video_tokens.device,
                ),
                timestep_values=frontend_output.video_tokens.new_zeros((batch_size, seq_len), dtype=frontend_output.video_tokens.dtype),
                text_context=frontend_output.conditioning.text_context,
                conditioning=frontend_output.conditioning,
            )
        )

    def run_decode(self, frontend_output, core_output):
        return self.decoder(frontend_output=frontend_output, core_output=core_output)

    def forward_default(self, canonical_video, include_decode: bool = False) -> VisualStageOutputs:
        frontend_output = self.run_frontend(canonical_video)
        core_output = self.run_default_core(frontend_output)
        decode_output = self.run_decode(frontend_output, core_output) if include_decode else None
        return VisualStageOutputs(frontend=frontend_output, core=core_output, decode=decode_output)
