from __future__ import annotations

from torch import nn

from open_wam.data.raw_video import ViewPlacement
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig

from .contracts import VisualCoreInput, VisualStageOutputs
from .core import LingbotVisualCore
from .decoder import VisualFeatureDecoder
from .frontend import LingbotVisualFrontend
from .grid_ids import build_video_grid_ids
from .reference_core_weights import ReferenceCoreLoadReport, load_reference_weights_into_replica_core
from .replica_core import LingbotReplicaVisualCore
from .reference_transformer import build_reference_transformer, preferred_reference_dtype


class VisualTower(nn.Module):
    """Stage-aware visual tower used by all policy variants."""

    def __init__(
        self,
        config: LingbotCompatibleVideoBackboneConfig | None = None,
        *,
        action_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or LingbotCompatibleVideoBackboneConfig()
        self.action_dim = action_dim
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
        self._lingbot_reference_transformers = nn.ModuleDict()
        self.reference_core_load_report: ReferenceCoreLoadReport | None = None
        if self.config.load_reference_core_weights:
            if self.config.implementation != "lingbot_replica":
                raise ValueError("`backbone.load_reference_core_weights` requires `backbone.implementation = lingbot_replica`.")
            if self.action_dim is None:
                raise ValueError("VisualTower requires `action_dim` to load LingBot reference weights into the shared core.")
            self.reference_core_load_report = load_reference_weights_into_replica_core(
                self.core,
                backbone_config=self.config,
                action_dim=self.action_dim,
            )

    def run_frontend(
        self,
        canonical_video,
        *,
        placements: tuple[ViewPlacement, ...] | None = None,
        task_text: tuple[str | None, ...] | None = None,
        text_context=None,
        preserve_stream_cache: bool = False,
    ):
        return self.frontend(
            canonical_video,
            placements=placements,
            task_text=task_text,
            text_context=text_context,
            preserve_stream_cache=preserve_stream_cache,
        )

    def run_frontend_from_latents(
        self,
        video_latents,
        *,
        task_text: tuple[str | None, ...] | None = None,
        text_context=None,
        canonical_video=None,
    ):
        return self.frontend.from_video_latents(
            video_latents,
            task_text=task_text,
            text_context=text_context,
            canonical_video=canonical_video,
        )

    def reset_runtime_state(self) -> None:
        self.frontend.reset_runtime_state()

    def run_core(self, core_input: VisualCoreInput):
        core_output = self.core(core_input)
        core_output.aux.setdefault(
            "weight_source",
            "lingbot_reference_init" if self.reference_core_load_report is not None else "local_init",
        )
        if self.reference_core_load_report is not None:
            core_output.aux.setdefault("reference_core_loaded_keys", len(self.reference_core_load_report.loaded_keys))
        return core_output

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
                stream_ids=frontend_output.video_tokens.new_zeros((batch_size, seq_len), dtype=frontend_output.video_tokens.dtype).long(),
                text_context=frontend_output.conditioning.text_context,
                conditioning=frontend_output.conditioning,
            )
        )

    def run_decode(self, frontend_output, core_output):
        return self.decoder(frontend_output=frontend_output, core_output=core_output)

    def get_lingbot_reference_transformer(self, *, action_dim: int) -> nn.Module:
        key = str(action_dim)
        if key not in self._lingbot_reference_transformers:
            self._lingbot_reference_transformers[key] = build_reference_transformer(self.config, action_dim=action_dim)
        return self._lingbot_reference_transformers[key]

    def ensure_lingbot_reference_transformer_device(self, *, action_dim: int, device) -> nn.Module:
        transformer = self.get_lingbot_reference_transformer(action_dim=action_dim)
        target_dtype = preferred_reference_dtype(device)
        parameter = next(transformer.parameters())
        if parameter.device != device or parameter.dtype != target_dtype:
            transformer.to(device=device, dtype=target_dtype)
        return transformer

    def reset_lingbot_reference_runtime(self, *, action_dim: int, cache_name: str = "open_wam_exact") -> None:
        transformer = self.get_lingbot_reference_transformer(action_dim=action_dim)
        try:
            transformer.clear_pred_cache(cache_name)
        except KeyError:
            pass
        try:
            transformer.clear_cache(cache_name)
        except KeyError:
            pass

    def forward_default(
        self,
        canonical_video,
        *,
        placements: tuple[ViewPlacement, ...] | None = None,
        task_text: tuple[str | None, ...] | None = None,
        include_decode: bool = False,
    ) -> VisualStageOutputs:
        frontend_output = self.run_frontend(canonical_video, placements=placements, task_text=task_text)
        core_output = self.run_default_core(frontend_output)
        decode_output = self.run_decode(frontend_output, core_output) if include_decode else None
        return VisualStageOutputs(frontend=frontend_output, core=core_output, decode=decode_output)
