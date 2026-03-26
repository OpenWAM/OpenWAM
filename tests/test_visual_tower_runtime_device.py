from __future__ import annotations

from types import SimpleNamespace

import torch

from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.models.visual_tower import VisualTower


def test_run_frontend_keeps_frontend_modules_on_input_device(monkeypatch) -> None:
    tower = VisualTower(
        LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            load_wan_vae_frontend=False,
            load_text_conditioning=False,
        )
    )
    recorded_devices: list[torch.device] = []
    original_to = tower.frontend.to
    original_parameters = tower.frontend.parameters

    def tracked_to(*args, **kwargs):
        device = kwargs.get("device")
        if device is None and args:
            device = args[0]
        if device is not None:
            recorded_devices.append(torch.device(device))
        return original_to(*args, **kwargs)

    monkeypatch.setattr(tower.frontend, "to", tracked_to)
    monkeypatch.setattr(
        tower.frontend,
        "parameters",
        lambda recurse=True: iter((SimpleNamespace(device=torch.device("meta")), *tuple(original_parameters(recurse=recurse)))),
    )

    video = torch.randn(1, 3, 1, 32, 32)
    output = tower.run_frontend(video)

    assert recorded_devices == [video.device]
    assert output.video_latents.device == video.device
    assert output.video_tokens.device == video.device
