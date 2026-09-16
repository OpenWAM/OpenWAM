from __future__ import annotations

import pickle

import torch

from open_wam.models.decoder_artifacts import ParallelTrainArtifacts


def test_training_artifact_contract_roundtrip() -> None:
    value = ParallelTrainArtifacts(
        input_dict={"probe": torch.tensor([1.0])},
        latent_scheduler=object(),
        action_scheduler=object(),
    )
    restored = pickle.loads(pickle.dumps(value))
    assert type(restored) is ParallelTrainArtifacts
    assert torch.equal(restored.input_dict["probe"], value.input_dict["probe"])
    assert restored.dynamics_objective is value.dynamics_objective
