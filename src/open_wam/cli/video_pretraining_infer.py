"""Generate VPM future-video comparisons from a weights-only pretraining checkpoint."""

import argparse
import json
from dataclasses import replace
from pathlib import Path


def _select_sample(
    dataset, *, source_id: str | None, sample_index: int, multi_view_only: bool
):
    matches = 0
    for index in range(len(dataset)):
        window = dataset.sample_index[index]
        episode = dataset.episode_records[window.episode_key]
        if source_id and episode.source_id != source_id:
            continue
        if multi_view_only and not any(
            stream.augmentation == "multi_view" for stream in episode.streams
        ):
            continue
        if matches == sample_index:
            return dataset[index]
        matches += 1
    raise IndexError("No matching sample at this index")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument(
        "--assets",
        required=True,
        type=Path,
        help="Root with the matching vae/ subdirectory",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", help="Optional source_id filter, such as VPT-09")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--multi-view-only", action="store_true",
        help="Select manifest rows explicitly tagged augmentation=multi_view",
    )
    args = parser.parse_args()
    import torch

    from open_wam.artifacts import load_tensor_artifact
    from open_wam.configs import load_experiment_config
    from open_wam.data import (
        build_train_val_latent_datasets,
        collate_latent_wam_samples,
        move_latent_wam_batch_to_device,
    )
    from open_wam.evals.video_artifacts import to_uint8, write_video_frames
    from open_wam.evals.video_prediction import (
        decode_canonical_latent_views,
        rollout_causal_video_prediction,
    )
    from open_wam.pipelines import build_variant_pipeline_from_config
    from open_wam.utils import seed_everywhere

    if args.out.exists():
        raise FileExistsError("Use a new inference output directory")
    seed_everywhere(args.seed)
    cfg = load_experiment_config(args.cfg)
    cfg = replace(
        cfg,
        backbone=replace(
            cfg.backbone,
            pretrained_model_name_or_path=str(args.assets),
            load_reference_core_weights=False,
            load_wan_vae_frontend=True,
        ),
        inference=replace(cfg.inference, video_num_inference_steps=args.steps),
    )
    train, val = build_train_val_latent_datasets(cfg.data)
    dataset = train if args.split == "train" else val
    sample = _select_sample(
        dataset, source_id=args.source, sample_index=args.sample_index,
        multi_view_only=args.multi_view_only,
    )
    device = torch.device(args.device)
    pipeline = build_variant_pipeline_from_config(cfg)
    payload = load_tensor_artifact(args.weights)
    weights = (
        payload.get("model_state_dict", payload) if isinstance(payload, dict) else None
    )
    if not isinstance(weights, dict):
        raise ValueError("Expected a model state dictionary")
    pipeline.load_state_dict(weights, strict=True)
    del payload, weights
    pipeline.to(device=device).eval()
    batch = move_latent_wam_batch_to_device(
        collate_latent_wam_samples([sample]), device
    )
    with torch.inference_mode():
        rollout = rollout_causal_video_prediction(
            pipeline, batch, num_chunks=args.num_chunks
        )
        layout = batch.metadata[0].get("latent_layout")
        target = decode_canonical_latent_views(
            pipeline, rollout.target_latents, latent_layout=layout, decode_device=device
        )
        predicted = decode_canonical_latent_views(
            pipeline,
            rollout.predicted_latents,
            latent_layout=layout,
            decode_device=device,
        )
    args.out.mkdir(parents=True)
    target, predicted = to_uint8(target), to_uint8(predicted)
    write_video_frames(args.out / "target.mp4", target, fps=15.0)
    write_video_frames(args.out / "prediction.mp4", predicted, fps=15.0)
    (args.out / "summary.json").write_text(
        json.dumps(
            dict(
                source=args.source,
                split=args.split,
                sample_index=args.sample_index,
                seed=args.seed,
                observed_latent_frames=rollout.observed_latent_frames,
                future_latent_frames=rollout.future_latent_frames,
                first_chunk_future_latent_mse=rollout.first_chunk_future_mse,
                sample_metadata=batch.metadata[0],
                steps=args.steps,
            ),
            default=str,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(args.out)


if __name__ == "__main__":
    main()
