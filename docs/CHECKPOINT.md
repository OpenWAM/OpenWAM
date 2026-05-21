# OpenWAM Checkpoint: LIBERO-10 + OXE-OpenVLA Pretrain

> **Repository:** [`openwam-data/libero-oxe-pretrain-5k`](https://huggingface.co/openwam-data/libero-oxe-pretrain-5k) (private — request access from Yao Feng)

## Model

| Attribute | Value |
|-----------|-------|
| Architecture | OpenWAM causal video prediction (WAN2.2 TI2V-5B) |
| Parameters | ~5.1B |
| Init weights | [robbyant/lingbot-va-base](https://huggingface.co/robbyant/lingbot-va-base) |
| Latent space | WAN2.2 TI2V-5B VAE — 16x spatial, 4x temporal, 48 channels |
| Precision | bf16-mixed |

## Training

| Attribute | Value |
|-----------|-------|
| Hardware | 4x NVIDIA B200 (FSDP), then resumed on 8x B200 |
| Steps | 5,000 optimizer steps (100,000 seen batches at grad_accum=20) |
| Optimizer | AdamW (lr=1e-5, beta1=0.9, beta2=0.95, wd=0.01) |
| Warmup | 100 steps, constant schedule after |
| Final train loss | ~0.032 |
| W&B run | `gis8g3is` (project: `openwam-mixed-video-pretrain`) |

## Training Data

Two latent-encoded video sources with equal sampling weight (1.0 each):

| Source | Episodes | Format |
|--------|----------|--------|
| LIBERO-10 | 371 | WAN VAE latent (.pt) |
| OXE-OpenVLA | 2,973 | WAN VAE latent (.pt) |

Total: ~3,344 episodes encoded at 128x128, 15 FPS, through WAN2.2 TI2V-5B VAE.
The 128x128 size is the VAE preprocessing resolution, not necessarily the
native camera resolution of the source datasets.

The released latent bundle should be treated as single-slot visual training
data. Some source manifests expose multiple camera streams, but this checkpoint
release does not publish separate per-camera latent sidecars that can be
recombined into new multi-angle layouts.

Sample construction: `causal_prefix_suffix` with buckets `[(1,1), (2,2)]`.

## Files in HF Repo

| File | Size | Use |
|------|------|-----|
| `transformer/diffusion_pytorch_model.safetensors` | 10.2 GB | Inference-ready backbone export |
| `transformer/config.json` | 1 KB | Transformer architecture config |
| `model_state.pt` | 20.5 GB | Full model weights (for fine-tuning or resume) |
| `resolved_config.yaml` | — | Complete training configuration |
| `train_state.json` | — | Step/epoch metadata |
| `README.md` | — | HuggingFace model card |

> **Note:** `full_training_state.pt` (60.5 GB, optimizer + scheduler state) was not uploaded due to the HuggingFace 50 GB per-file limit. Fine-tuning can start from `model_state.pt`; exact optimizer/scheduler resume state is not included.

## Quick Start

```bash
# Download checkpoint (inference weights only, ~10 GB)
python scripts/download_checkpoint.py --mode inference

# Download full model bundle (weights + config/metadata, no optimizer state, ~30 GB)
python scripts/download_checkpoint.py --mode full
```

See [`scripts/download_checkpoint.py`](../scripts/download_checkpoint.py) for details and [`examples/inference_libero_oxe.md`](../examples/inference_libero_oxe.md) for a `model_state.pt` loading example that uses the full bundle.
