from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a lightweight model_state.pt from a full_training_state.pt checkpoint."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to full_training_state.pt or a checkpoint_step_* directory containing it.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output path. Defaults to sibling model_state.pt.",
    )
    args = parser.parse_args()

    input_path = _resolve_input_checkpoint(Path(args.input))
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else input_path.parent / "model_state.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"extract.input {input_path}", flush=True)
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Expected checkpoint payload to be a dict.")
    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise ValueError("Expected `model_state_dict` inside full_training_state payload.")
    print(f"extract.num_tensors {sum(1 for value in model_state_dict.values() if isinstance(value, torch.Tensor))}", flush=True)
    torch.save({"model_state_dict": model_state_dict}, output_path)
    print(f"extract.output {output_path}", flush=True)


def _resolve_input_checkpoint(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if candidate.is_file():
        return candidate
    full_state = candidate / "full_training_state.pt"
    if full_state.is_file():
        return full_state
    raise FileNotFoundError(f"Could not resolve full_training_state.pt from {path}.")


if __name__ == "__main__":
    main()
