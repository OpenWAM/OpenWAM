"""Read an explicitly selected, provenance-checked offline UMT5 prompt cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import torch

from open_wam.artifacts.resolver import ArtifactResolver

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"Prompt cache {field} must be a lowercase SHA256 digest.")
    return value


class OfflinePromptCache:
    """CPU-backed cache with immutable prompt inventory and bounded device copies.

    The encoder fingerprint authenticates the declared encoder-file hashes; an
    optional expected fingerprint pins the selected encoder revision. Large model
    files are not re-read at runtime. Embeddings use weights-only tensor loading.
    """

    def __init__(
        self,
        root: Path,
        *,
        max_text_tokens: int,
        text_dim: int,
        expected_encoder_fingerprint: str | None = None,
        resolver: ArtifactResolver | None = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.resolver = resolver or ArtifactResolver()
        self.max_text_tokens = int(max_text_tokens)
        self.text_dim = int(text_dim)
        if self.max_text_tokens <= 0 or self.text_dim <= 0:
            raise ValueError("Prompt cache runtime dimensions must be positive.")
        index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        if index.get("format_version") != 1 or index.get("complete") is not True:
            raise ValueError("Prompt cache index must be complete format_version=1.")
        if index.get("storage") != "unpadded" or index.get("dtype") != "bfloat16":
            raise ValueError("Prompt cache requires unpadded bfloat16 embeddings.")
        encoded_max_tokens = index.get("max_text_tokens")
        if (
            type(encoded_max_tokens) is not int
            or encoded_max_tokens != self.max_text_tokens
            or index.get("text_dim") != self.text_dim
        ):
            raise ValueError(
                "Prompt cache token limit and text_dim must match the model exactly; "
                "re-encode the cache with the configured max_text_tokens and text_dim."
            )
        self.encoded_max_tokens = encoded_max_tokens
        hashes = index.get("encoder_files_sha256")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("Prompt cache is missing encoder-file provenance.")
        for name, digest in hashes.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Prompt cache encoder filenames must be nonempty strings.")
            _require_sha256(digest, field=f"encoder_files_sha256[{name!r}]")
        self.encoder_fingerprint = _require_sha256(
            index.get("encoder_fingerprint"), field="encoder_fingerprint"
        )
        actual_fingerprint = hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if actual_fingerprint != self.encoder_fingerprint:
            raise ValueError("Prompt cache encoder fingerprint does not match its provenance.")
        if expected_encoder_fingerprint is not None:
            expected = _require_sha256(expected_encoder_fingerprint, field="expected encoder fingerprint")
            if self.encoder_fingerprint != expected:
                raise ValueError("Prompt cache encoder fingerprint differs from the expected encoder.")
        inventory = (self.root / "prompts.jsonl").read_bytes()
        expected_inventory = _require_sha256(index.get("prompts_sha256"), field="prompts_sha256")
        if hashlib.sha256(inventory).hexdigest() != expected_inventory:
            raise ValueError("Prompt cache prompt inventory checksum mismatch.")
        self.entries: dict[str, tuple[str, int]] = {}
        for line in inventory.decode("utf-8").splitlines():
            entry = json.loads(line)
            prompt, key, tokens = entry.get("prompt"), entry.get("sha256"), entry.get("tokens")
            if not isinstance(prompt, str) or prompt in self.entries:
                raise ValueError("Prompt cache inventory has an invalid or duplicate prompt.")
            if key != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
                raise ValueError("Prompt cache inventory contains a mismatched prompt SHA256.")
            if type(tokens) is not int or not 0 < tokens <= self.encoded_max_tokens:
                raise ValueError("Prompt cache inventory has an invalid token count.")
            self.entries[prompt] = (key, tokens)
        if (
            not self.entries
            or index.get("cached_prompts") != len(self.entries)
            or index.get("unique_prompts") != len(self.entries)
        ):
            raise ValueError("Prompt cache inventory count does not match its complete index.")

    def encode_prompts(
        self,
        prompts: tuple[str, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not dtype.is_floating_point:
            raise ValueError("Prompt cache output dtype must be floating point.")
        if not prompts:
            return torch.empty((0, self.max_text_tokens, self.text_dim), device=device, dtype=dtype)
        missing = [prompt for prompt in prompts if prompt not in self.entries]
        if missing:
            key = hashlib.sha256(missing[0].encode("utf-8")).hexdigest()
            raise KeyError(f"Prompt SHA256 {key} is absent from the explicit prompt cache; regenerate its inventory.")
        output = torch.zeros((len(prompts), self.max_text_tokens, self.text_dim), dtype=dtype)
        for row, prompt in enumerate(prompts):
            key, tokens = self.entries[prompt]
            path = self.root / "embeddings" / f"{key}.pt"
            with self.resolver.materialize(path) as local_path:
                tensor = torch.load(local_path, map_location="cpu", weights_only=True)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.layout != torch.strided
                or tensor.dtype != torch.bfloat16
                or tuple(tensor.shape) != (tokens, self.text_dim)
                or not torch.isfinite(tensor).all().item()
            ):
                raise ValueError(f"Prompt cache embedding has invalid dtype, shape, or values: {path}")
            output[row, :tokens] = tensor.to(dtype=dtype)
        return output.to(device=device)
