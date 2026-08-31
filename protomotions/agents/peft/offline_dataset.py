# SPDX-License-Identifier: Apache-2.0
"""Small, auditable expert-transition dataset for offline PEFT SFT."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class OfflineSFTDataset(Dataset):
    """Eagerly load valid frames from per-clip cache files.

    The initial 50-clip pilot is intentionally small enough to fit in host
    memory.  Keeping chunks separate on disk makes it straightforward to move
    to a sharded/memory-mapped loader when the dataset grows.
    """

    def __init__(self, root: str | Path, splits: set[str]):
        self.root = Path(root).resolve()
        manifests = sorted(self.root.rglob("cache_manifest.json"))
        if not manifests:
            raise FileNotFoundError(f"No cache_manifest.json found under {self.root}")
        self.chunks: list[dict[str, torch.Tensor]] = []
        self.index: list[tuple[int, int]] = []
        self.clip_metadata: list[dict] = []
        common_keys: set[str] | None = None
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("split", "train") not in splits:
                continue
            for entry in manifest["clips"]:
                cache_path = manifest_path.parent / entry["file"]
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                tensors = payload["tensors"]
                keys = set(tensors)
                common_keys = keys if common_keys is None else common_keys & keys
                valid = tensors.get("valid")
                valid_indices = (
                    torch.arange(next(iter(tensors.values())).shape[0])
                    if valid is None
                    else valid.bool().nonzero(as_tuple=False).squeeze(-1)
                )
                chunk_index = len(self.chunks)
                self.chunks.append(tensors)
                self.clip_metadata.append(payload["metadata"])
                self.index.extend((chunk_index, int(frame)) for frame in valid_indices)
        if not self.index:
            raise RuntimeError(
                f"No valid frames for splits {sorted(splits)} under {self.root}"
            )
        if common_keys is None:
            raise RuntimeError("Offline cache contains no tensor keys")
        self.keys = sorted(common_keys - {"valid", "terminated", "motion_time"})

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        chunk_index, frame = self.index[index]
        chunk = self.chunks[chunk_index]
        result = {}
        for key in self.keys:
            value = chunk[key][frame]
            # Cache geometry/state compactly, but execute the model in its
            # native float32 precision.
            result[key] = value.float() if value.is_floating_point() else value
        return result
