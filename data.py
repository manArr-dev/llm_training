"""Token datasets backed by memory-mapped binary files written by `llm.prepare`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

META_FILE = "meta.json"
TOKENIZER_FILE = "tokenizer.json"


def load_meta(data_dir: str | Path) -> dict:
    path = Path(data_dir) / META_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `llm-prepare` to build the dataset first")
    return json.loads(path.read_text(encoding="utf-8"))


class TokenDataset:
    """A flat stream of token ids, sampled as random fixed-length windows."""

    def __init__(self, path: str | Path, block_size: int, dtype: str):
        self.block_size = block_size
        self.data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        if len(self.data) <= block_size:
            raise ValueError(f"{path} has {len(self.data)} tokens, need more than block_size={block_size}")

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(
        self, batch_size: int, device: torch.device, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inputs and next-token targets, each (batch_size, block_size)."""
        T = self.block_size
        starts = torch.randint(len(self.data) - T, (batch_size,), generator=generator).tolist()
        chunk = torch.from_numpy(np.stack([self.data[i : i + T + 1] for i in starts]).astype(np.int64))
        x, y = chunk[:, :-1], chunk[:, 1:]
        if device.type == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)
