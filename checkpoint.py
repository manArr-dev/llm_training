"""Checkpoint save/load."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from llm.config import Config
from llm.model import GPT

LAST = "last.pt"
BEST = "best.pt"


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Write atomically, so a crash mid-save never leaves a truncated checkpoint."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    # weights_only: checkpoints hold tensors and plain containers, never pickled code.
    return torch.load(path, map_location=device, weights_only=True)


def load_model(path: str | Path, device: torch.device | str = "cpu") -> tuple[GPT, Config]:
    """Rebuild a trained model from a checkpoint, in eval mode."""
    ckpt = load_checkpoint(path, device)
    config = Config.from_dict(ckpt["config"])
    model = GPT(config.model)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), config
