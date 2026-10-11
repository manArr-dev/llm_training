"""Experiment configuration, loaded from YAML with optional dotted overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

from llm.model import GPTConfig


@dataclass
class DataConfig:
    dir: str = "data/tiny"  # output directory of `llm-prepare`


@dataclass
class TrainConfig:
    out_dir: str = "runs/default"
    seed: int = 1337
    device: str = "auto"  # auto | cpu | cuda | cuda:N | mps
    amp: bool = True  # mixed precision (CUDA only)
    compile: bool = False  # torch.compile the model

    batch_size: int = 32
    grad_accum_steps: int = 1  # effective batch = batch_size * grad_accum_steps
    max_steps: int = 5000

    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # 0 disables clipping

    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10


@dataclass
class Config:
    model: GPTConfig = field(default_factory=GPTConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        raw = raw or {}
        unknown = set(raw) - {"model", "train", "data"}
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        return cls(
            model=_build(GPTConfig, raw.get("model"), "model"),
            train=_build(TrainConfig, raw.get("train"), "train"),
            data=_build(DataConfig, raw.get("data"), "data"),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def load(cls, path: str | Path, overrides: Sequence[str] = ()) -> "Config":
        """Load a YAML file, then apply overrides like `train.max_steps=100`."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for item in overrides:
            key, sep, value = item.partition("=")
            section, dot, name = key.partition(".")
            if not sep or not dot:
                raise ValueError(f"override must look like section.key=value, got {item!r}")
            raw.setdefault(section, {})[name] = yaml.safe_load(value)
        return cls.from_dict(raw)


def _build(cls, raw: dict[str, Any] | None, section: str):
    raw = raw or {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise ValueError(f"unknown keys in [{section}]: {sorted(unknown)}")
    return cls(**{name: _coerce(value, fields[name].default, f"{section}.{name}") for name, value in raw.items()})


def _coerce(value: Any, default: Any, name: str) -> Any:
    """Convert `value` to the type of the field's default.

    PyYAML reads `3e-4` as a string (it wants `3.0e-4`), so floats in
    particular need this.
    """
    kind = type(default)
    try:
        if kind is bool:
            if not isinstance(value, bool):
                raise ValueError
            return value
        if kind is int:
            if isinstance(value, bool) or float(value) != int(float(value)):
                raise ValueError
            return int(float(value))
        if kind is float:
            if isinstance(value, bool):
                raise ValueError
            return float(value)
        if kind is str:
            if not isinstance(value, str):
                raise ValueError
            return value
    except (TypeError, ValueError):
        raise ValueError(f"{name}: expected {kind.__name__}, got {value!r}") from None
    return value
