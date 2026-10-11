"""Training loop."""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Sequence

import torch
import yaml

from llm.checkpoint import BEST, LAST, load_checkpoint, save_checkpoint
from llm.config import Config, TrainConfig
from llm.data import TOKENIZER_FILE, TokenDataset, load_meta
from llm.model import GPT

log = logging.getLogger(__name__)

METRICS_FILE = "metrics.jsonl"


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay from learning_rate to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, datasets: dict[str, TokenDataset], cfg: TrainConfig, device, autocast) -> dict[str, float]:
    was_training = model.training
    model.eval()
    out = {}
    for split, dataset in datasets.items():
        # Fixed seed: every evaluation scores the same batches, so losses are comparable.
        generator = torch.Generator().manual_seed(cfg.seed)
        total = 0.0
        for _ in range(cfg.eval_iters):
            x, y = dataset.get_batch(cfg.batch_size, device, generator)
            with autocast:
                _, loss = model(x, y)
            total += loss.item()
        out[split] = total / cfg.eval_iters
    model.train(was_training)
    return out


def train(cfg: Config, resume: bool = False) -> dict[str, float]:
    """Train to `cfg.train.max_steps`. Returns the final step and losses."""
    tc = cfg.train
    device = resolve_device(tc.device)
    out_dir = Path(tc.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(cfg.data.dir)
    meta = load_meta(data_dir)
    if cfg.model.vocab_size == 0:
        cfg.model.vocab_size = meta["vocab_size"]
    elif cfg.model.vocab_size < meta["vocab_size"]:
        raise ValueError(f"model.vocab_size={cfg.model.vocab_size} is smaller than the dataset's {meta['vocab_size']}")
    datasets = {
        split: TokenDataset(data_dir / f"{split}.bin", cfg.model.block_size, meta["dtype"])
        for split in ("train", "val")
    }

    ckpt = None
    if resume:
        ckpt = load_checkpoint(out_dir / LAST, device)
        if ckpt["config"]["model"] != cfg.to_dict()["model"]:
            raise ValueError(f"model config differs from the checkpoint's: {ckpt['config']['model']}")

    # Keep the run directory self-contained: generation needs only this folder.
    if (data_dir / TOKENIZER_FILE).exists():
        shutil.copyfile(data_dir / TOKENIZER_FILE, out_dir / TOKENIZER_FILE)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")

    torch.manual_seed(tc.seed)
    torch.set_float32_matmul_precision("high")

    raw_model = GPT(cfg.model).to(device)
    optimizer = raw_model.configure_optimizer(
        tc.learning_rate, tc.weight_decay, (tc.beta1, tc.beta2), fused=device.type == "cuda"
    )

    use_amp = tc.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    autocast = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and amp_dtype is torch.float16)

    step = 0
    best_val_loss = float("inf")
    if ckpt is not None:
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]
        log.info("resumed from %s at step %d", out_dir / LAST, step)
    del ckpt

    model = torch.compile(raw_model) if tc.compile else raw_model
    log.info(
        "device=%s amp=%s params=%.2fM train_tokens=%d val_tokens=%d",
        device, amp_dtype if use_amp else "off", raw_model.num_params() / 1e6,
        len(datasets["train"]), len(datasets["val"]),
    )  # fmt: skip

    def checkpoint_payload() -> dict:
        return {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "config": cfg.to_dict(),
        }

    # Offset by step so a resumed run does not replay the batches it already saw.
    generator = torch.Generator().manual_seed(tc.seed + step)
    result = {"step": step, "best_val_loss": best_val_loss}
    tokens_per_step = tc.batch_size * tc.grad_accum_steps * cfg.model.block_size
    model.train()
    optimizer.zero_grad(set_to_none=True)
    window_start, window_steps = time.perf_counter(), 0

    with open(out_dir / METRICS_FILE, "a", encoding="utf-8") as metrics:

        def record(row: dict) -> None:
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()

        try:
            while step < tc.max_steps:
                lr = lr_at(step, tc)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                loss_value = 0.0
                for _ in range(tc.grad_accum_steps):
                    x, y = datasets["train"].get_batch(tc.batch_size, device, generator)
                    with autocast:
                        _, loss = model(x, y)
                    loss = loss / tc.grad_accum_steps
                    scaler.scale(loss).backward()
                    loss_value += loss.item()
                if not math.isfinite(loss_value):
                    raise FloatingPointError(f"loss is {loss_value} at step {step}")

                if tc.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), tc.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                window_steps += 1

                if step % tc.log_interval == 0:
                    elapsed = time.perf_counter() - window_start
                    tok_per_s = window_steps * tokens_per_step / elapsed
                    log.info("step %d/%d loss %.4f lr %.2e %.0f tok/s", step, tc.max_steps, loss_value, lr, tok_per_s)
                    record({"step": step, "train_loss": loss_value, "lr": lr, "tokens_per_s": tok_per_s})
                    window_start, window_steps = time.perf_counter(), 0

                if step % tc.eval_interval == 0 or step == tc.max_steps:
                    losses = estimate_loss(model, datasets, tc, device, autocast)
                    improved = losses["val"] < best_val_loss
                    best_val_loss = min(best_val_loss, losses["val"])
                    log.info(
                        "eval step %d train %.4f val %.4f%s",
                        step, losses["train"], losses["val"], " (best)" if improved else "",
                    )  # fmt: skip
                    record({"step": step, "eval_train_loss": losses["train"], "eval_val_loss": losses["val"]})
                    payload = checkpoint_payload()
                    save_checkpoint(out_dir / LAST, payload)
                    if improved:
                        save_checkpoint(out_dir / BEST, payload)
                    result = {
                        "step": step,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                        "best_val_loss": best_val_loss,
                    }
                    window_start, window_steps = time.perf_counter(), 0
        except KeyboardInterrupt:
            log.warning("interrupted at step %d; saving %s (continue with --resume)", step, out_dir / LAST)
            save_checkpoint(out_dir / LAST, checkpoint_payload())
            result["step"] = step

    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a GPT language model.")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--resume", action="store_true", help="continue from <out_dir>/last.pt")
    parser.add_argument(
        "--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override a config value; repeatable"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(Config.load(args.config, args.set), resume=args.resume)


if __name__ == "__main__":
    main()
