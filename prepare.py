"""Build a training dataset from raw text: train the tokenizer, encode, split.

Writes train.bin, val.bin, tokenizer.json and meta.json to the output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from llm.data import META_FILE, TOKENIZER_FILE
from llm.tokenizer import BPETokenizer

log = logging.getLogger(__name__)


def _iter_text(path: Path, chunk_bytes: int = 1 << 20) -> Iterator[str]:
    """Yield a file as chunks of whole lines, so large files are never fully in memory."""
    with open(path, encoding="utf-8", errors="replace") as f:
        while lines := f.readlines(chunk_bytes):
            yield "".join(lines)


def prepare(
    inputs: Sequence[str | Path],
    out_dir: str | Path,
    vocab_size: int = 4096,
    val_fraction: float = 0.1,
    min_frequency: int = 2,
    tokenizer_path: str | Path | None = None,
) -> dict:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    paths = [Path(p) for p in inputs]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if tokenizer_path is not None:
        tokenizer = BPETokenizer.load(tokenizer_path)
        log.info("loaded tokenizer from %s (vocab %d)", tokenizer_path, tokenizer.vocab_size)
    else:
        log.info("training tokenizer (target vocab %d)", vocab_size)
        tokenizer = BPETokenizer.train(
            (chunk for p in paths for chunk in _iter_text(p)), vocab_size, min_frequency=min_frequency
        )
        log.info("tokenizer vocab size: %d", tokenizer.vocab_size)
    tokenizer.save(out / TOKENIZER_FILE)

    dtype = np.uint16 if tokenizer.vocab_size <= 2**16 else np.uint32
    eot = tokenizer.eot_id
    arrays: list[np.ndarray] = []
    for path in paths:
        for chunk in _iter_text(path):
            arrays.append(np.asarray(tokenizer.encode(chunk), dtype=dtype))
        if eot is not None:
            # Each input file is one document.
            arrays.append(np.asarray([eot], dtype=dtype))
    tokens = np.concatenate(arrays) if arrays else np.empty(0, dtype=dtype)

    n_val = int(len(tokens) * val_fraction)
    n_train = len(tokens) - n_val
    if n_train == 0 or n_val == 0:
        raise ValueError(f"not enough data to split: {len(tokens)} tokens")
    tokens[:n_train].tofile(out / "train.bin")
    tokens[n_train:].tofile(out / "val.bin")

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "dtype": np.dtype(dtype).name,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "eot_id": eot,
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote %s: %d train tokens, %d val tokens", out, n_train, n_val)
    return meta


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="UTF-8 text files; each is treated as one document")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--val-fraction", type=float, default=0.1, help="tail fraction of the token stream held out")
    parser.add_argument("--min-frequency", type=int, default=2, help="stop merging below this pair count")
    parser.add_argument("--tokenizer", help="reuse an existing tokenizer.json instead of training one")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepare(args.input, args.out_dir, args.vocab_size, args.val_fraction, args.min_frequency, args.tokenizer)


if __name__ == "__main__":
    main()
