"""Sample text from a trained run directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch

from llm.checkpoint import BEST, load_model
from llm.data import TOKENIZER_FILE
from llm.tokenizer import BPETokenizer
from llm.train import resolve_device


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run directory (the training out_dir)")
    parser.add_argument("--checkpoint", default=BEST, help="checkpoint file inside the run directory")
    parser.add_argument("--prompt", default="", help="text to continue; empty starts from the end-of-text token")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    run = Path(args.run)
    device = resolve_device(args.device)
    tokenizer = BPETokenizer.load(run / TOKENIZER_FILE)
    model, _ = load_model(run / args.checkpoint, device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        if tokenizer.eot_id is None:
            parser.error("--prompt is required: this tokenizer has no end-of-text token to start from")
        prompt_ids = [tokenizer.eot_id]
    idx = torch.tensor([prompt_ids] * args.num_samples, dtype=torch.long, device=device)

    out = model.generate(
        idx,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_id=tokenizer.eot_id,
    )

    # Windows consoles often default to a legacy code page.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for i, row in enumerate(out.tolist()):
        if i:
            print("\n" + "-" * 40)
        print(tokenizer.decode(row, skip_special=True))


if __name__ == "__main__":
    main()
