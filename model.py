"""Decoder-only transformer (GPT-style) language model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Per-layer attention (key, value) tensors, each (B, n_head, T, head_dim).
KVCache = list[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class GPTConfig:
    vocab_size: int = 0  # 0 = take it from the dataset's meta.json at train time
    block_size: int = 256  # maximum context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False  # bias in Linear and LayerNorm


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, past: tuple[torch.Tensor, torch.Tensor] | None = None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0
        if past is None:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
            # New positions see all cached positions; among themselves they stay causal.
            mask = None
            if T > 1:
                mask = torch.ones(T, k.size(2), dtype=torch.bool, device=x.device).tril(k.size(2) - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y)), (k, v)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, past=None):
        a, present = self.attn(self.ln_1(x), past)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be set")
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying

        self.apply(self._init_weights)
        # Scale down residual-branch outputs so activations don't grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _hidden(self, idx: torch.Tensor, past: KVCache | None = None) -> tuple[torch.Tensor, KVCache]:
        """Final hidden states for `idx`, which continues the sequence held in `past`."""
        T = idx.size(1)
        offset = 0 if past is None else past[0][0].size(2)
        if offset + T > self.config.block_size:
            raise ValueError(f"sequence length {offset + T} exceeds block_size {self.config.block_size}")
        pos = torch.arange(offset, offset + T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        present: KVCache = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, None if past is None else past[i])
            present.append(kv)
        return self.ln_f(x), present

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Returns (logits, loss). `loss` is None when no targets are given.

        Target positions equal to -100 are ignored.
        """
        x, _ = self._hidden(idx)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def configure_optimizer(
        self, lr: float, weight_decay: float, betas: tuple[float, float], fused: bool = False
    ) -> torch.optim.AdamW:
        # Decay weight matrices and embeddings; leave biases and LayerNorm gains alone.
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend `idx` (B, T) by up to `max_new_tokens` tokens.

        `temperature=0` is greedy decoding. Generation stops early once every
        row has produced `eos_id`; finished rows are padded with `eos_id`.
        """
        if idx.size(1) == 0:
            raise ValueError("generate needs at least one prompt token")
        block_size = self.config.block_size
        was_training = self.training
        self.eval()
        try:
            past: KVCache | None = None
            finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
            for _ in range(max_new_tokens):
                if idx.size(1) > block_size:
                    # The window has started sliding, which shifts every absolute
                    # position, so cached keys/values are no longer valid.
                    past = None
                    x, _ = self._hidden(idx[:, -block_size:])
                elif past is None:
                    x, past = self._hidden(idx)
                else:
                    x, past = self._hidden(idx[:, -1:], past)
                next_id = _sample(self.lm_head(x[:, -1]), temperature, top_k, top_p)
                if eos_id is not None:
                    next_id = next_id.masked_fill(finished.unsqueeze(1), eos_id)
                    finished = finished | (next_id.squeeze(1) == eos_id)
                idx = torch.cat((idx, next_id), dim=1)
                if eos_id is not None and bool(finished.all()):
                    break
            return idx
        finally:
            self.train(was_training)


def _sample(logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None) -> torch.Tensor:
    """Pick the next token for each row of `logits` (B, V). Returns (B, 1)."""
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits.float() / temperature
    if top_k is not None and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        # Drop a token once the mass of the tokens ranked above it reaches top_p;
        # the top token is therefore always kept.
        remove = probs.cumsum(dim=-1) - probs >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return torch.multinomial(logits.softmax(dim=-1), num_samples=1)
