"""Byte-level BPE tokenizer.

Ids 0..255 are raw bytes, followed by one id per learned merge, followed by the
special tokens. Because the base alphabet is bytes, any string round-trips
exactly and there is no unknown token.
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

EOT = "<|endoftext|>"

FORMAT_VERSION = 1

# Pre-tokenization: merges never cross these chunk boundaries. A leading space
# is attached to the word that follows it, GPT-2 style. The alternatives cover
# every character, so "".join(findall(text)) == text.
_SPLIT = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+|_+")

_CACHE_LIMIT = 200_000


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    a, b = pair
    out: list[int] = []
    i, n = 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges: Sequence[tuple[int, int]], special_tokens: Sequence[str] = (EOT,)):
        self.merges: list[tuple[int, int]] = [(int(a), int(b)) for a, b in merges]
        self.special_tokens: list[str] = list(special_tokens)
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special tokens must be unique")

        self._vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for i, (a, b) in enumerate(self.merges):
            if not (0 <= a < 256 + i and 0 <= b < 256 + i):
                raise ValueError(f"merge {i} references an id that does not exist yet: {(a, b)}")
            self._vocab.append(self._vocab[a] + self._vocab[b])
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}

        base = len(self._vocab)
        self._special_to_id = {tok: base + i for i, tok in enumerate(self.special_tokens)}
        self._id_to_special = {i: tok.encode("utf-8") for tok, i in self._special_to_id.items()}
        self._special_re = (
            re.compile("(" + "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)) + ")")
            if self.special_tokens
            else None
        )
        self._cache: dict[str, tuple[int, ...]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab) + len(self.special_tokens)

    @property
    def eot_id(self) -> int | None:
        return self._special_to_id.get(EOT)

    def token_to_id(self, token: str) -> int:
        return self._special_to_id[token]

    # ------------------------------------------------------------------ train

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: Sequence[str] = (EOT,),
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Learn merges from `texts`.

        The result may have fewer than `vocab_size` tokens if the corpus runs
        out of pairs occurring at least `min_frequency` times.
        """
        num_merges = vocab_size - 256 - len(special_tokens)
        if num_merges < 0:
            raise ValueError(f"vocab_size must be at least {256 + len(special_tokens)}")

        chunk_freqs: Counter[str] = Counter()
        for text in texts:
            chunk_freqs.update(_SPLIT.findall(text))
        words = [list(chunk.encode("utf-8")) for chunk in chunk_freqs]
        freqs = list(chunk_freqs.values())

        pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        # Words that may contain each pair. Entries can go stale; merging a
        # word that no longer has the pair is a no-op.
        where: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        for i, (word, freq) in enumerate(zip(words, freqs)):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freq
                where[pair].add(i)

        # Max-heap with lazy invalidation: every count change pushes a fresh
        # entry, and entries whose count is out of date are skipped on pop.
        # Ties break towards the smallest pair, which keeps training deterministic.
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        while len(merges) < num_merges and heap:
            neg_count, pair = heapq.heappop(heap)
            count = pair_counts.get(pair, 0)
            if count != -neg_count:
                continue
            if count < min_frequency:
                break

            new_id = 256 + len(merges)
            merges.append(pair)
            changed: set[tuple[int, int]] = set()
            for i in where.pop(pair):
                word, freq = words[i], freqs[i]
                for p in zip(word, word[1:]):
                    pair_counts[p] -= freq
                    changed.add(p)
                word = words[i] = _merge(word, pair, new_id)
                for p in zip(word, word[1:]):
                    pair_counts[p] += freq
                    where[p].add(i)
                    changed.add(p)
            for p in changed:
                count = pair_counts[p]
                if count > 0:
                    heapq.heappush(heap, (-count, p))
                else:
                    del pair_counts[p]

        return cls(merges, special_tokens)

    # ---------------------------------------------------------- encode/decode

    def _encode_chunk(self, chunk: str) -> tuple[int, ...]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        ranks = self._ranks
        while len(ids) >= 2:
            # Apply merges in the order they were learned.
            pair = min(zip(ids, ids[1:]), key=lambda p: ranks.get(p, float("inf")))
            rank = ranks.get(pair)
            if rank is None:
                break
            ids = _merge(ids, pair, 256 + rank)
        out = tuple(ids)
        if len(self._cache) < _CACHE_LIMIT:
            self._cache[chunk] = out
        return out

    def _encode_ordinary(self, text: str, out: list[int]) -> None:
        for chunk in _SPLIT.findall(text):
            out.extend(self._encode_chunk(chunk))

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Encode text to token ids.

        Special tokens appearing in `text` are only mapped to their ids when
        `allow_special` is set; otherwise they are encoded as plain text, so
        untrusted input cannot inject control tokens.
        """
        out: list[int] = []
        if not allow_special or self._special_re is None:
            self._encode_ordinary(text, out)
            return out
        for part in self._special_re.split(text):
            special_id = self._special_to_id.get(part)
            if special_id is not None:
                out.append(special_id)
            elif part:
                self._encode_ordinary(part, out)
        return out

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        parts: list[bytes] = []
        n_ordinary = len(self._vocab)
        for i in ids:
            i = int(i)
            if 0 <= i < n_ordinary:
                parts.append(self._vocab[i])
            elif i in self._id_to_special:
                if not skip_special:
                    parts.append(self._id_to_special[i])
            else:
                raise ValueError(f"token id {i} is out of range for vocab size {self.vocab_size}")
        # A truncated sequence can end mid-character; don't fail on it.
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        payload = {
            "version": FORMAT_VERSION,
            "merges": [list(m) for m in self.merges],
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported tokenizer format version: {payload.get('version')!r}")
        return cls([tuple(m) for m in payload["merges"]], payload["special_tokens"])
