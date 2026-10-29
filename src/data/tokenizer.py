"""Character-level tokenizer for compact LaTeX strings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class LatexTokenizer:
    """A deterministic character tokenizer with four reserved tokens."""

    PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"

    def __init__(self, vocabulary: Iterable[str] | None = None) -> None:
        chars = sorted(set(vocabulary or []))
        self.tokens = [self.PAD, self.SOS, self.EOS, self.UNK] + [
            char
            for char in chars
            if char not in {self.PAD, self.SOS, self.EOS, self.UNK}
        ]
        self.stoi = {token: index for index, token in enumerate(self.tokens)}
        self.itos = {index: token for token, index in self.stoi.items()}

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]

    @property
    def sos_id(self) -> int:
        return self.stoi[self.SOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.UNK]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self.stoi.get(char, self.unk_id) for char in text]
        if add_special_tokens:
            return [self.sos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        special = {self.PAD, self.SOS, self.EOS, self.UNK}
        result: list[str] = []
        for index in ids:
            token = self.itos.get(int(index), self.UNK)
            if token == self.EOS:
                break
            if not (skip_special_tokens and token in special):
                result.append(token)
        return "".join(result)

    def to_dict(self) -> dict[str, list[str]]:
        return {"tokens": self.tokens}

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> "LatexTokenizer":
        tokenizer = cls([])
        tokenizer.tokens = list(data["tokens"])
        tokenizer.stoi = {token: index for index, token in enumerate(tokenizer.tokens)}
        tokenizer.itos = {index: token for token, index in tokenizer.stoi.items()}
        return tokenizer

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "LatexTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
