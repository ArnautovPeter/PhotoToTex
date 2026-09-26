"""JSONL image/LaTeX dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .tokenizer import LatexTokenizer


class LatexImageDataset(Dataset[dict[str, Any]]):
    """Loads generated PNGs and tokenized LaTeX targets."""

    def __init__(
        self,
        annotation_file: str | Path,
        tokenizer: LatexTokenizer,
        image_size: int = 64,
        max_length: int = 160,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.root = self.annotation_file.parent
        self.records = [
            json.loads(line)
            for line in self.annotation_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = Path(record["image"])
        if not image_path.is_absolute():
            image_path = self.root / image_path
        image = self.transform(Image.open(image_path).convert("RGB"))
        target = self.tokenizer.encode(record["latex"])[: self.max_length]
        if target[-1] != self.tokenizer.eos_id:
            target[-1] = self.tokenizer.eos_id
        return {"image": image, "tokens": torch.tensor(target, dtype=torch.long), "id": record["id"], "latex": record["latex"]}


def collate_batch(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch])
    max_length = max(item["tokens"].numel() for item in batch)
    tokens = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    for row, item in enumerate(batch):
        tokens[row, : item["tokens"].numel()] = item["tokens"]
    return {"image": images, "tokens": tokens, "id": [x["id"] for x in batch], "latex": [x["latex"] for x in batch]}
