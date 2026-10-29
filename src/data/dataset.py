"""JSONL image/LaTeX dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .tokenizer import LatexTokenizer


def split_annotation(data_dir: str | Path, split: str) -> Path:
    """Prefer physically separated v2 splits, while retaining old checkpoint support."""
    root = Path(data_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not json.loads(manifest_path.read_text()).get(
        "complete", False
    ):
        raise ValueError(f"Dataset generation is incomplete: {root}")
    modern = root / split / "annotations.jsonl"
    return modern if modern.is_file() else root / f"{split}.jsonl"


def image_size_to_hw(image_size: int | list[int] | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        height = width = image_size
    elif isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        height, width = image_size
    else:
        raise ValueError(
            "image_size must be a positive integer or a [height, width] pair"
        )
    if (
        not isinstance(height, int)
        or not isinstance(width, int)
        or height < 1
        or width < 1
    ):
        raise ValueError("image_size dimensions must be positive integers")
    return height, width


def preprocess_image(
    image: Image.Image,
    image_size: int | list[int] | tuple[int, int],
) -> torch.Tensor:
    """Resize an image to fit a fixed canvas while preserving its aspect ratio."""
    height, width = image_size_to_hw(image_size)
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(
        resized, ((width - resized.width) // 2, (height - resized.height) // 2)
    )
    return TF.to_tensor(canvas)


def augment_photo(image: Image.Image) -> Image.Image:
    """Apply mild camera-like transforms without altering equation content."""
    image = transforms.RandomPerspective(
        distortion_scale=0.1,
        p=0.5,
        interpolation=InterpolationMode.BICUBIC,
        fill=(255, 255, 255),
    )(image)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.12))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))

    pixels = np.asarray(image).astype(np.float32)
    height, width = pixels.shape[:2]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x = x / max(width - 1, 1)
    y = y / max(height - 1, 1)
    illumination = (
        1
        + random.uniform(-0.18, 0.18) * (x - 0.5)
        + random.uniform(-0.18, 0.18) * (y - 0.5)
    )
    if random.random() < 0.6:
        center_x, center_y = random.random(), random.random()
        radius_x, radius_y = random.uniform(0.18, 0.65), random.uniform(0.2, 0.8)
        shadow = np.exp(
            -(((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2) * 2
        )
        illumination *= 1 - random.uniform(0.08, 0.32) * shadow
    pixels *= illumination[..., None]
    pixels += np.random.normal(0, random.uniform(0.5, 4.0), pixels.shape).astype(
        np.float32
    )
    image = Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))

    if random.random() < 0.25:
        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(45, 88))
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            image = compressed.convert("RGB")
    return image


class LatexImageDataset(Dataset[dict[str, Any]]):
    """Loads generated PNGs and tokenized LaTeX targets."""

    def __init__(
        self,
        annotation_file: str | Path,
        tokenizer: LatexTokenizer,
        image_size: int | list[int] | tuple[int, int] = 64,
        max_length: int = 160,
        augment: bool = False,
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
        self.image_size = image_size_to_hw(image_size)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = Path(record["image"])
        if not image_path.is_absolute():
            image_path = self.root / image_path
        with Image.open(image_path) as source:
            source_image = source.convert("RGB")
        if self.augment:
            source_image = augment_photo(source_image)
        image = preprocess_image(source_image, self.image_size)
        target = self.tokenizer.encode(record["latex"])
        if len(target) > self.max_length:
            raise ValueError(
                f"Target in {self.annotation_file} record {record['id']} has {len(target)} "
                f"tokens, exceeding max_length={self.max_length}; increase max_length."
            )
        return {
            "image": image,
            "tokens": torch.tensor(target, dtype=torch.long),
            "id": record["id"],
            "latex": record["latex"],
        }


def collate_batch(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch])
    max_length = max(item["tokens"].numel() for item in batch)
    tokens = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    for row, item in enumerate(batch):
        tokens[row, : item["tokens"].numel()] = item["tokens"]
    return {
        "image": images,
        "tokens": tokens,
        "id": [x["id"] for x in batch],
        "latex": [x["latex"] for x in batch],
    }
