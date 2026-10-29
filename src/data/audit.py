"""Audit a physically separated generated dataset without training a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from .generate_synthetic import SPLITS, audit_records


def audit_dataset(root: str | Path) -> dict:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Dataset generation is incomplete")
    records = {}
    paths = set()
    for split in SPLITS:
        annotation = root / split / "annotations.jsonl"
        rows = [
            json.loads(line)
            for line in annotation.read_text().splitlines()
            if line.strip()
        ]
        if not rows or len(rows) != manifest["counts"][split]:
            raise ValueError(f"Incorrect count in {split}")
        for row in rows:
            path = (annotation.parent / row["image"]).resolve()
            if (
                not path.is_relative_to((root / split / "images").resolve())
                or path in paths
            ):
                raise ValueError(f"Image path escapes split or is duplicated: {path}")
            paths.add(path)
            with Image.open(path) as source:
                image = source.convert("RGB")
                digest = hashlib.sha256(
                    image.tobytes() + str(image.size).encode()
                ).hexdigest()
            if digest != row["pixel_sha256"]:
                raise ValueError(f"Image checksum mismatch: {path}")
            if len(row["latex"]) + 2 > manifest.get("max_target_tokens", 256):
                raise ValueError(f"Target too long: {row['id']}")
        records[split] = rows
    unique = audit_records(records)
    train_chars = set("".join(row["latex"] for row in records["train"]))
    for split in ("val", "test"):
        unseen = set("".join(row["latex"] for row in records[split])) - train_chars
        if unseen:
            raise ValueError(f"{split} has unseen target characters: {unseen}")
    fonts = [{row["font"] for row in records[s]} for s in SPLITS]
    if any(fonts[i] & fonts[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Held-out font pools overlap")
    return {
        "counts": manifest["counts"],
        "unique": unique,
        "verified_images": len(paths),
        "cross_split_leakage": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    print(json.dumps(audit_dataset(args.root), indent=2))


if __name__ == "__main__":
    main()
