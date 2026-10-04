"""Evaluate a checkpoint on a generated split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.dataset import LatexImageDataset, collate_batch
from .infer import load_checkpoint
from .train import cer, resolve_device, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--annotations", help="Evaluate a JSONL file of labeled images outside the configured dataset")
    parser.add_argument("--name", help="Metrics key for --annotations (defaults to the filename stem)")
    parser.add_argument("--device", help="Override the configured device (for example, cuda or cuda:0)")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    del checkpoint
    device = resolve_device(args.device or config.get("device", "cpu"))
    model, tokenizer, config = load_checkpoint(args.checkpoint, device)
    split_file = Path(args.annotations) if args.annotations else Path(config["data_dir"]) / f"{args.split}.jsonl"
    if not split_file.is_file():
        raise FileNotFoundError(f"Evaluation split does not exist: {split_file}")
    dataset = LatexImageDataset(split_file, tokenizer, config["image_size"], config["max_length"])
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        collate_fn=lambda b: collate_batch(b, tokenizer.pad_id),
        pin_memory=device.type == "cuda",
    )
    exact, total_cer, count = 0, 0.0, 0
    unknown_characters, target_characters = 0, 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            predictions = model.greedy_decode(images, tokenizer.sos_id, tokenizer.eos_id)
            for prediction, target in zip(predictions, batch["latex"]):
                value = tokenizer.decode(prediction.tolist())
                exact += value == target
                total_cer += cer(target, value)
                target_tokens = tokenizer.encode(target, add_special_tokens=False)
                unknown_characters += sum(token == tokenizer.unk_id for token in target_tokens)
                target_characters += len(target_tokens)
                count += 1
                if count <= 3:
                    print(f"target: {target}\nprediction: {value}\n")
    if count == 0:
        raise ValueError(f"Evaluation split is empty: {split_file}")

    result = {
        "count": count,
        "exact_match": exact / count,
        "cer": total_cer / count,
        "unknown_char_rate": unknown_characters / max(target_characters, 1),
    }
    metrics_path = Path(args.checkpoint).with_suffix(".metrics.json")
    metrics = {"splits": {}}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_name = args.name or (split_file.stem if args.annotations else args.split)
    metrics.setdefault("splits", {})[metrics_name] = result
    write_json_atomic(metrics_path, metrics)
    print(f"exact_match: {result['exact_match']:.4f}")
    print(f"character_error_rate: {result['cer']:.4f}")
    print(f"unknown_character_rate: {result['unknown_char_rate']:.4f}")


if __name__ == "__main__":
    main()
