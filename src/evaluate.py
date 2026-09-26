"""Evaluate a checkpoint on a generated split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.dataset import LatexImageDataset, collate_batch
from .infer import load_checkpoint
from .train import cer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    device = torch.device("cpu")
    model, tokenizer, config = load_checkpoint(args.checkpoint, device)
    split_file = Path(config["data_dir"]) / f"{args.split}.jsonl"
    dataset = LatexImageDataset(split_file, tokenizer, config["image_size"], config["max_length"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=lambda b: collate_batch(b, tokenizer.pad_id))
    exact, total_cer, count = 0, 0.0, 0
    for batch in loader:
        predictions = model.greedy_decode(batch["image"], tokenizer.sos_id, tokenizer.eos_id)
        for prediction, target in zip(predictions, batch["latex"]):
            value = tokenizer.decode(prediction.tolist())
            exact += value == target
            total_cer += cer(target, value)
            count += 1
            if count <= 3:
                print(f"target: {target}\nprediction: {value}\n")
    print(f"exact_match: {exact / max(count, 1):.4f}")
    print(f"character_error_rate: {total_cer / max(count, 1):.4f}")


if __name__ == "__main__":
    main()
