"""Train the baseline image-to-LaTeX recognizer."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import LatexImageDataset, collate_batch
from .data.tokenizer import LatexTokenizer
from .models.seq2seq import ImageToLatex

LOGGER = logging.getLogger(__name__)


def cer(reference: str, hypothesis: str) -> float:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, 1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1] / max(1, len(reference))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_model(model, loader, tokenizer, device) -> tuple[float, float]:
    model.eval()
    exact, errors, count = 0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            predictions = model.greedy_decode(batch["image"].to(device), tokenizer.sos_id, tokenizer.eos_id)
            for prediction, target in zip(predictions, batch["latex"]):
                value = tokenizer.decode(prediction.tolist())
                exact += value == target
                errors += cer(target, value)
                count += 1
    return exact / max(count, 1), errors / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    device = torch.device(config.get("device", "cpu"))
    data_dir = Path(config["data_dir"])
    records = [json.loads(line) for line in (data_dir / "train.jsonl").read_text().splitlines() if line]
    tokenizer = LatexTokenizer(char for row in records for char in row["latex"])
    dataset_args = {"tokenizer": tokenizer, "image_size": config["image_size"], "max_length": config["max_length"]}
    train_set = LatexImageDataset(data_dir / "train.jsonl", **dataset_args)
    val_set = LatexImageDataset(data_dir / "val.jsonl", **dataset_args)
    def collate(batch): return collate_batch(batch, tokenizer.pad_id)
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, collate_fn=collate)
    model = ImageToLatex(len(tokenizer), config["d_model"], config["nhead"],
                         config["layers"], config["ff_dim"], config["max_length"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    best = float("inf")
    for epoch in range(config["epochs"]):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            tokens = batch["tokens"].to(device)
            logits = model(batch["image"].to(device), tokens[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        exact, char_error = evaluate_model(model, val_loader, tokenizer, device)
        if epoch % 50 == 0 or epoch == config["epochs"] - 1:
            LOGGER.info("epoch=%d loss=%.4f val_exact=%.3f val_cer=%.3f", epoch + 1, np.mean(losses), exact, char_error)
        if np.mean(losses) < best:
            best = float(np.mean(losses))
            checkpoint = {"model": model.state_dict(), "tokenizer": tokenizer.to_dict(), "config": config}
            Path(config["checkpoint"]).parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, config["checkpoint"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
