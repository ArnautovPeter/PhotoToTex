"""Train the baseline image-to-LaTeX recognizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import LatexImageDataset, collate_batch, split_annotation
from .data.tokenizer import LatexTokenizer
from .models.seq2seq import ImageToLatex

LOGGER = logging.getLogger(__name__)

MODEL_CONFIG_KEYS = ("d_model", "nhead", "layers", "ff_dim", "max_length", "image_size")
RESUME_CONFIG_KEYS = (
    "seed",
    "data_dir",
    "batch_size",
    "learning_rate",
    "min_learning_rate",
    "lr_patience",
    "early_stopping_patience",
    "min_delta",
    "train_augmentation",
    "num_workers",
)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def load_trusted_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_checkpoint_compatibility(
    checkpoint: dict[str, Any],
    tokenizer: LatexTokenizer,
    config: dict[str, Any],
    *,
    resume: bool,
    allow_legacy_preprocessing: bool = False,
    data_fingerprint: str | None = None,
) -> None:
    old_config = checkpoint.get("config")
    if not isinstance(old_config, dict):
        raise ValueError("Checkpoint has no compatible training config.")

    tokenizer_data = checkpoint.get("tokenizer")
    old_tokenizer = tokenizer_data.get("tokens") if isinstance(tokenizer_data, dict) else None
    if old_tokenizer != tokenizer.to_dict()["tokens"]:
        raise ValueError(
            "Checkpoint tokenizer does not exactly match the current training data. "
            "Use data with the same LaTeX character vocabulary or start a fresh model."
        )

    mismatches = [
        f"{key}: checkpoint={old_config.get(key)!r}, current={config.get(key)!r}"
        for key in MODEL_CONFIG_KEYS
        if old_config.get(key) != config.get(key)
    ]
    if mismatches:
        raise ValueError("Checkpoint model settings do not match:\n  " + "\n  ".join(mismatches))

    legacy_preprocessing = (
        checkpoint.get("preprocessing") != "aspect_pad_v1"
        and isinstance(old_config.get("image_size"), int)
    )
    if legacy_preprocessing and not allow_legacy_preprocessing:
        raise ValueError(
            "Checkpoint uses legacy square-stretch preprocessing. It cannot be safely "
            "continued with the current aspect-preserving preprocessing. Use a compatible "
            "checkpoint, or explicitly pass --allow-legacy-preprocessing after reviewing "
            "the expected image-domain mismatch."
        )
    if legacy_preprocessing:
        LOGGER.warning(
            "Using legacy checkpoint preprocessing with the current aspect-preserving pipeline; "
            "warm-start quality may degrade."
        )

    if resume:
        if checkpoint.get("preprocessing") != "aspect_pad_v1":
            raise ValueError(
                "Cannot exactly resume a checkpoint without an explicit aspect_pad_v1 preprocessing marker. "
                "Use --warm-start after checking tokenizer/settings compatibility."
            )
        for key in RESUME_CONFIG_KEYS:
            if old_config.get(key) != config.get(key):
                raise ValueError(
                    f"Cannot resume: training setting {key!r} differs "
                    f"(checkpoint={old_config.get(key)!r}, current={config.get(key)!r})."
                )
        if not isinstance(checkpoint.get("training_state"), dict):
            raise ValueError(
                "Checkpoint does not contain optimizer/scheduler state and cannot be exactly resumed. "
                "Use --warm-start instead."
            )
        old_fingerprint = checkpoint["training_state"].get("data_fingerprint")
        if old_fingerprint != data_fingerprint:
            raise ValueError(
                "Cannot resume because the training/validation data changed since the checkpoint. "
                "Use --warm-start to keep weights and begin a new run."
            )


def fingerprint_dataset(data_dir: Path) -> str:
    digest = hashlib.sha256()
    for split in ("train", "val"):
        annotation_path = split_annotation(data_dir, split)
        annotation_bytes = annotation_path.read_bytes()
        digest.update(split.encode("utf-8"))
        digest.update(annotation_bytes)
        for line in annotation_bytes.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = annotation_path.parent / image_path
            stat = image_path.stat()
            digest.update(str(image_path).encode("utf-8"))
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    return digest.hexdigest()


def numpy_rng_state() -> dict[str, Any]:
    state = np.random.get_state()
    return {
        "bit_generator": state[0],
        "state": state[1].tolist(),
        "position": state[2],
        "has_gauss": state[3],
        "cached_gaussian": state[4],
    }


def restore_numpy_rng_state(state: dict[str, Any]) -> None:
    np.random.set_state(
        (
            state["bit_generator"],
            np.asarray(state["state"], dtype=np.uint32),
            state["position"],
            state["has_gauss"],
            state["cached_gaussian"],
        )
    )


def save_torch_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    try:
        device = torch.device(device_name)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid training device {device_name!r}. Use 'cpu', 'cuda', or 'cuda:N'.") from error

    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported training device {device_name!r}. Use 'cpu', 'cuda', or 'cuda:N'.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            cuda_build = torch.version.cuda or "CPU-only"
            raise RuntimeError(
                f"CUDA was requested, but it is unavailable (PyTorch {torch.__version__}, "
                f"CUDA build {cuda_build}). Install a PyTorch CUDA build compatible with the "
                "installed NVIDIA driver; see the GPU setup instructions in README.md. "
                "Training was not started on CPU."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} is unavailable; "
                f"found {torch.cuda.device_count()} CUDA device(s)."
            )
    return device


def evaluate_model(model, loader, tokenizer, device) -> tuple[float, float]:
    model.eval()
    exact, errors, count = 0, 0.0, 0
    with torch.inference_mode():
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
    parser.add_argument("--config", default="configs/data_v2.yaml")
    parser.add_argument("--device", help="Override the configured device (for example, cuda or cuda:0)")
    parser.add_argument("--checkpoint", help="Output best-checkpoint path (use a new path for each experiment)")
    parser.add_argument("--epochs", type=int, help="Override the total epoch limit from the config")
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--warm-start", help="Initialize weights from a compatible checkpoint")
    continuation.add_argument("--resume", help="Resume training from a checkpoint containing optimizer state")
    parser.add_argument(
        "--allow-legacy-preprocessing",
        action="store_true",
        help="Allow warm start from a checkpoint trained with legacy square-stretch preprocessing",
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be at least 1.")
        config["epochs"] = args.epochs
    checkpoint_path = Path(config["checkpoint"])
    for source in (args.warm_start, args.resume):
        if source and Path(source).resolve() == checkpoint_path.resolve():
            raise ValueError("Input checkpoint and output checkpoint must be different paths.")

    device = resolve_device(args.device or config.get("device", "cpu"))
    if device.type == "cuda":
        selected_index = device.index if device.index is not None else torch.cuda.current_device()
        LOGGER.info("Training on GPU %d: %s", selected_index, torch.cuda.get_device_name(selected_index))
    else:
        LOGGER.info("Training on CPU")
    set_seed(int(config["seed"]))
    data_dir = Path(config["data_dir"])
    data_fingerprint = fingerprint_dataset(data_dir)
    records = [json.loads(line) for line in (split_annotation(data_dir, "train")).read_text().splitlines() if line]
    if not records:
        raise ValueError(f"Training split is empty: {split_annotation(data_dir, 'train')}")
    tokenizer = LatexTokenizer(char for row in records for char in row["latex"])
    dataset_args = {"tokenizer": tokenizer, "image_size": config["image_size"], "max_length": config["max_length"]}
    train_set = LatexImageDataset(
        split_annotation(data_dir, "train"),
        **dataset_args,
        augment=bool(config.get("train_augmentation", False)),
    )
    val_set = LatexImageDataset(split_annotation(data_dir, "val"), **dataset_args)
    if not val_set:
        raise ValueError(f"Validation split is empty: {split_annotation(data_dir, 'val')}")

    def collate(batch):
        return collate_batch(batch, tokenizer.pad_id)

    loader_generator = torch.Generator().manual_seed(int(config["seed"]))
    num_workers = int(config.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("num_workers must be zero or greater.")
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate,
        generator=loader_generator,
        pin_memory=device.type == "cuda",
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        num_workers=num_workers,
    )
    model = ImageToLatex(
        len(tokenizer),
        config["d_model"],
        config["nhead"],
        config["layers"],
        config["ff_dim"],
        config["max_length"],
    ).to(device)
    source_checkpoint = None
    if args.warm_start or args.resume:
        source_path = args.warm_start or args.resume
        source_checkpoint = load_trusted_checkpoint(source_path)
        validate_checkpoint_compatibility(
            source_checkpoint,
            tokenizer,
            config,
            resume=bool(args.resume),
            allow_legacy_preprocessing=args.allow_legacy_preprocessing and bool(args.warm_start),
            data_fingerprint=data_fingerprint,
        )
        model.load_state_dict(source_checkpoint["model"], strict=True)
        LOGGER.info("Loaded compatible model weights from %s", source_path)
    elif checkpoint_path.exists() or checkpoint_path.with_suffix(".last.pt").exists():
        raise FileExistsError(
            f"Training output already exists at {checkpoint_path}. Choose a new --checkpoint path, "
            "or use --warm-start with a separate output path."
        )

    if args.warm_start and (checkpoint_path.exists() or checkpoint_path.with_suffix(".last.pt").exists()):
        raise FileExistsError(
            f"Warm-start output already exists at {checkpoint_path}; use a fresh --checkpoint path "
            "to preserve previous experiments."
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=int(config.get("lr_patience", 3)),
        min_lr=float(config.get("min_learning_rate", 1e-5)),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    best_cer = float("inf")
    best_exact = 0.0
    epochs_without_improvement = 0
    patience = int(config.get("early_stopping_patience", 10))
    min_delta = float(config.get("min_delta", 0.001))
    checkpoint_path = Path(config["checkpoint"])
    history_path = checkpoint_path.with_suffix(".history.json")
    metrics_path = checkpoint_path.with_suffix(".metrics.json")
    last_checkpoint_path = checkpoint_path.with_suffix(".last.pt")
    history: list[dict[str, float | int]] = []
    start_epoch = 0
    training_state: dict[str, Any] = {}
    if args.resume:
        training_state = source_checkpoint["training_state"]
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        history = training_state["history"]
        start_epoch = training_state["next_epoch"]
        best_cer = training_state["best_cer"]
        best_exact = training_state["best_exact"]
        epochs_without_improvement = training_state["epochs_without_improvement"]
        loader_generator.set_state(training_state["loader_generator_state"])
        random.setstate(training_state["python_rng_state"])
        restore_numpy_rng_state(training_state["numpy_rng_state"])
        torch.set_rng_state(training_state["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_states" in training_state:
            torch.cuda.set_rng_state_all(training_state["cuda_rng_states"])
        LOGGER.info("Resuming at epoch %d from %s", start_epoch + 1, args.resume)
        write_json_atomic(history_path, {"history": history})
    else:
        if args.warm_start:
            LOGGER.info("Warm start initializes weights only; optimizer and schedule start fresh.")
        write_json_atomic(history_path, {"history": history})
        write_json_atomic(metrics_path, {"splits": {}})

    if start_epoch >= int(config["epochs"]):
        raise ValueError(
            f"Resume checkpoint is already at epoch {start_epoch}, while configured epochs="
            f"{config['epochs']}. Increase epochs to continue."
        )

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            tokens = batch["tokens"].to(device, non_blocking=device.type == "cuda")
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            logits = model(images, tokens[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        exact, char_error = evaluate_model(model, val_loader, tokenizer, device)
        scheduler.step(char_error)
        learning_rate = optimizer.param_groups[0]["lr"]
        LOGGER.info(
            "epoch=%d train_loss=%.4f val_exact=%.3f val_cer=%.3f lr=%.6g",
            epoch + 1,
            np.mean(losses),
            exact,
            char_error,
            learning_rate,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_exact_match": exact,
                "val_cer": char_error,
                "learning_rate": learning_rate,
            }
        )
        write_json_atomic(history_path, {"history": history})

        improved = char_error < best_cer - min_delta or (
            abs(char_error - best_cer) <= min_delta and exact > best_exact
        )
        if improved:
            best_cer = char_error
            best_exact = exact
            epochs_without_improvement = 0
            checkpoint = {
                "model": model.state_dict(),
                "tokenizer": tokenizer.to_dict(),
                "config": config,
                "epoch": epoch + 1,
                "validation": {"exact_match": exact, "cer": char_error},
                "history": history.copy(),
                "preprocessing": "aspect_pad_v1",
                "warm_started_from": args.warm_start,
            }
            save_torch_checkpoint(checkpoint_path, checkpoint)
            LOGGER.info("Saved best validation checkpoint to %s", checkpoint_path)
        else:
            epochs_without_improvement += 1

        state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "next_epoch": epoch + 1,
            "best_cer": best_cer,
            "best_exact": best_exact,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history.copy(),
            "loader_generator_state": loader_generator.get_state(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": numpy_rng_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "data_fingerprint": data_fingerprint,
        }
        save_torch_checkpoint(
            last_checkpoint_path,
            {
                "model": model.state_dict(),
                "tokenizer": tokenizer.to_dict(),
                "config": config,
                "epoch": epoch + 1,
                "history": history.copy(),
                "preprocessing": "aspect_pad_v1",
                "training_state": state,
            },
        )
        if epochs_without_improvement >= patience:
            LOGGER.info(
                "Early stopping after %d epochs without validation improvement; best val_cer=%.3f",
                epochs_without_improvement,
                best_cer,
            )
            break

    LOGGER.info("Best validation metrics: exact_match=%.3f cer=%.3f", best_exact, best_cer)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
