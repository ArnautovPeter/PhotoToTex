import pytest
import torch

from src.data.tokenizer import LatexTokenizer
from src.train import resolve_device, validate_checkpoint_compatibility


def test_resolve_device_accepts_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "__version__", "test-build")
    monkeypatch.setattr(torch.version, "cuda", "13.0")

    with pytest.raises(RuntimeError, match="Training was not started on CPU"):
        resolve_device("cuda")


def test_resolve_device_rejects_unknown_device():
    with pytest.raises(ValueError, match="Unsupported training device"):
        resolve_device("mps")


def test_warm_start_requires_matching_tokenizer_and_model_config():
    tokenizer = LatexTokenizer("x^2")
    config = {"d_model": 16, "nhead": 4, "layers": 1, "ff_dim": 32, "max_length": 8, "image_size": [32, 96]}
    checkpoint = {
        "tokenizer": tokenizer.to_dict(),
        "config": config.copy(),
        "preprocessing": "aspect_pad_v1",
    }

    validate_checkpoint_compatibility(checkpoint, tokenizer, config, resume=False)

    mismatched_config = {**config, "d_model": 32}
    with pytest.raises(ValueError, match="model settings do not match"):
        validate_checkpoint_compatibility(checkpoint, tokenizer, mismatched_config, resume=False)

    with pytest.raises(ValueError, match="tokenizer does not exactly match"):
        validate_checkpoint_compatibility(checkpoint, LatexTokenizer("y^2"), config, resume=False)


def test_legacy_preprocessing_requires_explicit_opt_in():
    tokenizer = LatexTokenizer("x")
    config = {"d_model": 16, "nhead": 4, "layers": 1, "ff_dim": 32, "max_length": 8, "image_size": 64}
    checkpoint = {"tokenizer": tokenizer.to_dict(), "config": config.copy()}

    with pytest.raises(ValueError, match="legacy square-stretch preprocessing"):
        validate_checkpoint_compatibility(checkpoint, tokenizer, config, resume=False)

    validate_checkpoint_compatibility(
        checkpoint,
        tokenizer,
        config,
        resume=False,
        allow_legacy_preprocessing=True,
    )


def test_resume_requires_full_training_state():
    tokenizer = LatexTokenizer("x")
    config = {
        "d_model": 16,
        "nhead": 4,
        "layers": 1,
        "ff_dim": 32,
        "max_length": 8,
        "image_size": [32, 96],
        "seed": 1,
        "data_dir": "data",
        "batch_size": 4,
        "learning_rate": 0.001,
        "min_learning_rate": 0.00001,
        "lr_patience": 3,
        "early_stopping_patience": 10,
        "min_delta": 0.001,
        "train_augmentation": False,
        "num_workers": 0,
    }
    checkpoint = {
        "tokenizer": tokenizer.to_dict(),
        "config": config.copy(),
        "preprocessing": "aspect_pad_v1",
    }

    with pytest.raises(ValueError, match="cannot be exactly resumed"):
        validate_checkpoint_compatibility(checkpoint, tokenizer, config, resume=True)
