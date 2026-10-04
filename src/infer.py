"""Run greedy image-to-LaTeX inference."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from PIL import Image

from .data.dataset import preprocess_image
from .data.tokenizer import LatexTokenizer
from .models.seq2seq import ImageToLatex

LOGGER = logging.getLogger(__name__)


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    tokenizer = LatexTokenizer.from_dict(checkpoint["tokenizer"])
    config = checkpoint["config"]
    if "validation" not in checkpoint or isinstance(config.get("image_size"), int):
        LOGGER.warning(
            "Checkpoint %s predates validation-based training and aspect-preserving preprocessing; "
            "regenerate the dataset and retrain before relying on its predictions.",
            path,
        )
    model = ImageToLatex(len(tokenizer), config["d_model"], config["nhead"], config["layers"], config["ff_dim"], config["max_length"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer, config


def predict(checkpoint_path: str, image_path: str) -> str:
    device = torch.device("cpu")
    model, tokenizer, config = load_checkpoint(checkpoint_path, device)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    return predict_image(image, model, tokenizer, config)


def predict_image(image: Image.Image, model, tokenizer, config) -> str:
    tensor = preprocess_image(image, config["image_size"]).unsqueeze(0)
    with torch.inference_mode():
        tokens = model.greedy_decode(tensor, tokenizer.sos_id, tokenizer.eos_id)[0].tolist()
    return tokenizer.decode(tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    print(predict(args.checkpoint, args.image))


if __name__ == "__main__":
    main()
