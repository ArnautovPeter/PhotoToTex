"""Run greedy image-to-LaTeX inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .data.tokenizer import LatexTokenizer
from .models.seq2seq import ImageToLatex


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    tokenizer = LatexTokenizer.from_dict(checkpoint["tokenizer"])
    config = checkpoint["config"]
    model = ImageToLatex(len(tokenizer), config["d_model"], config["nhead"], config["layers"], config["ff_dim"], config["max_length"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer, config


def predict(checkpoint_path: str, image_path: str) -> str:
    device = torch.device("cpu")
    model, tokenizer, config = load_checkpoint(checkpoint_path, device)
    image = Image.open(image_path).convert("RGB")
    tensor = transforms.Compose([transforms.Resize((config["image_size"], config["image_size"])), transforms.ToTensor()])(image).unsqueeze(0)
    return tokenizer.decode(model.greedy_decode(tensor, tokenizer.sos_id, tokenizer.eos_id)[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    print(predict(args.checkpoint, args.image))


if __name__ == "__main__":
    main()
