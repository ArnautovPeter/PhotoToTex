"""Generate an offline synthetic image-to-LaTeX dataset."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def template_latex(index: int) -> str:
    formulas = [
        r"E=mc^2",
        r"\frac{a}{b}+\frac{c}{d}",
        r"\int_0^1 x^2\,dx=\frac{1}{3}",
        r"\sum_{i=1}^{n} i=\frac{n(n+1)}{2}",
        r"\sqrt{x^2+y^2}",
        r"\sin^2\theta+\cos^2\theta=1",
        r"e^{i\pi}+1=0",
    ]
    formula = formulas[index % len(formulas)]
    return f"The equation is ${formula}$."


def render_formula(latex: str, path: Path, rng: random.Random, size: int = 384) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(4, 2), dpi=size // 4)
        fig.patch.set_facecolor((rng.uniform(0.88, 1),) * 3)
        fig.text(0.5, 0.5, latex, ha="center", va="center", fontsize=18, color="black")
        fig.savefig(path, dpi=size // 4, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        image = Image.open(path).convert("RGB").resize((size, size))
    except Exception as exc:
        LOGGER.warning("matplotlib rendering failed (%s); using Pillow fallback", exc)
        image = Image.new("RGB", (size, size), "white")
    if rng.random() < 0.7:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0, 0.7)))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
    array = np.asarray(image).astype(np.int16)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 3, array.shape)
    image = Image.fromarray(np.uint8(np.clip(array + noise, 0, 255)))
    image.save(path)


def generate(num: int, out: str | Path, seed: int = 42, image_size: int = 384) -> None:
    rng = random.Random(seed)
    out = Path(out)
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in tqdm(range(num), desc="Generating"):
        latex = template_latex(index)
        image_path = image_dir / f"{index:06d}.png"
        render_formula(latex, image_path, rng, image_size)
        records.append({"id": f"sample-{index:06d}", "image": str(image_path.relative_to(out)), "latex": latex})
    rng.shuffle(records)
    splits = {"train": records[: int(num * 0.8)], "val": records[int(num * 0.8): int(num * 0.9)], "test": records[int(num * 0.9):]}
    for split, rows in splits.items():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            handle.write("\n".join(json.dumps(row) for row in rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-model", help="Reserved for an optional local HF backend; templates remain the offline default.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.llm_model:
        LOGGER.warning("The optional LLM backend is not enabled in this minimal offline baseline; using templates.")
    generate(args.num, args.out, args.seed)


if __name__ == "__main__":
    main()
