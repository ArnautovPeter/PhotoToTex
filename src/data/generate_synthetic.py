"""Generate a varied, offline synthetic image-to-LaTeX dataset."""

from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from tqdm import tqdm

def random_formula(rng: random.Random) -> str:
    """Build a randomized, standalone formula using Matplotlib mathtext."""
    kind = rng.randrange(18)
    variable = rng.choice(("x", "y", "z", "t"))
    if kind == 0:
        coefficient, offset, solution = (rng.randint(2, 20), rng.randint(1, 30), rng.randint(1, 30))
        return rf"{coefficient}{variable}+{offset}={coefficient * solution + offset}"
    if kind == 1:
        coefficient, offset, solution = (rng.randint(2, 20), rng.randint(1, 30), rng.randint(1, 30))
        return rf"{coefficient}{variable}-{offset}={coefficient * solution - offset}"
    if kind == 2:
        first_root, second_root = rng.randint(1, 15), rng.randint(1, 15)
        return rf"{variable}^2-{first_root + second_root}{variable}+{first_root * second_root}=0"
    if kind == 3:
        a, b, c, d = (rng.randint(1, 20) for _ in range(4))
        total = Fraction(a, b) + Fraction(c, d)
        return rf"\frac{{{a}}}{{{b}}}+\frac{{{c}}}{{{d}}}=\frac{{{total.numerator}}}{{{total.denominator}}}"
    if kind == 4:
        value = rng.randint(2, 20)
        return rf"\sqrt{{{value * value}}}={value}"
    if kind == 5:
        first, second = rng.randint(1, 20), rng.randint(1, 20)
        return rf"x^2+y^2={first * first + second * second}"
    if kind == 6:
        upper = rng.randint(2, 30)
        return rf"\sum_{{i=1}}^{{{upper}}}i=\frac{{{upper * (upper + 1)}}}{{2}}"
    if kind == 7:
        upper = rng.randint(2, 20)
        return rf"\int_0^{{{upper}}}x\,dx=\frac{{{upper * upper}}}{{2}}"
    if kind == 8:
        return r"\sin^2(x)+\cos^2(x)=1"
    if kind == 9:
        exponent = rng.randint(1, 12)
        return rf"e^{{{exponent}}}={np.e**exponent:.2f}"
    if kind == 10:
        base, exponent = rng.randint(2, 12), rng.randint(2, 8)
        return rf"\log_{{{base}}}({base}^{exponent})={exponent}"
    if kind == 11:
        first, second = rng.randint(1, 15), rng.randint(1, 15)
        return rf"({variable}+{first})({variable}+{second})={variable}^2+{first + second}{variable}+{first * second}"
    if kind == 12:
        numerator, denominator = rng.randint(1, 30), rng.randint(2, 20)
        return rf"\frac{{{numerator}}}{{{denominator}}}={numerator / denominator:.2f}"
    if kind == 13:
        value = rng.randint(1, 30)
        return rf"|{variable}|={value}"
    if kind == 14:
        left, right = rng.randint(1, 20), rng.randint(21, 100)
        return rf"{left}{variable}\leq {right}"
    if kind == 15:
        upper = rng.randint(3, 12)
        lower = rng.randint(1, upper - 1)
        return rf"\binom{{{upper}}}{{{lower}}}=\frac{{{upper}!}}{{{lower}!({upper - lower})!}}"
    if kind == 16:
        numerator, denominator = rng.randint(1, 15), rng.randint(2, 15)
        return rf"\left(\frac{{{numerator}}}{{{denominator}}}\right)^2=\frac{{{numerator * numerator}}}{{{denominator * denominator}}}"
    first, second = rng.randint(1, 12), rng.randint(1, 12)
    return rf"\lim_{{x\to 0}}\frac{{\sin({first}x)}}{{{first}x}}=1"


def template_latex(rng: random.Random) -> str:
    """Compatibility wrapper retained for callers of the original generator."""
    return rf"${random_formula(rng)}$"


def render_formula(latex: str, path: Path, rng: random.Random, size: int = 384) -> None:
    """Render one formula on a wide paper-like canvas without distorting it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width = size
    height = max(64, size // 3)
    paper = rng.uniform(0.88, 1.0)
    background = (paper, paper * rng.uniform(0.99, 1.0), paper * rng.uniform(0.97, 1.0))
    ink = rng.choice(("#101010", "#202020", "#171c27", "#252525"))
    fontset = rng.choice(("cm", "stix", "stixsans", "dejavusans"))
    figure = plt.figure(figsize=(width / 96, height / 96), dpi=96, facecolor=background)
    try:
        with matplotlib.rc_context({"mathtext.fontset": fontset}):
            figure.text(
                0.5,
                0.5,
                f"${latex.strip('$')}$",
                ha="center",
                va="center",
                fontsize=rng.randint(22, 38),
                color=ink,
            )
            buffer = BytesIO()
            figure.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.05, facecolor=background)
    finally:
        plt.close(figure)

    buffer.seek(0)
    with Image.open(buffer) as rendered:
        formula_image = rendered.convert("RGB")
    formula_image = ImageOps.contain(formula_image, (width - 24, height - 20), method=Image.Resampling.LANCZOS)
    image = Image.new("RGB", (width, height), tuple(round(channel * 255) for channel in background))
    image.paste(formula_image, ((width - formula_image.width) // 2, (height - formula_image.height) // 2))

    if rng.random() < 0.25:
        image = image.rotate(rng.uniform(-2.0, 2.0), resample=Image.Resampling.BICUBIC, fillcolor=image.getpixel((0, 0)))
    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.6)))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.88, 1.12))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
    array = np.asarray(image).astype(np.int16)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(0.5, 5), array.shape)
    image = Image.fromarray(np.uint8(np.clip(array + noise, 0, 255)))
    if rng.random() < 0.15:
        jpeg = BytesIO()
        image.save(jpeg, format="JPEG", quality=rng.randint(65, 92))
        jpeg.seek(0)
        with Image.open(jpeg) as compressed:
            image = compressed.convert("RGB")
    image.save(path)


def generate(num: int, out: str | Path, seed: int = 42, image_size: int = 384) -> None:
    if num < 3:
        raise ValueError("num must be at least 3 to create train, validation, and test splits")
    if image_size < 96:
        raise ValueError("image_size must be at least 96 pixels")
    rng = random.Random(seed)
    out = Path(out)
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    seen_formulas: set[str] = set()
    for index in tqdm(range(num), desc="Generating"):
        for _ in range(1000):
            latex = template_latex(rng)
            formula = latex.strip("$")
            if formula not in seen_formulas:
                seen_formulas.add(formula)
                break
        else:
            raise RuntimeError(f"Could not generate {num} unique equations")
        image_path = image_dir / f"{index:06d}.png"
        render_formula(latex, image_path, rng, image_size)
        records.append({"id": f"sample-{index:06d}", "image": str(image_path.relative_to(out)), "latex": latex})
    rng.shuffle(records)
    train_end = max(1, min(num - 2, int(num * 0.8)))
    val_end = max(train_end + 1, min(num - 1, int(num * 0.9)))
    splits = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }
    for split, rows in splits.items():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            handle.write("\n".join(json.dumps(row) for row in rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=5000)
    parser.add_argument("--out", default="data/formula_synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=384, help="Image width; generated images are three times wider than tall")
    args = parser.parse_args()
    generate(args.num, args.out, args.seed, args.image_size)


if __name__ == "__main__":
    main()
