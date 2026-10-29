"""Compositional formulas, structural holdouts and physically separated splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm

SPLITS = ("train", "val", "test")
STYLE_FONTS = {"train": ("cm", "newtx"), "val": ("pazo",), "test": ("euler",)}


def composed_formula(rng, depth=3, renderer="tex"):
    """Return LaTeX and a value-independent expression-tree signature."""
    if depth == 0 or rng.random() < 0.22:
        return rng.choice((rng.choice("xyztabc"), str(rng.randint(1, 99)))), "atom"
    operations = [
        "add",
        "sub",
        "mul",
        "frac",
        "sqrt",
        "power",
        "subscript",
        "sin",
        "cos",
        "log",
        "sum",
        "integral",
        "equal",
        "paren",
    ]
    if renderer == "tex":
        operations += ["matrix", "cases"]
    op = rng.choice(operations)
    a, sa = composed_formula(rng, depth - 1, renderer)
    if op in ("sqrt", "sin", "cos", "log", "paren", "sum", "integral"):
        values = {
            "sqrt": rf"\sqrt{{{a}}}",
            "sin": rf"\sin\left({a}\right)",
            "cos": rf"\cos\left({a}\right)",
            "log": rf"\log\left({a}\right)",
            "paren": rf"\left({a}\right)",
            "sum": rf"\sum_{{i=1}}^{{{rng.randint(2, 9)}}}\left({a}\right)",
            "integral": rf"\int_{{0}}^{{{rng.randint(1, 9)}}}\left({a}\right)\,dx",
        }
        return values[op], f"{op}({sa})"
    b, sb = composed_formula(rng, depth - 1, renderer)
    values = {
        "add": rf"{{{a}}}+{{{b}}}",
        "sub": rf"{{{a}}}-{{{b}}}",
        "mul": rf"\left({a}\right)\cdot\left({b}\right)",
        "frac": rf"\frac{{{a}}}{{{b}}}",
        "power": rf"{{{a}}}^{{{b}}}",
        "subscript": rf"{{{a}}}_{{{b}}}",
        "equal": rf"{{{a}}}={{{b}}}",
        "matrix": rf"\begin{{pmatrix}}{a} & {b} \\ {rng.randint(1, 9)} & {rng.choice('xy')}\end{{pmatrix}}",
        "cases": rf"\begin{{cases}}{a} & x>0 \\ {b} & x\leq 0\end{{cases}}",
    }
    # Also group reversed commutative expressions, conservatively preventing leakage.
    children = sorted((sa, sb)) if op in ("add", "mul", "equal") else (sa, sb)
    return values[op], f"{op}({','.join(children)})"


def assign_split(signature, seed):
    bucket = (
        int(hashlib.sha256(f"split-v2:{seed}:{signature}".encode()).hexdigest()[:8], 16)
        % 100
    )
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def render_tex(latex, rng, font, workdir):
    package = {
        "cm": "",
        "newtx": r"\usepackage{newtxmath}",
        "pazo": r"\usepackage{mathpazo}",
        "euler": r"\usepackage{eulervm}",
    }[font]
    size = rng.choice((14, 17, 20, 24))
    source = (
        r"\documentclass[border=2pt]{standalone}\usepackage{amsmath,amssymb}"
        + package
        + r"\pagestyle{empty}\begin{document}"
        + rf"\fontsize{{{size}}}{{{size + 4}}}\selectfont"
        + r"$\displaystyle "
        + latex
        + r"$\end{document}"
    )
    (workdir / "formula.tex").write_text(source)
    subprocess.run(
        [
            "/usr/bin/pdflatex",
            "-no-shell-escape",
            "-interaction=batchmode",
            "-halt-on-error",
            "formula.tex",
        ],
        cwd=workdir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
        timeout=25,
    )
    subprocess.run(
        [
            "/usr/bin/pdftoppm",
            "-f",
            "1",
            "-singlefile",
            "-r",
            str(rng.choice((110, 150, 180))),
            "-png",
            "formula.pdf",
            "formula",
        ],
        cwd=workdir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
        timeout=25,
    )
    with Image.open(workdir / "formula.png") as source_image:
        image = source_image.convert("RGB")
    from PIL import ImageChops

    bounds = (
        ImageChops.difference(image, Image.new("RGB", image.size, "white"))
        .convert("L")
        .point(lambda p: 255 if p > 35 else 0)
        .getbbox()
    )
    if not bounds:
        raise ValueError("TeX produced an empty image")
    return image.crop(bounds)


def vary_capture(image, rng):
    """Static, seeded capture variation: includes clean and degraded samples."""
    image = ImageOps.expand(
        image, border=(rng.randint(8, 35), rng.randint(8, 25)), fill="white"
    )
    profile = rng.choice(("clean", "scan", "camera"))
    if profile == "clean":
        return image, profile
    if profile == "camera":
        # Rotation uses expand so content isn't clipped; perspective keeps a margin.
        image = image.rotate(
            rng.uniform(-4, 4),
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor="white",
        )
        w, h = image.size
        shear = rng.uniform(-0.07, 0.07)
        image = image.transform(
            (w, h),
            Image.Transform.AFFINE,
            (1, shear, -shear * h / 2, rng.uniform(-0.025, 0.025), 1, 0),
            resample=Image.Resampling.BICUBIC,
            fillcolor="white",
        )
    pixels = np.asarray(image).astype(np.float32)
    h, w = pixels.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    illumination = (
        1
        + rng.uniform(-0.22, 0.22) * (xx / max(w - 1, 1) - 0.5)
        + rng.uniform(-0.18, 0.18) * (yy / max(h - 1, 1) - 0.5)
    )
    pixels *= illumination[..., None]
    pixels += np.random.default_rng(rng.randrange(2**32)).normal(
        0, rng.uniform(0.2, 4), pixels.shape
    )
    image = Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))
    image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0, 0.65)))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(55, 95))
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB"), profile


def audit_records(records):
    """Reject cross-split content groups, duplicate labels and duplicate pixels."""
    owners = {key: {} for key in ("group_id", "latex", "pixel_sha256")}
    for split, rows in records.items():
        for row in rows:
            for key, seen in owners.items():
                value = row[key]
                if value in seen and seen[value] != split:
                    raise ValueError(f"Cross-split leakage for {key}: {value}")
                seen[value] = split
    return {key: len(values) for key, values in owners.items()}


def generate(num, out, seed=42, image_size=384, renderer="tex", variants=2):
    """Create a fresh v2 dataset. num counts images, not expression groups."""
    if num < 30 or variants < 1 or image_size < 96:
        raise ValueError("num >= 30, variants >= 1 and image_size >= 96 are required")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Dataset destination must be empty: {out}")
    if renderer not in ("tex", "mathtext"):
        raise ValueError("renderer must be tex or mathtext")
    if renderer == "tex" and not (
        Path("/usr/bin/pdflatex").exists() and Path("/usr/bin/pdftoppm").exists()
    ):
        raise RuntimeError(
            "TeX rendering requires pdflatex and pdftoppm; choose --renderer mathtext explicitly for lightweight tests"
        )
    rng = random.Random(seed)
    records = {split: [] for split in SPLITS}
    # Exact counts are quotas; assignment of groups is independent of quotas.
    quotas = {
        "train": num - 2 * max(1, num // 10),
        "val": max(1, num // 10),
        "test": max(1, num // 10),
    }
    labels = set()
    group_counts = {}
    failures = 0
    attempts = 0
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps({"schema_version": 2, "complete": False})
    )
    for split in SPLITS:
        (out / split / "images").mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="photototex-render-") as temporary:
        workdir = Path(temporary)
        with tqdm(total=num, desc="Generating v2") as progress:
            while any(len(records[s]) < quotas[s] for s in SPLITS):
                attempts += 1
                if attempts > num * 300:
                    raise RuntimeError(
                        "Could not fill splits; inspect rendering failures or increase formula diversity"
                    )
                latex, signature = composed_formula(
                    rng, rng.choice((2, 3, 4)), renderer
                )
                if len(latex) + 2 > 256 or latex in labels:
                    continue
                split = assign_split(signature, seed)
                if group_counts.get(signature, 0) >= 8:
                    continue
                if len(records[split]) >= quotas[split]:
                    continue
                font = (
                    rng.choice(STYLE_FONTS[split])
                    if renderer == "tex"
                    else {"train": "cm", "val": "stix", "test": "dejavusans"}[split]
                )
                try:
                    if renderer == "tex":
                        base = render_tex(latex, rng, font, workdir)
                    else:
                        import matplotlib

                        matplotlib.use("Agg")
                        from matplotlib import rc_context
                        from matplotlib.mathtext import math_to_image

                        buffer = BytesIO()
                        with rc_context({"mathtext.fontset": font}):
                            math_to_image(
                                "$" + latex + "$",
                                buffer,
                                dpi=rng.choice((110, 150, 180)),
                                format="png",
                            )
                        buffer.seek(0)
                        with Image.open(buffer) as im:
                            base = im.convert("RGB")
                except (
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                    ValueError,
                    RuntimeError,
                ):
                    failures += 1
                    continue
                labels.add(latex)
                group_counts[signature] = group_counts.get(signature, 0) + 1
                group = hashlib.sha256(signature.encode()).hexdigest()
                for variant in range(
                    min(variants, quotas[split] - len(records[split]))
                ):
                    sample_rng = random.Random(rng.randrange(2**63))
                    image, profile = vary_capture(base, sample_rng)
                    # Variable image geometry; cap only very large crops, without stretching.
                    image.thumbnail(
                        (image_size * 2, image_size), Image.Resampling.LANCZOS
                    )
                    index = len(records[split])
                    sample_id = f"{split}-{index:06d}"
                    path = out / split / "images" / f"{sample_id}.png"
                    image.save(path)
                    records[split].append(
                        {
                            "id": sample_id,
                            "image": f"images/{sample_id}.png",
                            "latex": latex,
                            "group_id": group,
                            "structure": signature,
                            "variant": variant,
                            "font": font,
                            "capture": profile,
                            "renderer": renderer,
                            "pixel_sha256": hashlib.sha256(
                                image.tobytes() + str(image.size).encode()
                            ).hexdigest(),
                        }
                    )
                    progress.update(1)
    unique = audit_records(records)
    train_chars = set("".join(row["latex"] for row in records["train"]))
    coverage = {
        s: sorted(set("".join(row["latex"] for row in records[s])) - train_chars)
        for s in ("val", "test")
    }
    if any(coverage.values()):
        raise ValueError(
            f"Evaluation contains characters absent from train: {coverage}; increase num or use another seed"
        )
    for split, rows in records.items():
        (out / split / "annotations.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
    manifest = {
        "schema_version": 2,
        "seed": seed,
        "renderer": renderer,
        "image_count": num,
        "variants_per_formula": variants,
        "counts": {s: len(v) for s, v in records.items()},
        "unique": unique,
        "render_failures": failures,
        "unseen_target_characters": coverage,
        "split_policy": "sha256 of value-independent expression tree; 80/10/10 hash buckets, exact image quotas; all sibling variants and same structures stay together",
        "fonts": STYLE_FONTS
        if renderer == "tex"
        else {"train": ["cm"], "val": ["stix"], "test": ["dejavusans"]},
        "evaluation_scope": "synthetic structural and font holdout, not real-photo accuracy",
        "complete": True,
    }
    manifest.update(
        {
            "generator_algorithm_version": "2.0",
            "max_target_tokens": 256,
            "depth_range": [2, 4],
            "max_formulas_per_structure": 8,
            "max_image_size": [image_size * 2, image_size],
            "capture_profiles": ["clean", "scan", "camera"],
        }
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "README.md").write_text(
        "# Synthetic formula dataset v2\n\nPhysical splits: train/, val/, test/. Each contains images/ and annotations.jsonl.\nImage paths are relative to that split's annotation file. manifest.json records generation and audit.\nValidation and test hold out expression-tree groups and fonts. They are still synthetic; use independent real photographs for real-world validation.\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=5000)
    parser.add_argument("--out", default="data/formula_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--renderer", choices=("tex", "mathtext"), default="tex")
    parser.add_argument("--variants", type=int, default=2)
    args = parser.parse_args()
    generate(
        args.num, args.out, args.seed, args.image_size, args.renderer, args.variants
    )


if __name__ == "__main__":
    main()
