# PhotoToTex · Formula workbench

![PhotoToTex workbench](assets/workbench.jpg)

## Learning notes and architecture review

See [the architecture review](docs/architecture_review.md) for a Russian-language
explanation of the current model, its limitations, pretrained-model candidates,
and the proposed experiment plan. [The learning log](docs/learning_log.md)
records decisions and distinguishes proposed work from completed experiments.
See [the web and career stack review](docs/web_and_career_stack.md) for the
proposed API architecture and a staged technology plan for junior DS/ML/DE roles.

A local FastAPI workbench for recognizing cropped formulas with pretrained
pix2tex and comparing it with a small educational CNN/Transformer baseline.
Inspect the rendered formula, edit LaTeX and export the result. Full-page
layout detection is not included. Recognition quality must be checked on your
own photographs; neither backend guarantees handwriting support.

## Install and run with uv

Python 3.11 and dependencies are managed by `uv`; activation and pip are not
needed. Choose the CPU or CUDA 12.4 extra consistently when syncing and running:

```bash
uv sync --frozen --extra cpu
uv run --frozen --extra cpu python -m src.download_models
uv run --frozen --extra cpu python -m src.web_app
```

Open <http://127.0.0.1:8000>. Official pix2tex weights are downloaded once into
`checkpoints/pix2tex`; recognition then runs locally. The frontend includes
local KaTeX assets and needs no CDN. Upload one cropped formula, select pix2tex
or the retained baseline, inspect/edit the rendered result and copy/download
LaTeX. `/docs` documents the API. Baseline metrics are labelled separately.

For the existing NVIDIA setup, use `--extra cu124` instead of `--extra cpu`;
add `--device cuda:0` to the web CLI when a working GPU is available. These extras
are mutually exclusive. `uv` still creates a project `.venv`, but manages it
from `pyproject.toml` and `uv.lock`. Commit both files; add dependencies with
`uv add`, then update the lock deliberately rather than installing with pip.

```bash
uv run --frozen --extra cpu pytest -q
```

Ruff checks Python code and import ordering using the project configuration:

```bash
uv run --frozen --extra cpu ruff check .
```

With Ruff installed on your PATH (`uv tool install ruff==0.17.2`), you can also
run `ruff check .`, `ruff check . --fix`, or `ruff format .` directly. Run from
the project root to check the whole project, or pass a file/directory to limit
the scope.

Docker uses the same lock and the CPU extra:

```bash
docker compose up --build
```

Run the weight-download command on the host first; Compose mounts `checkpoints`
and `data`. Models and datasets are excluded from the Docker image and Git.

## Generate data

The v2 generator composes nested expressions, renders with real TeX, and creates
physical `train/`, `val/`, and `test/` folders. Each contains its own `images/`
and `annotations.jsonl`; image paths are relative to that split's annotation.
All variants and matching expression-tree structures stay in one split.
Validation/test also use held-out fonts. This measures a combined structural
and style shift, not real-photo accuracy.

```bash
uv run --frozen --extra cpu python -m src.data.generate_synthetic --num 5000 --out data/formula_v2 --seed 42
uv run --frozen --extra cpu python -m src.data.audit data/formula_v2
```

TeX rendering requires system `pdflatex`, `standalone`, `amsmath`, `amssymb`,
`newtxmath`, `mathpazo`, `eulervm`, and `pdftoppm`. There is no silent fallback;
`--renderer mathtext` explicitly selects a smaller grammar for lightweight tests.
Output must be empty to avoid mixing dataset versions. Generation writes a
manifest and marks interrupted datasets incomplete. `--num` counts images;
by default each formula has up to two capture variants.

See [the data protocol](docs/data_pipeline.md) for split rules, limitations,
reproduction and independent real validation. Existing old datasets remain
supported by the loader for old experiments.

## Train, infer, and evaluate

The prepared v2 dataset is under `data/formula_v2`. Audit it before training;
if generating a replacement, use a new empty output directory. Training defaults
to `configs/data_v2.yaml`:

```bash
uv run --frozen --extra cpu python -m src.data.audit data/formula_v2
uv run --frozen --extra cu124 python -m src.train --config configs/data_v2.yaml --checkpoint checkpoints/runs/formula-v2/best.pt
uv run --frozen --extra cpu python -m src.infer --checkpoint checkpoints/runs/formula-v2/best.pt --image data/formula_v2/train/images/train-000000.png
uv run --frozen --extra cpu python -m src.evaluate --checkpoint checkpoints/runs/formula-v2/best.pt --split test
```

The model now preserves image aspect ratio on a 128x384 canvas and adds 2D
spatial position information to its CNN features. Training reports validation
character error rate (CER) and exact match every epoch, saves the best
validation checkpoint, reduces the learning rate when validation stalls, and
stops early rather than selecting by training loss. Retained checkpoints use the earlier synthetic dataset and a 128x384 canvas.
They remain useful for comparison, but are not compatible with the v2 tokenizer
and sequence limit. Start a fresh run for the new dataset.

Training is configured to require CUDA and will stop with an error rather than
silently falling back to CPU. To select a particular GPU, pass for example
`--device cuda:0`. The installed PyTorch CUDA runtime must be compatible with
the NVIDIA driver. On Linux, check `nvidia-smi` and `torch.version.cuda`; for
example, with a CUDA 12.4-capable driver, reinstall a matching PyTorch build
through the CUDA extra:

```bash
uv sync --frozen --extra cu124
uv run --frozen --extra cu124 python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Only start training when that last command reports `True`.
The default config uses four CPU data-loader workers to prepare camera-like
augmentations in parallel while CUDA trains. Set `num_workers: 0` in the
config if your platform does not support worker processes.

### Continue from an existing checkpoint

Use a warm start to keep only the learned weights and begin a new optimization
run. The tokenizer and model settings (including image size) must exactly
match the current run. This is appropriate for fine-tuning on new images only
when their LaTeX vocabulary is the same. The example below uses the retained
legacy dataset and config, not the new v2 dataset:

```bash
uv run --frozen --extra cu124 python -m src.train --config configs/baseline.yaml \
  --warm-start checkpoints/baseline.pt \
  --checkpoint checkpoints/runs/formula-warm/best.pt
```

Warm start resets the optimizer, scheduler, epoch count, and validation
early-stopping state. By contrast, exact resume continues from the last
completed epoch and restores optimizer, scheduler, RNG, and data-shuffle
states. Resume from the `.last.pt` file, not the best-validation checkpoint:

```bash
uv run --frozen --extra cu124 python -m src.train --config configs/baseline.yaml \
  --resume checkpoints/runs/formula-warm/best.last.pt \
  --checkpoint checkpoints/runs/formula-warm/best.pt \
  --epochs 120
```

The configured `epochs` is the total epoch limit. Resume requires the same
dataset, tokenizer, image preprocessing, batch size, and optimizer/scheduler
settings, including `num_workers`; only the epoch limit and output checkpoint
path may change. Each run must use its own `--checkpoint` path; existing
outputs are never overwritten implicitly.

Legacy checkpoints that used square-stretch preprocessing are rejected for
warm start by default. `--allow-legacy-preprocessing` opts in explicitly,
but the image geometry mismatch can hurt results. Changing the tokenizer or
architecture is not a warm start: train a fresh model instead.

## Local web interface and baseline history

Start the local interface after installing the dependencies and making sure
the checkpoint exists:

```bash
uv run --frozen --extra cpu python -m src.web_app
```

Then open <http://127.0.0.1:8000> in a browser, choose a cropped image of one
formula, and click **Распознать формулу**. The result can be copied or
downloaded as `formula.tex` (the file contains only the formula fragment, not
a standalone document; PDF compilation is not included yet). To use another
checkpoint or port:

```bash
uv run --frozen --extra cpu python -m src.web_app --checkpoint path/to/checkpoint.pt --port 8001
```

The **Модель и результаты** tab shows model settings, validation/test metrics,
and the train-loss/validation-CER curves. Compute the test results for the
selected checkpoint once with:

```bash
uv run --frozen --extra cpu python -m src.evaluate --checkpoint checkpoints/baseline.pt --split test
```

Each training run writes history beside its selected checkpoint as
`best.history.json`; evaluation writes `best.metrics.json`. For the retained
default UI checkpoint these paths are `baseline.history.json` and
`baseline.metrics.json`. Restart the web server after replacing a
checkpoint so prediction uses the new weights. The results tab refreshes its
metrics and training history while open.

To evaluate labeled real photographs without mixing them into synthetic
training data, put a JSONL file and its images together, for example:

```json
{"id":"photo-001","image":"images/photo-001.jpg","latex":"\\frac{1}{2}+x"}
```

Run evaluation with an explicit metrics name; image paths are relative to the
JSONL file:

```bash
uv run --frozen --extra cpu python -m src.evaluate --checkpoint checkpoints/runs/formula-v2/best.pt \
  --annotations data/real/val.jsonl --name real_val --device cuda
```

The dashboard will show this result alongside the synthetic scores, including
the fraction of target characters absent from the model's vocabulary. Keep
this real validation set separate from the images used for warm-start
fine-tuning; use a different labeled set for final testing.

The server listens only on this computer (`127.0.0.1`); images are not uploaded
to an external service. It accepts PNG, JPEG, WebP, and BMP images up to 10 MB.
This baseline recognizes a single formula in a crop, not formulas on a full
page. The interface returns LaTeX source; it does not compile TeX into a PDF.

## Tests

```bash
uv run --frozen --extra cpu pytest -q
```

## Limitations

The model is still a compact character-level CNN/Transformer trained from
scratch on synthetic, typeset formula crops, uses train-only perspective,
lighting, shadow and JPEG augmentation, and decodes greedily. It is
not full document OCR, does not detect formula regions, and should not be
expected to read arbitrary handwriting, camera photos, or formula styles
outside the training distribution. To improve those cases, collect labeled
real examples and fine-tune/evaluate on a held-out real-image set; increasing
the synthetic dataset alone cannot teach handwriting styles that it never
sees. The detection interface is an explicit optional stub in
`src/pipeline/detect_and_recognize.py`.

## Practical next improvements

Train-only perspective, illumination-gradient, shadow, color/noise, and JPEG
augmentations are enabled in the default config; validation and test images
remain unchanged. These are inexpensive to try, but the real-photo validation
command above is needed to tell whether they help the intended input domain.

For the biggest trustworthy improvement on your own camera photos, first
label a small representative real validation set (even 50–200 crops is useful
for measuring progress), then add separate labeled real training examples.
This is feasible now with the existing JSONL format and requires no model
rewrite. A real LaTeX renderer (`pdflatex` plus `pdftoppm`) is available in the
current development environment and is a reasonable next synthetic-data
upgrade, but requires those external tools on every machine that generates
data. Beam search is a relatively isolated inference improvement that can
also be compared on validation/test without retraining.

A pretrained vision encoder may improve recognition with limited data, but it
requires a different model/checkpoint and adds substantial dependencies; this
machine's RTX 3050 has 4 GB VRAM, so it is not the simplest safe next step.
Changing from character tokens to LaTeX tokens also changes the vocabulary and
output layers, so existing checkpoints cannot be warm-started across that
change. Full-page detection, handwriting, and camera-photo robustness still
need labeled examples and/or a separate localization model.
