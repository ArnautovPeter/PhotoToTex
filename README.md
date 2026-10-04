# Photo-to-LaTeX baseline

This repository contains an offline baseline for recognizing a cropped,
typeset formula image as LaTeX. It is an experimental recognizer, not a
ready-made OCR system for arbitrary photographs or handwriting.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Generate data

The generator is deterministic for a given seed and needs no network or API.
It generates standalone formulas from multiple structures, renders them with
varied math fonts and paper-like image effects, and puts unique formulas into
train, validation, and test splits:

```bash
python -m src.data.generate_synthetic --num 5000 --out data/formula_synthetic --seed 42
```

Each split is JSONL with `id`, relative `image`, and `latex` fields. The
generated images are wide crops, not full documents. Synthetic font and
photometric variation help with robustness, but do not substitute for labeled
real photographs or handwritten examples.

## Train, infer, and evaluate

Generate the training data first; the default config expects it under
`data/formula_synthetic`:

```bash
python -m src.data.generate_synthetic --num 5000 --out data/formula_synthetic --seed 42
python -m src.train --config configs/baseline.yaml --checkpoint checkpoints/runs/formula-v2/best.pt
python -m src.infer --checkpoint checkpoints/runs/formula-v2/best.pt --image data/formula_synthetic/images/000000.png
python -m src.evaluate --checkpoint checkpoints/runs/formula-v2/best.pt --split test
```

The model now preserves image aspect ratio on a 128x384 canvas and adds 2D
spatial position information to its CNN features. Training reports validation
character error rate (CER) and exact match every epoch, saves the best
validation checkpoint, reduces the learning rate when validation stalls, and
stops early rather than selecting by training loss. Existing checkpoints were
trained with the old 64x64 preprocessing and small dataset; regenerate the
data and retrain before using the updated interface.

Training is configured to require CUDA and will stop with an error rather than
silently falling back to CPU. To select a particular GPU, pass for example
`--device cuda:0`. The installed PyTorch CUDA runtime must be compatible with
the NVIDIA driver. On Linux, check `nvidia-smi` and `torch.version.cuda`; for
example, with a CUDA 12.4-capable driver, reinstall a matching PyTorch build
inside the active virtual environment:

```bash
python -m pip uninstall -y torch torchvision
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

Only start training when that last command reports `True` and the GPU name.
The default config uses four CPU data-loader workers to prepare camera-like
augmentations in parallel while CUDA trains. Set `num_workers: 0` in the
config if your platform does not support worker processes.

### Continue from an existing checkpoint

Use a warm start to keep only the learned weights and begin a new optimization
run. The tokenizer and model settings (including image size) must exactly
match the current run. This is appropriate for fine-tuning on new images only
when their LaTeX vocabulary is the same:

```bash
python -m src.train --config configs/baseline.yaml \
  --warm-start checkpoints/baseline.pt \
  --checkpoint checkpoints/runs/formula-warm/best.pt
```

Warm start resets the optimizer, scheduler, epoch count, and validation
early-stopping state. By contrast, exact resume continues from the last
completed epoch and restores optimizer, scheduler, RNG, and data-shuffle
states. Resume from the `.last.pt` file, not the best-validation checkpoint:

```bash
python -m src.train --config configs/baseline.yaml \
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

## Local web interface

Start the local interface after installing the dependencies and making sure
the checkpoint exists:

```bash
python -m src.web_app
```

Then open <http://127.0.0.1:8000> in a browser, choose a cropped image of one
formula, and click **Распознать формулу**. The result can be copied or
downloaded as `formula.tex` (the file contains only the formula fragment, not
a standalone document; PDF compilation is not included yet). To use another
checkpoint or port:

```bash
python -m src.web_app --checkpoint path/to/checkpoint.pt --port 8001
```

The **Модель и результаты** tab shows model settings, validation/test metrics,
and the train-loss/validation-CER curves. Compute the test results for the
selected checkpoint once with:

```bash
python -m src.evaluate --checkpoint checkpoints/baseline.pt --split test
```

New training runs write their per-epoch history to
`checkpoints/baseline.history.json`; evaluation stores results in
`checkpoints/baseline.metrics.json`. Restart the web server after replacing a
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
python -m src.evaluate --checkpoint checkpoints/runs/formula-v2/best.pt \
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
pytest -q
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
