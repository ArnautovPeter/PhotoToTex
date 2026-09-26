# Photo-to-LaTeX baseline

This repository contains an offline, CPU-friendly baseline for recognizing a
cropped formula image as LaTeX.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Generate data

The template backend is deterministic and needs no network or API:

```bash
python -m src.data.generate_synthetic --num 100 --out data/synthetic --seed 42
```

Each split is JSONL with `id`, relative `image`, and `latex` fields. The
optional `--llm-model` flag is reserved for a future local Hugging Face backend;
the baseline deliberately falls back to templates.

## Train, infer, and evaluate

```bash
python -m src.train --config configs/baseline.yaml
python -m src.infer --checkpoint checkpoints/baseline.pt --image data/synthetic/images/000000.png
python -m src.evaluate --checkpoint checkpoints/baseline.pt --split test
```

The default config uses one epoch and a small model. Set `epochs`, image size,
and model dimensions in `configs/baseline.yaml` for a larger experiment.

## Tests

```bash
pytest -q
```

## Limitations

This is synthetic-only training on cropped formula images, with a small
character-level model and greedy decoding. It is not full document OCR and
does not yet detect text/formula regions. The detection interface is an
explicit optional stub in `src/pipeline/detect_and_recognize.py`.
