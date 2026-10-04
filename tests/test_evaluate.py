import json
import sys

import torch
from PIL import Image

from src.data.tokenizer import LatexTokenizer
from src.evaluate import main
from src.models.seq2seq import ImageToLatex


def test_evaluate_external_real_annotations_records_unknown_characters(tmp_path, monkeypatch):
    tokenizer = LatexTokenizer("x")
    config = {
        "device": "cpu",
        "data_dir": str(tmp_path),
        "image_size": [32, 96],
        "max_length": 8,
        "batch_size": 1,
        "d_model": 16,
        "nhead": 4,
        "layers": 1,
        "ff_dim": 32,
    }
    model = ImageToLatex(len(tokenizer), 16, 4, 1, 32, 8)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "config": config,
            "tokenizer": tokenizer.to_dict(),
            "model": model.state_dict(),
            "preprocessing": "aspect_pad_v1",
        },
        checkpoint_path,
    )
    images = tmp_path / "real" / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (120, 40), "white").save(images / "formula.png")
    annotations = tmp_path / "real" / "val.jsonl"
    annotations.write_text(
        json.dumps({"id": "real-1", "image": "images/formula.png", "latex": "xy"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint_path),
            "--annotations",
            str(annotations),
            "--name",
            "real_val",
        ],
    )

    main()

    metrics = json.loads(checkpoint_path.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["splits"]["real_val"]["count"] == 1
    assert metrics["splits"]["real_val"]["unknown_char_rate"] == 0.5
