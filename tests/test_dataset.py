import json

from PIL import Image

from src.data.dataset import LatexImageDataset
from src.data.tokenizer import LatexTokenizer


def test_dataset_loads(tmp_path):
    image = tmp_path / "sample.png"
    Image.new("RGB", (16, 16), "white").save(image)
    annotation = tmp_path / "train.jsonl"
    annotation.write_text(json.dumps({"id": "x", "image": "sample.png", "latex": "x^2"}) + "\n")
    dataset = LatexImageDataset(annotation, LatexTokenizer("x^2"), image_size=16)
    item = dataset[0]
    assert tuple(item["image"].shape) == (3, 16, 16)
    assert item["latex"] == "x^2"
