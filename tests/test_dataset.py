import json
import re

from PIL import Image

from src.data.dataset import LatexImageDataset, augment_photo, preprocess_image
from src.data.generate_synthetic import generate
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


def test_generated_splits_have_unique_varied_examples(tmp_path):
    generate(num=30, out=tmp_path, seed=17, image_size=96)
    splits = {}
    for split in ("train", "val", "test"):
        annotation = tmp_path / f"{split}.jsonl"
        splits[split] = [json.loads(line) for line in annotation.read_text().splitlines()]

    latex_sets = [{row["latex"] for row in rows} for rows in splits.values()]
    assert all(latex_sets[i].isdisjoint(latex_sets[j]) for i in range(3) for j in range(i + 1, 3))
    all_latex = [row["latex"] for rows in splits.values() for row in rows]
    assert len(all_latex) == len(set(all_latex)) == 30
    formulas = {re.search(r"\$(.*?)\$", text).group(1) for text in all_latex}
    assert len(formulas) == 30
    assert all(text.startswith("$") and text.endswith("$") for text in all_latex)
    assert len({re.search(r"\\[A-Za-z]+", text).group(0) for text in all_latex if re.search(r"\\[A-Za-z]+", text)}) > 5
    with Image.open(tmp_path / splits["train"][0]["image"]) as image:
        assert image.size == (96, 64)


def test_preprocessing_preserves_aspect_ratio_and_pads():
    source = Image.new("RGB", (200, 50), "black")
    tensor = preprocess_image(source, (64, 192))

    assert tuple(tensor.shape) == (3, 64, 192)
    assert tensor[:, 0].min() == 1
    assert tensor[:, 32, 96].max() == 0


def test_photo_augmentation_preserves_image_dimensions():
    source = Image.new("RGB", (192, 64), (240, 240, 240))

    augmented = augment_photo(source)

    assert augmented.size == source.size
    assert augmented.mode == "RGB"


def test_dataset_rejects_silently_truncated_targets(tmp_path):
    image = tmp_path / "sample.png"
    Image.new("RGB", (16, 16), "white").save(image)
    annotation = tmp_path / "train.jsonl"
    annotation.write_text(json.dumps({"id": "long", "image": "sample.png", "latex": "x" * 20}) + "\n")
    dataset = LatexImageDataset(annotation, LatexTokenizer("x"), image_size=16, max_length=8)

    try:
        dataset[0]
    except ValueError as error:
        assert "exceeding max_length=8" in str(error)
    else:
        raise AssertionError("Overlong targets must not be silently truncated")
