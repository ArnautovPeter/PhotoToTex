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
    manifest = generate(num=120, out=tmp_path, seed=17, image_size=192, renderer="mathtext", variants=2)
    splits = {}
    for split in ("train", "val", "test"):
        annotation = tmp_path / split / "annotations.jsonl"
        splits[split] = [json.loads(line) for line in annotation.read_text().splitlines()]
        assert len(splits[split]) == manifest["counts"][split]
        for row in splits[split]:
            assert (annotation.parent / row["image"]).is_file()
    for field in ("latex", "group_id", "pixel_sha256", "font"):
        values = [{row[field] for row in rows} for rows in splits.values()]
        assert all(values[i].isdisjoint(values[j]) for i in range(3) for j in range(i+1, 3))
    assert manifest["complete"]
    assert not any(manifest["unseen_target_characters"].values())
    from src.data.audit import audit_dataset
    assert audit_dataset(tmp_path)["verified_images"] == 120
    from src.train import fingerprint_dataset
    assert len(fingerprint_dataset(tmp_path)) == 64
    from src.data.dataset import split_annotation
    tokenizer = LatexTokenizer(char for row in splits["train"] for char in row["latex"])
    for split in splits:
        dataset = LatexImageDataset(split_annotation(tmp_path, split), tokenizer, (128,384), 256)
        assert dataset[0]["image"].shape == (3,128,384)
    import pytest
    with pytest.raises(FileExistsError):
        generate(120, tmp_path, renderer="mathtext")
    # A changed or corrupted image must be detected before training.
    changed = tmp_path / "train" / splits["train"][0]["image"]
    Image.new("RGB", (20, 20), "black").save(changed)
    with pytest.raises(ValueError, match="checksum mismatch"):
        audit_dataset(tmp_path)


def test_structural_split_and_audit():
    from src.data.generate_synthetic import assign_split, audit_records, composed_formula
    import random
    import pytest
    assert assign_split("frac(atom,atom)", 42) == assign_split("frac(atom,atom)", 42)
    first = [composed_formula(random.Random(seed), 3) for seed in range(20)]
    second = [composed_formula(random.Random(seed), 3) for seed in range(20)]
    assert first == second
    row = {"group_id":"same", "latex":"x", "pixel_sha256":"pixels"}
    with pytest.raises(ValueError, match="Cross-split leakage"):
        audit_records({"train":[row], "val":[row]})


def test_split_resolver_prefers_new_layout_and_rejects_incomplete(tmp_path):
    from src.data.dataset import split_annotation
    import pytest
    old = tmp_path / "train.jsonl"
    old.write_text("")
    assert split_annotation(tmp_path, "train") == old
    modern = tmp_path / "train" / "annotations.jsonl"
    modern.parent.mkdir()
    modern.write_text("")
    assert split_annotation(tmp_path, "train") == modern
    (tmp_path / "manifest.json").write_text('{"complete": false}')
    with pytest.raises(ValueError, match="incomplete"):
        split_annotation(tmp_path, "train")


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
