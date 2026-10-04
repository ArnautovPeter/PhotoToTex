import torch

from src.models.seq2seq import ImageToLatex


def test_model_forward_and_decode():
    model = ImageToLatex(vocab_size=12, d_model=16, nhead=4, layers=1, ff_dim=32, max_length=8)
    images = torch.randn(2, 3, 128, 384)
    target = torch.randint(0, 12, (2, 5))
    assert model(images, target).shape == (2, 5, 12)
    assert model.greedy_decode(images, 1, 2, max_length=4).shape[1] <= 4
