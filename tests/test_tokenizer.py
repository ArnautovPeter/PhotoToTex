from src.data.tokenizer import LatexTokenizer


def test_round_trip():
    tokenizer = LatexTokenizer("x^2+")
    encoded = tokenizer.encode("x^2+")
    assert tokenizer.decode(encoded) == "x^2+"
    assert tokenizer.pad_id == 0
