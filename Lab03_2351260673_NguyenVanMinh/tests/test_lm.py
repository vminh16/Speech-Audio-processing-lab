import math

import pytest

from src.config import BOS
from src.lm import BigramLM, parse_lexicon, read_corpus, rerank

VOCAB = ["a", "b", "c"]
SENTENCES = [["a", "b"], ["a", "c"], ["b", "c"]]


def test_parse_lexicon(tmp_path):
    p = tmp_path / "lex.txt"
    p.write_text("# comment\nmo  m o\ndong d o ng  # inline\n\n", encoding="utf-8")
    assert parse_lexicon(p) == {"mo": ["m", "o"], "dong": ["d", "o", "ng"]}


@pytest.mark.parametrize("text", ["mo\n", "mo m o\nmo m o\n"])
def test_parse_lexicon_rejects_bad_lines(tmp_path, text):
    p = tmp_path / "lex.txt"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        parse_lexicon(p)


def test_read_corpus_skips_comments(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("# header\na b\n\nc  # tail\n", encoding="utf-8")
    assert read_corpus(p) == [["a", "b"], ["c"]]


def test_bigram_rows_normalize_to_one():
    lm = BigramLM(SENTENCES, VOCAB, k=0.5)
    for prev in [BOS, *VOCAB]:
        assert sum(lm.prob(prev, w) for w in VOCAB) == pytest.approx(1.0)


def test_bigram_rejects_unknown_word():
    with pytest.raises(ValueError):
        BigramLM([["a", "zzz"]], VOCAB)


def test_bigram_value_and_predecessors():
    lm = BigramLM(SENTENCES, VOCAB, k=0.5)
    # C(<s>, a) = 2, C(<s>) = 3, V = 3  ->  (2 + .5) / (3 + 1.5)
    assert lm.prob(BOS, "a") == pytest.approx(2.5 / 4.5)
    assert lm.predecessors("c") == ["a", "b"]


def test_rerank_lambda_extremes():
    lm = BigramLM(SENTENCES, VOCAB, k=0.5)
    am = {"a": -1.0, "b": -1.1, "c": -5.0}
    assert rerank(am, BOS, lm, 0.0)[0] == "a"  # lambda = 0 -> chỉ còn âm học
    am_close = {"a": -1.00, "b": -1.01, "c": -1.02}
    # sau "b" chỉ có "c" xuất hiện trong corpus -> LM lớn kéo kết quả về "c"
    assert rerank(am_close, "b", lm, 5.0)[0] == "c"
    assert math.isfinite(rerank(am, "a", lm, 1.0)[1]["c"])
