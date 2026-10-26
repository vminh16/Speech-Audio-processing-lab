"""Lexicon, mô hình ngôn ngữ bigram (làm trơn Add-k) và xếp hạng lại MAP."""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from .config import BOS


def parse_lexicon(path: Path) -> dict[str, list[str]]:
    """Đọc ``WORD PHONE_1 ... PHONE_K`` (mỗi dòng một từ; ``#`` là chú thích) thành ``{word: [phones]}``."""
    lexicon: dict[str, list[str]] = {}
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        word, *phones = line.split()
        if not phones:
            raise ValueError(f"{path}:{lineno}: từ '{word}' không có âm vị nào.")
        if word in lexicon:
            raise ValueError(f"{path}:{lineno}: từ '{word}' bị khai báo trùng.")
        lexicon[word] = phones
    return lexicon


def read_corpus(path: Path) -> list[list[str]]:
    """Mỗi dòng là một câu lệnh gồm các từ cách nhau bởi khoảng trắng."""
    sentences = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        words = raw.split("#", 1)[0].split()
        if words:
            sentences.append(words)
    return sentences


def count_ngrams(sentences: list[list[str]]) -> tuple[Counter, Counter, Counter]:
    """``(unigram, history, bigram)``.

    ``history[w]`` là số lần ``w`` đứng trước một từ khác (kể cả ``<s>``). Dùng làm mẫu số bigram để
    phân phối ``P(. | w)`` chuẩn hóa về 1 ngay cả khi ``w`` là từ cuối câu.
    """
    unigram: Counter = Counter()
    history: Counter = Counter()
    bigram: Counter = Counter()
    for sent in sentences:
        unigram.update(sent)
        padded = [BOS, *sent]
        for prev, cur in zip(padded[:-1], padded[1:]):
            history[prev] += 1
            bigram[(prev, cur)] += 1
    return unigram, history, bigram


class BigramLM:
    """P(cur | prev) = (C(prev, cur) + k) / (C(prev) + k * V)."""

    def __init__(self, sentences: list[list[str]], vocab: list[str], k: float = 0.5) -> None:
        self.vocab = list(vocab)
        self.k = k
        self.unigram, self.history, self.bigram = count_ngrams(sentences)
        unknown = set(self.unigram) - set(self.vocab)
        if unknown:
            raise ValueError(f"Corpus chứa từ ngoài từ vựng: {sorted(unknown)}")

    def prob(self, prev: str, cur: str) -> float:
        num = self.bigram[(prev, cur)] + self.k
        den = self.history[prev] + self.k * len(self.vocab)
        return num / den

    def logprob(self, prev: str, cur: str) -> float:
        return math.log(self.prob(prev, cur))

    def predecessors(self, word: str) -> list[str]:
        """Các từ (hoặc ``<s>``) đã xuất hiện ngay trước ``word`` trong corpus."""
        return sorted({p for (p, c) in self.bigram if c == word})

    def table(self) -> list[list[float]]:
        """Ma trận P(cur | prev): hàng = [<s>] + vocab, cột = vocab."""
        return [[self.prob(p, c) for c in self.vocab] for p in [BOS, *self.vocab]]


def rerank(per_frame_am: dict[str, float], prev: str, lm: BigramLM, lam: float) -> tuple[str, dict[str, float]]:
    """Score(w) = (1/T) log P_AM(X|w) + lam * log P_LM(w | prev); trả về (từ tốt nhất, điểm từng từ)."""
    scores = {w: s + lam * lm.logprob(prev, w) for w, s in per_frame_am.items()}
    return max(scores, key=scores.get), scores
