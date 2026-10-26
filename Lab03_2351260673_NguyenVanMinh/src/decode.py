"""Giải mã: chấm điểm Forward (Maximum Likelihood) và gióng hàng Viterbi."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import GaussianHMM

from .acoustic import Models


@dataclass
class Recognition:
    """Kết quả nhận dạng một utterance bằng ML."""

    scores: dict[str, float]  # log P(X | lambda_w)
    n_frames: int

    @property
    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)

    @property
    def pred(self) -> str:
        return self.ranked[0][0]

    @property
    def margin(self) -> float:
        """Top-1 trừ Top-2 của log-likelihood."""
        r = self.ranked
        return r[0][1] - r[1][1] if len(r) > 1 else float("inf")

    @property
    def per_frame(self) -> dict[str, float]:
        return {w: s / self.n_frames for w, s in self.scores.items()}


def score_all(feat: np.ndarray, models: Models) -> dict[str, float]:
    """log P(X | lambda_w) bằng thuật toán Forward cho mọi từ."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {w: float(m.score(feat)) for w, m in models.items()}


def recognize(feat: np.ndarray, models: Models) -> Recognition:
    return Recognition(score_all(feat, models), len(feat))


def viterbi_path(feat: np.ndarray, model: GaussianHMM) -> tuple[float, np.ndarray]:
    """``(log P(X, Q* | lambda), Q*)`` với Q* là chuỗi trạng thái tối ưu theo Viterbi."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        logprob, states = model.decode(feat, algorithm="viterbi")
    return float(logprob), np.asarray(states)


def state_durations(states: np.ndarray, n_states: int) -> np.ndarray:
    """Số frame dừng ở mỗi trạng thái theo đường Viterbi."""
    return np.bincount(states, minlength=n_states)
