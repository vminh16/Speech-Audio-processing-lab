"""Baseline DTW (Lab 2): nhận dạng 1-NN theo template."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def dtw_distance(a: np.ndarray, b: np.ndarray, return_path: bool = False):
    """Khoảng cách DTW giữa hai chuỗi đặc trưng (Euclid giữa frame), chuẩn hóa theo ``len(a) + len(b)``.

    Bước đi đối xứng: (i-1, j), (i, j-1), (i-1, j-1).
    """
    cost = cdist(a, b, metric="euclidean")
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        c = cost[i - 1]
        # Bước (i-1, j) và (i-1, j-1) vector hóa được; bước (i, j-1) là một phép quét tích lũy:
        # row[j] = min_k<=j (tmp[k] + c[k+1..j]) = cs[j] + min_k<=j (tmp[k] - cs[k]).
        tmp = c + np.minimum(acc[i - 1, 1:], acc[i - 1, :-1])
        cs = np.cumsum(c)
        acc[i, 1:] = cs + np.minimum.accumulate(tmp - cs)
    dist = float(acc[n, m] / (n + m))
    if not return_path:
        return dist

    path = [(n - 1, m - 1)]
    i, j = n, m
    while (i, j) != (1, 1):
        steps = [(acc[i - 1, j - 1], i - 1, j - 1), (acc[i - 1, j], i - 1, j), (acc[i, j - 1], i, j - 1)]
        _, i, j = min(steps, key=lambda s: s[0])
        path.append((i - 1, j - 1))
    return dist, path[::-1]


class DTWRecognizer:
    """Lưu các template theo lớp; dự đoán lớp có template gần nhất."""

    def __init__(self) -> None:
        self.templates: dict[str, list[np.ndarray]] = {}

    def fit(self, groups: dict[str, list[np.ndarray]]) -> "DTWRecognizer":
        self.templates = {label: list(seqs) for label, seqs in groups.items()}
        return self

    def scores(self, feat: np.ndarray) -> dict[str, float]:
        """Khoảng cách nhỏ nhất tới template của mỗi lớp (càng nhỏ càng giống)."""
        return {label: min(dtw_distance(feat, t) for t in seqs) for label, seqs in self.templates.items()}

    def predict(self, feat: np.ndarray) -> tuple[str, dict[str, float]]:
        s = self.scores(feat)
        return min(s, key=s.get), s
