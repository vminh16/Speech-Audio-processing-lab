"""Chọn ngưỡng từ chối (reject) cho từ ngoài từ vựng, chỉ dùng dữ liệu train (leave-one-out)."""
from __future__ import annotations

import numpy as np

from .acoustic import train_models
from .config import HMMConfig
from .decode import score_all


def loo_scores(groups: dict[str, list[np.ndarray]], cfg: HMMConfig) -> tuple[np.ndarray, np.ndarray]:
    """Điểm log-likelihood theo frame trong leave-one-out trên tập train.

    Với mỗi utterance, mô hình của lớp đúng được huấn luyện lại *không* có utterance đó; các mô hình lớp khác
    giữ nguyên. Nhờ vậy điểm không bị thiên lệch lạc quan như khi chấm lại chính dữ liệu đã học và không
    cần đụng tới tập test.

    Returns:
        (điểm lớp đúng, điểm các lớp sai), đều đã chia cho số frame.
    """
    full_models, _ = train_models(groups, cfg)
    pos: list[float] = []
    neg: list[float] = []
    for label, seqs in groups.items():
        if len(seqs) < 2:
            continue
        for i, seq in enumerate(seqs):
            held = {**groups, label: seqs[:i] + seqs[i + 1 :]}
            loo_models, _ = train_models({label: held[label]}, cfg)
            models = {**full_models, label: loo_models[label]}
            for word, ll in score_all(seq, models).items():
                (pos if word == label else neg).append(ll / len(seq))
    return np.array(pos), np.array(neg)


def youden_threshold(pos: np.ndarray, neg: np.ndarray) -> float:
    """Ngưỡng t (chấp nhận nếu điểm >= t) tối đa hóa TPR - FPR (chỉ số Youden J)."""
    candidates = np.unique(np.concatenate([pos, neg]))
    best_t, best_j = float(candidates[0]), -np.inf
    for t in candidates:
        j = np.mean(pos >= t) - np.mean(neg >= t)
        if j > best_j:
            best_t, best_j = float(t), float(j)
    return best_t
