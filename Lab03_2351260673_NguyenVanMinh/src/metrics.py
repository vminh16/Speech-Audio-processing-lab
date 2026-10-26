"""Chỉ số đánh giá: accuracy, ma trận nhầm lẫn, cặp từ dễ nhầm."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if y_true else float("nan")


def confusion(y_true: list[str], y_pred: list[str], labels: tuple[str, ...]) -> np.ndarray:
    """Hàng = nhãn thực, cột = nhãn dự đoán."""
    return confusion_matrix(y_true, y_pred, labels=list(labels))


def most_confused(cm: np.ndarray, labels: tuple[str, ...], top: int = 3) -> list[tuple[str, str, int]]:
    """Các cặp (thực, đoán) sai nhiều nhất; bỏ qua đường chéo và các ô bằng 0."""
    pairs = [(labels[i], labels[j], int(cm[i, j])) for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j] > 0]
    return sorted(pairs, key=lambda p: p[2], reverse=True)[:top]


def per_class_accuracy(cm: np.ndarray, labels: tuple[str, ...]) -> dict[str, float]:
    totals = cm.sum(axis=1)
    return {l: (float(cm[i, i] / totals[i]) if totals[i] else float("nan")) for i, l in enumerate(labels)}
