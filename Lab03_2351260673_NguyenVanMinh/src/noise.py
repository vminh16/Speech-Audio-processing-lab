"""Thêm nhiễu trắng Gaussian theo SNR cho thí nghiệm độ bền vững (E5, E6)."""
from __future__ import annotations

import numpy as np


def add_white_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator, active_db: float = 30.0) -> np.ndarray:
    """Cộng nhiễu trắng sao cho SNR = ``snr_db`` tính trên phần có tiếng nói.

    Công suất tín hiệu được đo trên các mẫu có biên độ trên ngưỡng ``active_db`` dB dưới đỉnh,
    để khoảng lặng ở hai đầu file không làm SNR bị đánh giá thấp.
    """
    y = np.asarray(y, dtype=np.float64)
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak == 0.0:
        return y
    active = y[np.abs(y) >= peak * 10 ** (-active_db / 20.0)]
    p_signal = float(np.mean(active**2))
    p_noise = p_signal / (10 ** (snr_db / 10.0))
    return y + rng.normal(0.0, np.sqrt(p_noise), size=y.shape)
