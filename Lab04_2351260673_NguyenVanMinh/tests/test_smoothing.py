"""
Unit tests for boundary smoothing and cross-fade module (src/smoothing.py).
Following Test-Driven Development (TDD).
"""

import numpy as np
import pytest
from src.smoothing import (
    crossfade,
    crossfade_sequence,
    compute_spectral_discontinuity,
)


def test_crossfade_zero_overlap():
    """Khi n = 0, crossfade tương đương phép ghép nối np.concatenate."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    result = crossfade(a, b, n=0)
    expected = np.concatenate([a, b])
    np.testing.assert_allclose(result, expected)


def test_crossfade_overlap_length():
    """Độ dài sau crossfade phải bằng len(a) + len(b) - n."""
    a = np.ones(1000, dtype=np.float32)
    b = np.ones(800, dtype=np.float32)
    n = 240  # 15 ms ở 16kHz
    result = crossfade(a, b, n=n)
    assert len(result) == len(a) + len(b) - n


def test_crossfade_linear_blend():
    """Kiểm tra hàm dốc tuyến tính alpha trong vùng chồng lấn."""
    a = np.ones(100, dtype=np.float32)
    b = np.zeros(100, dtype=np.float32)
    n = 10
    result = crossfade(a, b, n=n)

    # Đoạn đầu từ 0 đến 90 phải toàn 1
    np.testing.assert_allclose(result[:90], 1.0)
    # Đoạn cuối từ 100 đến 190 phải toàn 0
    np.testing.assert_allclose(result[100:], 0.0)

    # Vùng chồng lấn từ index 90 đến 100: (1 - alpha) * 1 + alpha * 0 = 1 - alpha
    expected_overlap = 1.0 - np.linspace(0.0, 1.0, n, endpoint=True)
    np.testing.assert_allclose(result[90:100], expected_overlap, atol=1e-5)


def test_crossfade_sequence():
    """Kiểm tra ghép chuỗi 3 units và tính toán mốc thời gian biên nối."""
    sr = 16000
    u1 = np.ones(1600, dtype=np.float32)  # 0.1s
    u2 = np.ones(1600, dtype=np.float32)  # 0.1s
    u3 = np.ones(1600, dtype=np.float32)  # 0.1s

    # xfade_ms = 10ms -> 160 samples
    y_out, boundaries = crossfade_sequence([u1, u2, u3], sr=sr, xfade_ms=10.0)

    expected_len = 1600 * 3 - 2 * 160
    assert len(y_out) == expected_len
    # Phải có 2 biên nối cho 3 units
    assert len(boundaries) == 2
    # Biên 1 xấp xỉ quanh 0.1s
    assert np.isclose(boundaries[0], (1600 - 80) / sr, atol=0.01)


def test_compute_spectral_discontinuity():
    """Kiểm tra chỉ số gián đoạn phổ D_spec."""
    sr = 16000
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
    sine_a = np.sin(2 * np.pi * 400 * t).astype(np.float32)

    # 1. Tín hiệu giống hệt nhau -> D_spec phải xấp xỉ 0
    d_spec_same = compute_spectral_discontinuity(sine_a, sine_a, sr=sr)
    assert d_spec_same < 1e-4

    # 2. Hai tín hiệu tần số lệch nhau (400 Hz vs 3000 Hz) -> D_spec > 0
    sine_b = np.sin(2 * np.pi * 3000 * t).astype(np.float32)
    d_spec_diff = compute_spectral_discontinuity(sine_a, sine_b, sr=sr)
    assert d_spec_diff > 10.0


def test_evaluate_crossfade_experiment():
    """Kiểm tra hàm chạy Thí nghiệm 1 khảo sát 4 mức cross-fade."""
    from src.smoothing import evaluate_crossfade_experiment
    sr = 16000
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    u1 = np.sin(2 * np.pi * 300 * t).astype(np.float32)
    u2 = np.sin(2 * np.pi * 320 * t).astype(np.float32)
    u3 = np.sin(2 * np.pi * 350 * t).astype(np.float32)

    results = evaluate_crossfade_experiment([u1, u2, u3], sr=sr, xfade_levels=[0.0, 5.0, 15.0, 30.0])
    assert len(results) == 4
    for r in results:
        assert "xfade_ms" in r
        assert "mean_d_spec" in r
        assert r["num_boundaries"] == 2

