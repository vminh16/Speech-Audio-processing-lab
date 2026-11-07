"""
Unit tests for Prosody, Source-Filter Model, and Voice Conversion (src/prosody.py).
Following Test-Driven Development (TDD).
"""

import numpy as np
import pytest
from src.prosody import (
    estimate_pitch_period_autocorr,
    semitone_to_ratio,
    time_stretch,
    pitch_shift,
    voice_conversion_female,
    voice_conversion_southern,
    verify_f0_formula,
)


def test_autocorr_pitch_sine_wave():
    """Kiểm tra đo chu kỳ pitch N_p trên sóng sin 200 Hz chuẩn ở Fs = 16000 Hz."""
    sr = 16000
    f_target = 200.0
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)  # 50ms frame
    sine = np.sin(2 * np.pi * f_target * t).astype(np.float32)

    # N_p lý thuyết = 16000 / 200 = 80 mẫu
    f0_est, n_p = estimate_pitch_period_autocorr(sine, sr=sr)
    assert n_p == 80
    assert np.isclose(f0_est, 200.0, atol=1.0)


def test_semitone_formula():
    """Kiểm tra công thức tỷ số tần số r = 2^(delta_s / 12)."""
    assert np.isclose(semitone_to_ratio(0.0), 1.0)
    assert np.isclose(semitone_to_ratio(12.0), 2.0)
    assert np.isclose(semitone_to_ratio(-12.0), 0.5)
    assert np.isclose(semitone_to_ratio(2.0), 2.0 ** (2.0 / 12.0))
    assert np.isclose(semitone_to_ratio(-2.0), 2.0 ** (-2.0 / 12.0))


def test_time_stretch():
    """Kiểm tra co giãn thời lượng (Thí nghiệm 3)."""
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    y = np.sin(2 * np.pi * 300 * t).astype(np.float32)

    # 1. Tốc độ 1.2x -> thời lượng co lại còn ~0.5 / 1.2 = 0.417s
    y_fast = time_stretch(y, rate=1.2, sr=sr)
    dur_fast = len(y_fast) / sr
    assert np.isclose(dur_fast, 0.5 / 1.2, atol=0.03)

    # 2. Tốc độ 0.8x -> thời lượng giãn ra ~0.5 / 0.8 = 0.625s
    y_slow = time_stretch(y, rate=0.8, sr=sr)
    dur_slow = len(y_slow) / sr
    assert np.isclose(dur_slow, 0.5 / 0.8, atol=0.03)


def test_pitch_shift_preserves_length():
    """Kiểm tra dịch pitch (Thí nghiệm 4) bảo toàn thời lượng gốc."""
    sr = 16000
    t = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False)
    y = np.sin(2 * np.pi * 300 * t).astype(np.float32)

    y_high = pitch_shift(y, n_steps=2.0, sr=sr)
    y_low = pitch_shift(y, n_steps=-2.0, sr=sr)

    # Sai số thời lượng < 5%
    assert np.isclose(len(y_high) / sr, 0.4, atol=0.02)
    assert np.isclose(len(y_low) / sr, 0.4, atol=0.02)


def test_voice_conversion_female():
    """Kiểm tra biến đổi giọng Nữ giả lập (Male-to-Female)."""
    sr = 16000
    t = np.linspace(0, 0.3, int(sr * 0.3), endpoint=False)
    # Giọng nam trầm 120 Hz
    y_male = np.sin(2 * np.pi * 120 * t).astype(np.float32)

    y_female = voice_conversion_female(y_male, sr=sr, pitch_shift_steps=5.5)
    f0_male, _ = estimate_pitch_period_autocorr(y_male, sr=sr)
    f0_female, _ = estimate_pitch_period_autocorr(y_female, sr=sr)

    # F0 giọng nữ phải cao hơn F0 giọng nam từ 30% trở lên
    assert f0_female > f0_male * 1.30


def test_voice_conversion_southern():
    """Kiểm tra biến đổi ngữ điệu miền Nam (kéo dài âm ngân)."""
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    y = (0.08 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    y_south = voice_conversion_southern(y, sr=sr)
    # Ngữ điệu miền Nam kéo nhẹ âm đuôi
    assert len(y_south) >= len(y)


def test_verify_f0_real_file():
    """Kiểm tra kiểm chứng F0 = Fs / N_p trên file thật data/units/ba_01.wav."""
    res = verify_f0_formula("data/units/ba_01.wav", sr=16000)
    assert "f0_formula" in res
    assert "n_p" in res
    assert "sr" in res
    assert 70.0 <= res["f0_formula"] <= 250.0  # Tần số giọng người tự nhiên
