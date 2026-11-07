"""
Unit tests for audio utilities and metadata extraction in Lab 04 TTS.
Following Test-Driven Development (TDD).
"""

import numpy as np
import pytest
import soundfile as sf
from src.utils import (
    compute_rms,
    match_rms,
    check_audio_format,
    trim_silence,
    extract_inventory_metadata,
)


def test_compute_rms_silence_and_sine():
    """Kiểm tra tính toán RMS trên tín hiệu im lặng và sóng sin."""
    silence = np.zeros(16000, dtype=np.float32)
    # Silence RMS should be close to 0 (around sqrt(1e-12) = 1e-6)
    assert compute_rms(silence) < 1e-4

    # Sóng sin có biên độ A=1.0 thì RMS lý thuyết là 1/sqrt(2) approx 0.7071
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    expected_rms = 1.0 / np.sqrt(2.0)
    assert np.isclose(compute_rms(sine), expected_rms, atol=1e-3)


def test_match_rms():
    """Kiểm tra chuẩn hóa năng lượng RMS về target = 0.08."""
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    sine = 0.2 * np.sin(2 * np.pi * 300 * t).astype(np.float32)

    normalized = match_rms(sine, target_rms=0.08)
    assert np.isclose(compute_rms(normalized), 0.08, atol=1e-3)
    # Kiểm tra không bị vượt quá ngưỡng clip [-0.99, 0.99]
    assert np.max(np.abs(normalized)) <= 0.99


def test_check_audio_format_and_metadata(tmp_path):
    """Kiểm tra xác thực định dạng audio mono 16kHz PCM_16 và xuất metadata."""
    # Tạo một file wav test chuẩn 16kHz, mono
    sr = 16000
    duration_s = 0.4
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    y = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    units_dir = tmp_path / "units"
    units_dir.mkdir()
    sample_file = units_dir / "moi_01.wav"
    sf.write(str(sample_file), y, sr, subtype="PCM_16")

    # Kiểm tra format check
    fmt_info = check_audio_format(str(sample_file), expected_sr=16000)
    assert fmt_info["is_valid"] is True
    assert fmt_info["samplerate"] == 16000
    assert fmt_info["channels"] == 1

    # Kiểm tra trích xuất metadata ra CSV
    csv_file = tmp_path / "metadata.csv"
    df = extract_inventory_metadata(str(units_dir), str(csv_file))
    assert len(df) == 1
    assert df.iloc[0]["label"] == "moi"
    assert df.iloc[0]["candidate_id"] == "01"
    assert np.isclose(df.iloc[0]["duration"], duration_s, atol=0.01)
    assert df.iloc[0]["rms"] > 0
