"""
Unit tests for concatenative synthesis module (src/concatenative.py).
Following Test-Driven Development (TDD).
"""

import numpy as np
import pytest
import soundfile as sf
from src.concatenative import (
    load_unit,
    build_baseline_sequence,
    synthesize_baseline,
)


@pytest.fixture
def mock_units_dir(tmp_path):
    """Tạo một kho unit giả lập gồm 3 từ chuẩn 16kHz."""
    sr = 16000
    units_dir = tmp_path / "units"
    units_dir.mkdir()

    for word in ["moi", "khach", "hang"]:
        t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
        y = 0.1 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
        sf.write(str(units_dir / f"{word}_01.wav"), y, sr, subtype="PCM_16")

    return units_dir


def test_load_unit(mock_units_dir):
    """Kiểm tra đọc file unit từ thư mục."""
    y, sr = load_unit("moi", candidate_id="01", units_dir=str(mock_units_dir))
    assert sr == 16000
    assert len(y) == int(16000 * 0.2)
    assert y.ndim == 1


def test_build_baseline_sequence(mock_units_dir):
    """Kiểm tra nạp danh sách các unit theo thứ tự tokens."""
    tokens = ["mời", "khách", "hàng"]
    units = build_baseline_sequence(tokens, units_dir=str(mock_units_dir), candidate_id="01")
    assert len(units) == 3
    for u in units:
        assert len(u) == 3200


def test_synthesize_baseline(mock_units_dir):
    """Kiểm tra tổng hợp baseline (xfade = 0)."""
    tokens = ["mời", "khách"]
    y_raw, boundaries = synthesize_baseline(tokens, units_dir=str(mock_units_dir))

    # Ghép 2 unit mỗi unit 3200 mẫu -> 6400 mẫu
    assert len(y_raw) == 6400
    assert len(boundaries) == 1
    assert np.isclose(boundaries[0], 0.2, atol=1e-3)
