"""
Unit tests for Unit Selection and Viterbi Search (src/unit_selection.py).
Following Test-Driven Development (TDD).
"""

import numpy as np
import pytest
import soundfile as sf
from src.unit_selection import (
    get_candidates_for_token,
    compute_target_cost,
    compute_join_cost,
    viterbi_unit_selection,
    compare_candidate_strategies,
)


@pytest.fixture
def sample_units_dir(tmp_path):
    """Tạo một kho unit giả lập gồm 2 từ, mỗi từ có 3 candidate."""
    sr = 16000
    units_dir = tmp_path / "units"
    units_dir.mkdir()

    # Từ 'moi' có 3 candidate với các đặc tính khác nhau
    t1 = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
    sf.write(str(units_dir / "moi_01.wav"), 0.08 * np.sin(2 * np.pi * 300 * t1).astype(np.float32), sr)
    sf.write(str(units_dir / "moi_02.wav"), 0.08 * np.sin(2 * np.pi * 350 * t1).astype(np.float32), sr)
    # candidate _03 bị lệch năng lượng
    sf.write(str(units_dir / "moi_03.wav"), 0.16 * np.sin(2 * np.pi * 400 * t1).astype(np.float32), sr)

    # Từ 'khach' có 3 candidate
    t2 = np.linspace(0, 0.30, int(sr * 0.30), endpoint=False)
    # candidate _01 tần số lệch nhiều so với moi (1500 Hz)
    sf.write(str(units_dir / "khach_01.wav"), 0.08 * np.sin(2 * np.pi * 1500 * t2).astype(np.float32), sr)
    # candidate _02 tần số khớp mượt với đuôi moi_01 (300 Hz)
    sf.write(str(units_dir / "khach_02.wav"), 0.08 * np.sin(2 * np.pi * 300 * t2).astype(np.float32), sr)
    sf.write(str(units_dir / "khach_03.wav"), 0.08 * np.sin(2 * np.pi * 800 * t2).astype(np.float32), sr)

    return str(units_dir)


def test_get_candidates_for_token(sample_units_dir):
    """Kiểm tra tìm kiếm và nạp đầy đủ danh sách candidate của một từ."""
    candidates = get_candidates_for_token("mời", units_dir=sample_units_dir)
    assert len(candidates) == 3
    candidate_ids = [c["candidate_id"] for c in candidates]
    assert "01" in candidate_ids
    assert "02" in candidate_ids
    assert "03" in candidate_ids
    for c in candidates:
        assert "audio" in c
        assert "duration" in c
        assert "rms" in c


def test_compute_target_cost():
    """Kiểm tra hàm tính Target Cost."""
    # Unit chuẩn: duration gần mục tiêu (0.25s), RMS đúng 0.08
    cand_good = {"duration": 0.25, "rms": 0.08}
    cost_good = compute_target_cost(cand_good, avg_duration=0.25, target_rms=0.08)

    # Unit lệch RMS (0.16)
    cand_bad_rms = {"duration": 0.25, "rms": 0.16}
    cost_bad_rms = compute_target_cost(cand_bad_rms, avg_duration=0.25, target_rms=0.08)

    # Unit lệch duration (0.50s)
    cand_bad_dur = {"duration": 0.50, "rms": 0.08}
    cost_bad_dur = compute_target_cost(cand_bad_dur, avg_duration=0.25, target_rms=0.08)

    assert cost_good < cost_bad_rms
    assert cost_good < cost_bad_dur


def test_compute_join_cost():
    """Kiểm tra hàm tính Join Cost."""
    sr = 16000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    # Đoạn sóng 400 Hz
    wave_400 = (0.08 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    # Đoạn sóng 405 Hz (gần 400 Hz)
    wave_405 = (0.08 * np.sin(2 * np.pi * 405 * t)).astype(np.float32)
    # Đoạn sóng 3000 Hz (lệch nhiều)
    wave_3000 = (0.08 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)

    c1 = {"audio": wave_400, "rms": 0.08}
    c2 = {"audio": wave_405, "rms": 0.08}
    c3 = {"audio": wave_3000, "rms": 0.08}

    cost_smooth = compute_join_cost(c1, c2, sr=sr)
    cost_discontinuous = compute_join_cost(c1, c3, sr=sr)

    assert cost_smooth < cost_discontinuous


def test_viterbi_unit_selection(sample_units_dir):
    """Kiểm tra thuật toán Viterbi tìm ra chuỗi candidate có tổng chi phí nhỏ nhất."""
    tokens = ["mời", "khách"]
    optimal_sequence, metrics = viterbi_unit_selection(tokens, units_dir=sample_units_dir, weight_join=1.0)

    assert len(optimal_sequence) == 2
    assert "total_cost" in metrics
    assert "target_cost" in metrics
    assert "join_cost" in metrics

    # moi_01 (300Hz) ghép với khach_02 (300Hz) sẽ mượt hơn khach_01 (1500Hz)
    assert optimal_sequence[0]["candidate_id"] == "01"
    assert optimal_sequence[1]["candidate_id"] == "02"


def test_compare_candidate_strategies_real_data():
    """Kiểm tra Viterbi trên kho dữ liệu thực tế 58 units luôn có tổng chi phí <= fixed candidate _01."""
    tokens = ["mời", "khách", "hàng"]
    comparison = compare_candidate_strategies(tokens, units_dir="data/units")

    assert "fixed_01" in comparison
    assert "optimal_viterbi" in comparison

    cost_fixed_01 = comparison["fixed_01"]["total_cost"]
    cost_optimal = comparison["optimal_viterbi"]["total_cost"]

    # Viterbi tối ưu toàn cục phải có tổng chi phí nhỏ hơn hoặc bằng phương án cố định
    assert cost_optimal <= cost_fixed_01
