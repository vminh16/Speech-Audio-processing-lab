"""
Unit Selection Module using Viterbi Dynamic Programming.
CSE457 - Lab 04: Text-to-Speech
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import soundfile as sf

from src.concatenative import TOKEN_TO_FILENAME
from src.smoothing import compute_spectral_discontinuity
from src.utils import compute_rms


def get_candidates_for_token(token: str, units_dir: str = "data/units") -> List[Dict[str, Any]]:
    """
    Tìm kiếm và nạp toàn bộ các candidate WAV có sẵn cho một token trong kho units_dir.
    Trả về danh sách các dictionary chứa thông tin:
    {"token": token, "candidate_id": id, "audio": y, "duration": dur, "rms": rms, "file_path": path}
    """
    clean_tok = token.strip().lower()
    base_name = TOKEN_TO_FILENAME.get(clean_tok, clean_tok)
    units_path = Path(units_dir)

    # Tìm các file có định dạng base_*.wav hoặc token_*.wav
    matching_files = sorted(list(units_path.glob(f"{base_name}_*.wav")))
    if not matching_files:
        matching_files = sorted(list(units_path.glob(f"{clean_tok}_*.wav")))

    if not matching_files:
        raise FileNotFoundError(f"Không tìm thấy candidate nào cho từ '{token}' trong {units_dir}")

    candidates = []
    for f in matching_files:
        # Tách candidate_id từ stem (ví dụ moi_02 -> '02')
        stem = f.stem
        candidate_id = stem.rsplit("_", 1)[1] if "_" in stem else "01"

        try:
            audio, sr = sf.read(str(f))
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            audio = audio.astype(np.float32)

            dur = float(len(audio) / sr)
            rms_val = compute_rms(audio)

            candidates.append({
                "token": clean_tok,
                "candidate_id": candidate_id,
                "audio": audio,
                "samplerate": sr,
                "duration": dur,
                "rms": rms_val,
                "file_path": str(f),
            })
        except Exception as e:
            print(f"Lỗi khi nạp candidate {f}: {e}")

    if not candidates:
        raise FileNotFoundError(f"Không thể đọc được file candidate nào cho từ '{token}'")

    return candidates


def compute_target_cost(
    candidate: Dict[str, Any],
    avg_duration: Optional[float] = None,
    target_rms: float = 0.08,
    weight_dur: float = 1.0,
    weight_rms: float = 5.0,
) -> float:
    """
    Tính Target Cost C_target(theta_j):
    Đo độ sai lệch của candidate về thời lượng và năng lượng RMS so với giá trị mục tiêu.

    Công thức:
        C_target = w_d * ((duration - avg_duration) / avg_duration)^2
                 + w_e * ((rms - target_rms) / target_rms)^2
    """
    c_dur = 0.0
    dur = candidate.get("duration", 0.0)
    if avg_duration is not None and avg_duration > 1e-4:
        c_dur = ((dur - avg_duration) / avg_duration) ** 2

    rms = candidate.get("rms", 0.0)
    c_rms = 0.0
    if target_rms > 1e-4:
        c_rms = ((rms - target_rms) / target_rms) ** 2

    return float(weight_dur * c_dur + weight_rms * c_rms)


def compute_join_cost(
    candidate_a: Dict[str, Any],
    candidate_b: Dict[str, Any],
    sr: int = 16000,
    weight_spec: float = 1.0,
    weight_rms: float = 2.0,
) -> float:
    """
    Tính Join Cost C_join(theta_j, theta_{j+1}):
    Đo độ gián đoạn phổ D_spec kết hợp với độ lệch năng lượng RMS giữa đuôi unit A và đầu unit B.

    Công thức:
        C_join = w_spec * D_spec + w_rms * ((RMS_a - RMS_b) / 0.08)^2
    """
    audio_a = candidate_a["audio"]
    audio_b = candidate_b["audio"]

    d_spec = compute_spectral_discontinuity(audio_a, audio_b, sr=sr, window_ms=25.0)

    rms_a = candidate_a.get("rms", compute_rms(audio_a))
    rms_b = candidate_b.get("rms", compute_rms(audio_b))
    c_rms = ((rms_a - rms_b) / 0.08) ** 2

    return float(weight_spec * d_spec + weight_rms * c_rms)


def viterbi_unit_selection(
    tokens: List[str],
    units_dir: str = "data/units",
    weight_join: float = 1.0,
    sr: int = 16000,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Thuật toán Quy hoạch động Viterbi tìm chuỗi candidate tối ưu toàn cục.

    Đầu vào:
        tokens: Danh sách các từ cần ghép nối.
        units_dir: Thư mục chứa inventory.
        weight_join (lambda): Trọng số giữa Target Cost và Join Cost.

    Trả về:
        (optimal_candidate_sequence, cost_metrics)
    """
    if not tokens:
        return [], {"total_cost": 0.0, "target_cost": 0.0, "join_cost": 0.0}

    # 1. Nạp tất cả các candidate theo từng vị trí token
    lattice: List[List[Dict[str, Any]]] = []
    avg_durations: List[float] = []

    for tok in tokens:
        cands = get_candidates_for_token(tok, units_dir=units_dir)
        lattice.append(cands)
        avg_dur = float(np.mean([c["duration"] for c in cands]))
        avg_durations.append(avg_dur)

    num_stages = len(tokens)

    # 2. Khởi tạo Trellis Cost và Backpointers
    # V[j][c_idx] lưu tổng chi phí nhỏ nhất để đến candidate c_idx ở vị trí j
    V: List[List[float]] = []
    bp: List[List[int]] = []

    # Khởi tạo cho vị trí j = 0
    V_0 = []
    for c in lattice[0]:
        cost_t = compute_target_cost(c, avg_duration=avg_durations[0])
        V_0.append(cost_t)
    V.append(V_0)
    bp.append([0] * len(lattice[0]))

    # 3. Quy nạp Viterbi qua từng bước thời gian j = 1 ... J-1
    for j in range(1, num_stages):
        curr_cands = lattice[j]
        prev_cands = lattice[j - 1]
        V_j = []
        bp_j = []

        for curr_idx, curr_c in enumerate(curr_cands):
            cost_target = compute_target_cost(curr_c, avg_duration=avg_durations[j])

            # Tìm candidate ở bước trước p tối thiểu hóa: V[j-1][p] + lambda * C_join(p, curr_c)
            best_prev_cost = float("inf")
            best_p_idx = 0

            for prev_idx, prev_c in enumerate(prev_cands):
                cost_join = compute_join_cost(prev_c, curr_c, sr=sr)
                accum_cost = V[j - 1][prev_idx] + weight_join * cost_join

                if accum_cost < best_prev_cost:
                    best_prev_cost = accum_cost
                    best_p_idx = prev_idx

            total_node_cost = cost_target + best_prev_cost
            V_j.append(total_node_cost)
            bp_j.append(best_p_idx)

        V.append(V_j)
        bp.append(bp_j)

    # 4. Truy vết ngược (Backtracking)
    best_last_idx = int(np.argmin(V[-1]))
    optimal_path_indices = [best_last_idx]

    for j in range(num_stages - 1, 0, -1):
        prev_best_idx = bp[j][optimal_path_indices[-1]]
        optimal_path_indices.append(prev_best_idx)

    optimal_path_indices.reverse()

    # Trích xuất chuỗi candidate tối ưu
    optimal_sequence = [
        lattice[j][optimal_path_indices[j]] for j in range(num_stages)
    ]

    # Tính toán phân rã chi phí
    total_target = sum(
        compute_target_cost(optimal_sequence[j], avg_duration=avg_durations[j])
        for j in range(num_stages)
    )
    total_join = sum(
        compute_join_cost(optimal_sequence[j], optimal_sequence[j + 1], sr=sr)
        for j in range(num_stages - 1)
    )
    total_cost = total_target + weight_join * total_join

    metrics = {
        "total_cost": round(total_cost, 4),
        "target_cost": round(total_target, 4),
        "join_cost": round(total_join, 4),
        "optimal_path": [c["candidate_id"] for c in optimal_sequence],
    }

    return optimal_sequence, metrics


def evaluate_fixed_strategy(
    tokens: List[str],
    candidate_id: str = "01",
    units_dir: str = "data/units",
    weight_join: float = 1.0,
    sr: int = 16000,
) -> Dict[str, Any]:
    """Tính toán chi phí cho phương án chọn candidate cố định."""
    units = []
    for tok in tokens:
        cands = get_candidates_for_token(tok, units_dir=units_dir)
        # Tìm đúng candidate_id, nếu không có lấy candidate đầu tiên
        match = next((c for c in cands if c["candidate_id"] == candidate_id), cands[0])
        units.append(match)

    num_stages = len(tokens)
    avg_durations = [
        float(np.mean([c["duration"] for c in get_candidates_for_token(tok, units_dir=units_dir)]))
        for tok in tokens
    ]

    total_target = sum(
        compute_target_cost(units[j], avg_duration=avg_durations[j])
        for j in range(num_stages)
    )
    total_join = sum(
        compute_join_cost(units[j], units[j + 1], sr=sr)
        for j in range(num_stages - 1)
    )
    total_cost = total_target + weight_join * total_join

    return {
        "units": units,
        "total_cost": round(total_cost, 4),
        "target_cost": round(total_target, 4),
        "join_cost": round(total_join, 4),
        "path": [c["candidate_id"] for c in units],
    }


def compare_candidate_strategies(
    tokens: List[str],
    units_dir: str = "data/units",
    weight_join: float = 1.0,
    sr: int = 16000,
) -> Dict[str, Any]:
    """
    So sánh Thí nghiệm 2: Đối sánh chi phí giữa các phương án:
    1. Cố định candidate _01.
    2. Cố định candidate _02.
    3. Thuật toán Viterbi tìm kiếm toàn cục.
    """
    fixed_01 = evaluate_fixed_strategy(tokens, candidate_id="01", units_dir=units_dir, weight_join=weight_join, sr=sr)
    fixed_02 = evaluate_fixed_strategy(tokens, candidate_id="02", units_dir=units_dir, weight_join=weight_join, sr=sr)
    optimal_seq, optimal_metrics = viterbi_unit_selection(tokens, units_dir=units_dir, weight_join=weight_join, sr=sr)

    # Tính mức độ giảm chi phí (%) so với fixed_01
    cost_base = fixed_01["total_cost"]
    cost_opt = optimal_metrics["total_cost"]
    reduction_pct = round(((cost_base - cost_opt) / cost_base) * 100.0, 2) if cost_base > 0 else 0.0

    return {
        "fixed_01": fixed_01,
        "fixed_02": fixed_02,
        "optimal_viterbi": {
            "units": optimal_seq,
            "total_cost": optimal_metrics["total_cost"],
            "target_cost": optimal_metrics["target_cost"],
            "join_cost": optimal_metrics["join_cost"],
            "path": optimal_metrics["optimal_path"],
            "reduction_pct": reduction_pct,
        },
    }
