"""
HMM-based Parametric Speech Synthesis Module (Chế độ sinh tham số thống kê HMM).
CSE457 - Lab 04: Text-to-Speech

Nguyên lý hoạt động (SPEC Section 2.6 & 4.5):
1. Chuyển đổi chuỗi văn bản -> Chuỗi âm vị / trạng thái HMM (State Sequence).
2. Mỗi trạng thái HMM sở hữu vector kỳ vọng tham số tĩnh c_t (F0, F1, F2, F3, Energy).
3. Ràng buộc tham số động Delta c_t = c_t - c_{t-1} qua giải thuật MLPG (Maximum Likelihood Parameter Generation).
4. Bộ tổng hợp âm học Vocoder: Khởi tạo nguồn kích thích (Glottal Pulse / Noise)
   và lọc qua hệ thống bộ cộng hưởng Formant IIR theo thời gian.
"""

import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from src.text_norm import normalize_queue_text

# Bảng tham số thống kê ngữ âm miền hàng đợi (Phonetic Acoustic Models for Queue Words)
# Mỗi từ gồm 3 trạng thái HMM: [0: Khởi âm (Onset), 1: Hạt nhân nguyên âm (Nucleus), 2: Âm kết (Coda)]
# Tham số [F0_base_ratio, F1, F2, F3, duration_ms, is_voiced]
HMM_STATE_MODELS: Dict[str, List[Dict[str, Any]]] = {
    "mời": [
        {"f0": 135.0, "f1": 320.0, "f2": 950.0, "f3": 2400.0, "dur_ms": 70, "voiced": True, "gain": 0.5},
        {"f0": 125.0, "f1": 480.0, "f2": 1100.0, "f3": 2500.0, "dur_ms": 130, "voiced": True, "gain": 0.9},
        {"f0": 115.0, "f1": 350.0, "f2": 2100.0, "f3": 2800.0, "dur_ms": 80, "voiced": True, "gain": 0.4},
    ],
    "khách": [
        {"f0": 0.0, "f1": 450.0, "f2": 1500.0, "f3": 2600.0, "dur_ms": 80, "voiced": False, "gain": 0.4},
        {"f0": 165.0, "f1": 750.0, "f2": 1450.0, "f3": 2550.0, "dur_ms": 140, "voiced": True, "gain": 0.9},
        {"f0": 0.0, "f1": 500.0, "f2": 1800.0, "f3": 2700.0, "dur_ms": 70, "voiced": False, "gain": 0.3},
    ],
    "hàng": [
        {"f0": 0.0, "f1": 400.0, "f2": 1300.0, "f3": 2400.0, "dur_ms": 70, "voiced": False, "gain": 0.4},
        {"f0": 135.0, "f1": 780.0, "f2": 1320.0, "f3": 2450.0, "dur_ms": 150, "voiced": True, "gain": 0.95},
        {"f0": 120.0, "f1": 400.0, "f2": 1150.0, "f3": 2350.0, "dur_ms": 90, "voiced": True, "gain": 0.5},
    ],
    "số": [
        {"f0": 0.0, "f1": 380.0, "f2": 1600.0, "f3": 2700.0, "dur_ms": 90, "voiced": False, "gain": 0.4},
        {"f0": 160.0, "f1": 520.0, "f2": 980.0, "f3": 2350.0, "dur_ms": 130, "voiced": True, "gain": 0.9},
        {"f0": 178.0, "f1": 480.0, "f2": 920.0, "f3": 2300.0, "dur_ms": 60, "voiced": True, "gain": 0.6},
    ],
    "đến": [
        {"f0": 145.0, "f1": 350.0, "f2": 1650.0, "f3": 2600.0, "dur_ms": 60, "voiced": True, "gain": 0.6},
        {"f0": 168.0, "f1": 550.0, "f2": 1950.0, "f3": 2700.0, "dur_ms": 130, "voiced": True, "gain": 0.95},
        {"f0": 175.0, "f1": 400.0, "f2": 1600.0, "f3": 2500.0, "dur_ms": 80, "voiced": True, "gain": 0.5},
    ],
    "quầy": [
        {"f0": 0.0, "f1": 360.0, "f2": 1100.0, "f3": 2400.0, "dur_ms": 70, "voiced": False, "gain": 0.3},
        {"f0": 135.0, "f1": 580.0, "f2": 1200.0, "f3": 2450.0, "dur_ms": 130, "voiced": True, "gain": 0.9},
        {"f0": 120.0, "f1": 380.0, "f2": 2100.0, "f3": 2800.0, "dur_ms": 70, "voiced": True, "gain": 0.4},
    ],
    "không": [
        {"f0": 0.0, "f1": 400.0, "f2": 1400.0, "f3": 2500.0, "dur_ms": 70, "voiced": False, "gain": 0.3},
        {"f0": 150.0, "f1": 500.0, "f2": 950.0, "f3": 2300.0, "dur_ms": 140, "voiced": True, "gain": 0.9},
        {"f0": 148.0, "f1": 420.0, "f2": 1100.0, "f3": 2350.0, "dur_ms": 80, "voiced": True, "gain": 0.5},
    ],
    "một": [
        {"f0": 135.0, "f1": 340.0, "f2": 1050.0, "f3": 2400.0, "dur_ms": 60, "voiced": True, "gain": 0.6},
        {"f0": 125.0, "f1": 520.0, "f2": 980.0, "f3": 2350.0, "dur_ms": 110, "voiced": True, "gain": 0.9},
        {"f0": 105.0, "f1": 400.0, "f2": 1600.0, "f3": 2500.0, "dur_ms": 60, "voiced": False, "gain": 0.2},
    ],
    "hai": [
        {"f0": 0.0, "f1": 420.0, "f2": 1400.0, "f3": 2500.0, "dur_ms": 60, "voiced": False, "gain": 0.3},
        {"f0": 152.0, "f1": 760.0, "f2": 1350.0, "f3": 2500.0, "dur_ms": 130, "voiced": True, "gain": 0.95},
        {"f0": 150.0, "f1": 350.0, "f2": 2150.0, "f3": 2850.0, "dur_ms": 80, "voiced": True, "gain": 0.5},
    ],
    "ba": [
        {"f0": 148.0, "f1": 320.0, "f2": 1000.0, "f3": 2350.0, "dur_ms": 60, "voiced": True, "gain": 0.7},
        {"f0": 152.0, "f1": 770.0, "f2": 1320.0, "f3": 2480.0, "dur_ms": 140, "voiced": True, "gain": 0.95},
        {"f0": 150.0, "f1": 720.0, "f2": 1300.0, "f3": 2450.0, "dur_ms": 60, "voiced": True, "gain": 0.6},
    ],
    "bốn": [
        {"f0": 150.0, "f1": 340.0, "f2": 1050.0, "f3": 2400.0, "dur_ms": 60, "voiced": True, "gain": 0.7},
        {"f0": 168.0, "f1": 520.0, "f2": 1000.0, "f3": 2350.0, "dur_ms": 120, "voiced": True, "gain": 0.95},
        {"f0": 178.0, "f1": 420.0, "f2": 1600.0, "f3": 2500.0, "dur_ms": 70, "voiced": True, "gain": 0.5},
    ],
    "năm": [
        {"f0": 145.0, "f1": 340.0, "f2": 1550.0, "f3": 2550.0, "dur_ms": 60, "voiced": True, "gain": 0.6},
        {"f0": 152.0, "f1": 740.0, "f2": 1360.0, "f3": 2500.0, "dur_ms": 130, "voiced": True, "gain": 0.95},
        {"f0": 150.0, "f1": 380.0, "f2": 1100.0, "f3": 2400.0, "dur_ms": 70, "voiced": True, "gain": 0.5},
    ],
    "sáu": [
        {"f0": 0.0, "f1": 380.0, "f2": 1600.0, "f3": 2700.0, "dur_ms": 80, "voiced": False, "gain": 0.4},
        {"f0": 165.0, "f1": 720.0, "f2": 1350.0, "f3": 2480.0, "dur_ms": 120, "voiced": True, "gain": 0.95},
        {"f0": 180.0, "f1": 450.0, "f2": 900.0, "f3": 2300.0, "dur_ms": 60, "voiced": True, "gain": 0.5},
    ],
    "bảy": [
        {"f0": 135.0, "f1": 320.0, "f2": 1000.0, "f3": 2350.0, "dur_ms": 60, "voiced": True, "gain": 0.6},
        {"f0": 120.0, "f1": 680.0, "f2": 1400.0, "f3": 2550.0, "dur_ms": 130, "voiced": True, "gain": 0.9},
        {"f0": 142.0, "f1": 360.0, "f2": 2100.0, "f3": 2800.0, "dur_ms": 70, "voiced": True, "gain": 0.5},
    ],
    "tám": [
        {"f0": 0.0, "f1": 380.0, "f2": 1600.0, "f3": 2600.0, "dur_ms": 60, "voiced": False, "gain": 0.3},
        {"f0": 165.0, "f1": 760.0, "f2": 1350.0, "f3": 2500.0, "dur_ms": 130, "voiced": True, "gain": 0.95},
        {"f0": 178.0, "f1": 380.0, "f2": 1050.0, "f3": 2350.0, "dur_ms": 70, "voiced": True, "gain": 0.5},
    ],
    "chín": [
        {"f0": 0.0, "f1": 360.0, "f2": 1900.0, "f3": 2800.0, "dur_ms": 70, "voiced": False, "gain": 0.3},
        {"f0": 168.0, "f1": 320.0, "f2": 2250.0, "f3": 2950.0, "dur_ms": 130, "voiced": True, "gain": 0.95},
        {"f0": 182.0, "f1": 380.0, "f2": 1600.0, "f3": 2500.0, "dur_ms": 70, "voiced": True, "gain": 0.5},
    ],
    "_pause": [
        {"f0": 0.0, "f1": 300.0, "f2": 1000.0, "f3": 2200.0, "dur_ms": 50, "voiced": False, "gain": 0.0},
        {"f0": 0.0, "f1": 300.0, "f2": 1000.0, "f3": 2200.0, "dur_ms": 50, "voiced": False, "gain": 0.0},
        {"f0": 0.0, "f1": 300.0, "f2": 1000.0, "f3": 2200.0, "dur_ms": 50, "voiced": False, "gain": 0.0},
    ],
}


def apply_mlpg_smoothing(step_means: np.ndarray, delta_weight: float = 0.6) -> np.ndarray:
    """
    Giải thuật MLPG (Maximum Likelihood Parameter Generation) làm mượt quỹ đạo tham số HMM.
    Khắc phục hiện tượng bước nhảy bậc thang giữa các trạng thái bằng cách tối ưu hóa:
        argmin_c ||c - mu||^2 + lambda * ||D c||^2
    Nghiệm tối ưu thỏa mãn hệ phương trình ma trận dải đối xứng:
        (I + lambda * D^T D) c = mu
    """
    n = len(step_means)
    if n <= 2 or delta_weight <= 0.0:
        return step_means.copy()

    # Ma trận vi phân cấp 1 D: (D c)_t = c_{t+1} - c_t
    main_diag = np.ones(n) * (1.0 + 2.0 * delta_weight)
    main_diag[0] = 1.0 + delta_weight
    main_diag[-1] = 1.0 + delta_weight

    off_diag = np.ones(n - 1) * (-delta_weight)

    diagonals = [off_diag, main_diag, off_diag]
    A = diags(diagonals, [-1, 0, 1], shape=(n, n), format="csc")

    # Giải hệ phương trình tuyến tính A * c = step_means
    smoothed = spsolve(A, step_means)
    return np.asarray(smoothed, dtype=np.float64)


def generate_hmm_trajectories(
    text: str,
    sr: int = 16000,
    frame_shift_ms: float = 10.0,
    speed: float = 1.0,
    pitch_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    Sinh chuỗi trạng thái và quỹ đạo tham số âm học liên tục qua MLPG.
    """
    tokens = normalize_queue_text(text)
    if not tokens:
        tokens = ["mời"]

    # Xây dựng chuỗi trạng thái HMM
    raw_f0: List[float] = []
    raw_f1: List[float] = []
    raw_f2: List[float] = []
    raw_f3: List[float] = []
    raw_gain: List[float] = []
    state_labels: List[str] = []
    word_boundaries: List[int] = [0]

    for tok_idx, tok in enumerate(tokens):
        states = HMM_STATE_MODELS.get(tok, HMM_STATE_MODELS["mời"])
        for s_idx, state in enumerate(states):
            dur = max(2, int(round((state["dur_ms"] / speed) / frame_shift_ms)))
            target_f0 = state["f0"] * pitch_scale if state["voiced"] else 0.0
            raw_f0.extend([target_f0] * dur)
            raw_f1.extend([state["f1"]] * dur)
            raw_f2.extend([state["f2"]] * dur)
            raw_f3.extend([state["f3"]] * dur)
            raw_gain.extend([state["gain"]] * dur)
            state_labels.extend([f"{tok}_s{s_idx}"] * dur)

        # Chèn khoảng nghỉ ngắn 30ms giữa các từ
        pause_dur = max(1, int(round((30.0 / speed) / frame_shift_ms)))
        raw_f0.extend([0.0] * pause_dur)
        raw_f1.extend([300.0] * pause_dur)
        raw_f2.extend([1000.0] * pause_dur)
        raw_f3.extend([2200.0] * pause_dur)
        raw_gain.extend([0.0] * pause_dur)
        state_labels.extend(["pause"] * pause_dur)
        word_boundaries.append(len(raw_f0))

    f0_arr = np.array(raw_f0, dtype=np.float64)
    f1_arr = np.array(raw_f1, dtype=np.float64)
    f2_arr = np.array(raw_f2, dtype=np.float64)
    f3_arr = np.array(raw_f3, dtype=np.float64)
    gain_arr = np.array(raw_gain, dtype=np.float64)

    # Tách các đoạn voiced để làm mượt F0 (unvoiced F0 = 0 giữ nguyên)
    voiced_mask = f0_arr > 0
    smooth_f0 = f0_arr.copy()
    if np.any(voiced_mask):
        # Làm mượt MLPG trên từng đoạn voiced liên tục
        diffs = np.diff(np.concatenate(([0], voiced_mask.astype(int), [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            if e - s >= 3:
                smooth_f0[s:e] = apply_mlpg_smoothing(f0_arr[s:e], delta_weight=0.7)

    smooth_f1 = apply_mlpg_smoothing(f1_arr, delta_weight=0.8)
    smooth_f2 = apply_mlpg_smoothing(f2_arr, delta_weight=0.8)
    smooth_f3 = apply_mlpg_smoothing(f3_arr, delta_weight=0.8)
    smooth_gain = apply_mlpg_smoothing(gain_arr, delta_weight=0.5)
    smooth_gain = np.clip(smooth_gain, 0.0, 1.0)

    time_axis = np.arange(len(smooth_f0)) * (frame_shift_ms / 1000.0)

    return {
        "text": text,
        "tokens": tokens,
        "time": time_axis,
        "f0_step": f0_arr,
        "f0": smooth_f0,
        "f1_step": f1_arr,
        "f1": smooth_f1,
        "f2_step": f2_arr,
        "f2": smooth_f2,
        "f3": smooth_f3,
        "energy": smooth_gain,
        "states": state_labels,
        "word_boundaries": word_boundaries,
        "frame_shift_ms": frame_shift_ms,
    }


def parametric_vocoder(
    tracks: Dict[str, Any],
    sr: int = 16000,
) -> np.ndarray:
    """
    Vocoder tham số: Tạo tín hiệu nguồn (Source) và lọc qua bộ cộng hưởng Formant (Filter).
    """
    f0 = tracks["f0"]
    f1 = tracks["f1"]
    f2 = tracks["f2"]
    f3 = tracks["f3"]
    energy = tracks["energy"]
    frame_shift_ms = tracks.get("frame_shift_ms", 10.0)

    hop_samples = int(sr * frame_shift_ms / 1000.0)
    total_samples = len(f0) * hop_samples
    excitation = np.zeros(total_samples, dtype=np.float32)

    # 1. Sinh nguồn kích thích (Excitation)
    phase = 0.0
    for frame_idx in range(len(f0)):
        cur_f0 = f0[frame_idx]
        cur_gain = energy[frame_idx]
        idx_start = frame_idx * hop_samples
        idx_end = idx_start + hop_samples

        if cur_gain < 0.02 or cur_f0 <= 40.0:
            # Unvoiced hoặc khoảng lặng: Nhiễu trắng Gauss
            noise = np.random.normal(0, 0.05, hop_samples).astype(np.float32) * cur_gain
            excitation[idx_start:idx_end] = noise
        else:
            # Voiced: Dãy xung Glottal Pulse tuần hoàn
            dt = cur_f0 / sr
            for i in range(hop_samples):
                phase += dt
                if phase >= 1.0:
                    phase -= 1.0
                # Glottal pulse định hình mềm mại (Liljencrants-Fant / Rosenberg approximation)
                if phase < 0.35:
                    excitation[idx_start + i] = float(0.5 * (1.0 - np.cos(np.pi * phase / 0.35)) * cur_gain)
                else:
                    excitation[idx_start + i] = float(-0.1 * np.exp(-10.0 * (phase - 0.35)) * cur_gain)

    # 2. Bộ lọc cộng hưởng Formant IIR (Vocal Tract Filter)
    # Lọc nối tiếp qua 3 bộ lọc biquad cộng hưởng tại F1, F2, F3
    y_filtered = excitation.copy()
    bw1, bw2, bw3 = 80.0, 110.0, 140.0  # Độ rộng băng thông Formant điển hình

    for frame_idx in range(len(f0)):
        idx_start = frame_idx * hop_samples
        idx_end = min(total_samples, idx_start + hop_samples)
        segment = y_filtered[idx_start:idx_end]
        if len(segment) == 0:
            continue

        c_f1 = np.clip(f1[frame_idx], 200.0, sr / 2 - 100)
        c_f2 = np.clip(f2[frame_idx], 500.0, sr / 2 - 100)
        c_f3 = np.clip(f3[frame_idx], 1500.0, sr / 2 - 100)

        # Tạo bộ lọc IIR 2 cực cho F1, F2, F3
        for fc, bw in [(c_f1, bw1), (c_f2, bw2), (c_f3, bw3)]:
            w0 = 2 * np.pi * fc / sr
            q = max(1.0, fc / bw)
            b, a = signal.iirpeak(w0, q, fs=2 * np.pi)
            segment = signal.lfilter(b, a, segment)

        y_filtered[idx_start:idx_end] = segment

    # Chuẩn hóa biên độ âm lượng
    max_val = np.max(np.abs(y_filtered))
    if max_val > 1e-6:
        y_filtered = (y_filtered / max_val) * 0.75

    return y_filtered.astype(np.float32)


def synthesize_hmm(
    text: str,
    sr: int = 16000,
    speed: float = 1.0,
    pitch_scale: float = 1.0,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Giao diện tổng hợp HMM theo chuẩn SPEC Section 4.5:
        from hmm_tts_backend import synthesize_hmm
        y_hmm, sr, tracks = synthesize_hmm("Mời khách hàng số 23 đến quầy 5")
    """
    tracks = generate_hmm_trajectories(
        text=text,
        sr=sr,
        frame_shift_ms=10.0,
        speed=speed,
        pitch_scale=pitch_scale,
    )
    y_hmm = parametric_vocoder(tracks, sr=sr)
    return y_hmm, sr, tracks


def plot_hmm_trajectories(
    tracks: Dict[str, Any],
    y_audio: np.ndarray = None,
    sr: int = 16000,
    output_fig: str = "figures/04_hmm_trajectories.png",
) -> None:
    """
    Vẽ Hình 04/05 quỹ đạo tham số HMM (SPEC Section 4.5 & 6.1):
    1. Quỹ đạo tần số cơ bản F0 contour (Bước nhảy State Means vs Đường cong trơn MLPG).
    2. Quỹ đạo tham số phổ Formant F1, F2 và các ranh giới từ/trạng thái.
    3. Spectrogram của âm thanh tổng hợp tham số HMM.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    time = tracks["time"]
    f0 = tracks["f0"]
    f0_step = tracks["f0_step"]
    f1 = tracks["f1"]
    f1_step = tracks["f1_step"]
    f2 = tracks["f2"]
    f2_step = tracks["f2_step"]
    tokens = tracks["tokens"]
    boundaries = tracks["word_boundaries"]

    # 1. Đồ thị F0 Contour
    ax_f0 = axes[0]
    ax_f0.step(time, f0_step, where="post", color="#e74c3c", linestyle=":", alpha=0.6, label="Kỳ vọng trạng thái tĩnh (State Mean $\\mu$)")
    ax_f0.plot(time, f0, color="#2980b9", linewidth=2.0, label="Quỹ đạo tối ưu MLPG (Có ràng buộc $\\Delta c_t$)")
    ax_f0.set_title("1. Quỹ đạo Tần số Cơ bản $F_0$ Contour (HMM Generation Mode)", fontsize=12, fontweight="bold")
    ax_f0.set_ylabel("$F_0$ (Hz)", fontsize=10)
    ax_f0.grid(True, linestyle=":", alpha=0.6)
    ax_f0.legend(loc="upper right")

    # Đánh dấu ranh giới từ
    for b_idx in boundaries[1:-1]:
        if b_idx < len(time):
            ax_f0.axvline(time[b_idx], color="gray", linestyle="--", alpha=0.5)

    # 2. Đồ thị Formant F1 & F2
    ax_fmt = axes[1]
    ax_fmt.step(time, f1_step, where="post", color="#e67e22", linestyle=":", alpha=0.5, label="F1 State Mean")
    ax_fmt.plot(time, f1, color="#d35400", linewidth=1.8, label="F1 Formant Track (MLPG)")
    ax_fmt.step(time, f2_step, where="post", color="#27ae60", linestyle=":", alpha=0.5, label="F2 State Mean")
    ax_fmt.plot(time, f2, color="#16a085", linewidth=1.8, label="F2 Formant Track (MLPG)")
    ax_fmt.set_title("2. Quỹ đạo Tham số Bao hình Phổ Formant ($F_1, F_2$)", fontsize=12, fontweight="bold")
    ax_fmt.set_ylabel("Tần số (Hz)", fontsize=10)
    ax_fmt.grid(True, linestyle=":", alpha=0.6)
    ax_fmt.legend(loc="upper right")

    # Gắn nhãn từ vựng
    for i in range(len(tokens)):
        if i < len(boundaries) - 1:
            start_f = boundaries[i]
            end_f = boundaries[i + 1]
            if start_f < len(time) and end_f <= len(time):
                mid_t = (time[start_f] + time[min(end_f - 1, len(time) - 1)]) / 2.0
                ax_fmt.text(mid_t, 2100, tokens[i], ha="center", fontsize=10, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.4))

    # 3. Spectrogram của âm thanh HMM
    ax_spec = axes[2]
    if y_audio is not None and len(y_audio) > 0:
        import librosa.display
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_audio, n_fft=512, hop_length=160)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, hop_length=160, x_axis="time", y_axis="linear", ax=ax_spec, cmap="magma")
        ax_spec.set_title("3. Spectrogram Âm thanh Tổng hợp Tham số HMM (Parametric Vocoder)", fontsize=12, fontweight="bold")
        ax_spec.set_ylim([0, 5000])
        fig.colorbar(img, ax=ax_spec, format="%+2.0f dB")
    else:
        ax_spec.text(0.5, 0.5, "Không có dữ liệu âm thanh", ha="center", va="center")

    plt.suptitle("Hình: Phân tích Quỹ đạo Tham số Âm học HMM-based Synthesis (Giai đoạn 5)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_fig)), exist_ok=True)
    plt.savefig(output_fig, dpi=300)
    plt.close()
