"""
Boundary Smoothing and Cross-Fade Module for Concatenative TTS.
CSE457 - Lab 04: Text-to-Speech
"""

from typing import List, Tuple
import numpy as np


def crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """
    Trộn tuyến tính (Linear Cross-fade) hai mảng tín hiệu âm thanh với vùng chồng lấn n mẫu.

    Công thức:
        y[n] = (1 - alpha[n]) * a[-n:] + alpha[n] * b[:n], với alpha chạy từ 0 đến 1.
    """
    n = max(0, min(n, len(a), len(b)))
    if n == 0:
        return np.concatenate([a, b])

    alpha = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float32)
    mid = (1.0 - alpha) * a[-n:] + alpha * b[:n]
    return np.concatenate([a[:-n], mid, b[n:]])


def crossfade_sequence(
    units: List[np.ndarray],
    sr: int = 16000,
    xfade_ms: float = 15.0,
) -> Tuple[np.ndarray, List[float]]:
    """
    Ghép nối một chuỗi các unit âm thanh với độ dài cross-fade xác định.

    Trả về:
        (y_waveform, boundary_timestamps_sec):
        - y_waveform: Mảng âm thanh hoàn chỉnh.
        - boundary_timestamps_sec: Danh sách thời điểm (giây) tại tâm của mỗi vùng biên nối.
    """
    if not units:
        return np.array([], dtype=np.float32), []
    if len(units) == 1:
        return units[0].astype(np.float32), []

    n = int(sr * xfade_ms / 1000.0)
    current_y = units[0].astype(np.float32)
    boundaries = []

    for i in range(1, len(units)):
        next_u = units[i].astype(np.float32)
        actual_n = max(0, min(n, len(current_y), len(next_u)))

        # Vị trí biên nối là tâm của đoạn cross-fade
        boundary_idx = len(current_y) - actual_n / 2.0
        boundaries.append(float(boundary_idx / sr))

        current_y = crossfade(current_y, next_u, actual_n)

    return current_y, boundaries


def compute_spectral_discontinuity(
    a_tail: np.ndarray,
    b_head: np.ndarray,
    sr: int = 16000,
    window_ms: float = 25.0,
    nfft: int = 512,
) -> float:
    """
    Tính chỉ số gián đoạn phổ D_spec (Spectral Discontinuity) giữa đuôi unit a và đầu unit b.

    Công thức:
        D_spec = (1/K) * sum( (L_a[k] - L_b[k])^2 )
        với L = 20 * log10( max(|RFFT(x * w)|, 1e-8) ) trên cửa sổ Hanning 25ms.
    """
    n = int(sr * window_ms / 1000.0)
    if len(a_tail) < n or len(b_head) < n:
        n = min(len(a_tail), len(b_head))
        if n < 4:
            return 0.0

    wa = np.hanning(n)
    wb = np.hanning(n)

    fft_a = np.fft.rfft(a_tail[-n:] * wa, n=nfft)
    fft_b = np.fft.rfft(b_head[:n] * wb, n=nfft)

    la = 20.0 * np.log10(np.maximum(np.abs(fft_a), 1e-8))
    lb = 20.0 * np.log10(np.maximum(np.abs(fft_b), 1e-8))

    return float(np.mean((la - lb) ** 2))


def evaluate_crossfade_experiment(
    units: List[np.ndarray],
    sr: int = 16000,
    xfade_levels: List[float] = None,
    window_ms: float = 25.0,
) -> List[dict]:
    """
    Thực hiện Thí nghiệm 1: Khảo sát ảnh hưởng của độ dài Cross-fade (0, 5, 15, 30 ms)
    lên chỉ số gián đoạn phổ D_spec tại các biên nối đơn vị âm thanh.
    """
    if xfade_levels is None:
        xfade_levels = [0.0, 5.0, 15.0, 30.0]

    results = []
    window_samples = int(sr * window_ms / 1000.0)

    for xfade_ms in xfade_levels:
        y_out, boundaries = crossfade_sequence(units, sr=sr, xfade_ms=xfade_ms)
        d_spec_list = []

        # Đo D_spec quanh từng điểm biên nối
        for b_time in boundaries:
            b_idx = int(b_time * sr)
            if b_idx >= window_samples and (b_idx + window_samples) <= len(y_out):
                left_segment = y_out[b_idx - window_samples : b_idx]
                right_segment = y_out[b_idx : b_idx + window_samples]
                d = compute_spectral_discontinuity(left_segment, right_segment, sr=sr, window_ms=window_ms)
                d_spec_list.append(d)

        avg_d_spec = float(np.mean(d_spec_list)) if d_spec_list else 0.0

        results.append({
            "xfade_ms": xfade_ms,
            "mean_d_spec": round(avg_d_spec, 4),
            "num_boundaries": len(d_spec_list),
            "audio_length_sec": round(len(y_out) / sr, 3),
        })

    return results

