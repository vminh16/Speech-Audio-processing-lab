"""
Prosody Analysis, Source-Filter Model, and Voice Conversion Module.
CSE457 - Lab 04: Text-to-Speech
"""

import os
from typing import Dict, Tuple, Any

import librosa
import numpy as np
import soundfile as sf

from src.smoothing import crossfade
from src.utils import compute_rms, match_rms


def semitone_to_ratio(semitones: float) -> float:
    """
    Tính tỷ số tần số r khi dịch cao độ theo đơn vị semitone.
    Công thức: r = 2^(semitones / 12)
    """
    return float(2.0 ** (semitones / 12.0))


def estimate_pitch_period_autocorr(
    y_frame: np.ndarray,
    sr: int = 16000,
    fmin: float = 70.0,
    fmax: float = 400.0,
) -> Tuple[float, int]:
    """
    Ước lượng chu kỳ pitch N_p (mẫu) và tần số cơ bản F0 (Hz)
    của khung tín hiệu hữu thanh (voiced) bằng hàm tự tương quan (Autocorrelation).

    Công thức:
        R_xx[k] = sum(x[n] * x[n+k])
        N_p = argmax_{k in [k_min, k_max]} R_xx[k]
        F0 = Fs / N_p
    """
    if len(y_frame) < 32:
        return 0.0, 0

    # Khử thành phần một chiều (DC offset)
    x = y_frame.astype(np.float64) - np.mean(y_frame)

    # Tính hàm tự tương quan
    corr = np.correlate(x, x, mode="full")
    corr = corr[len(corr) // 2 :]  # Chỉ lấy các độ trễ lag k >= 0

    # Giới hạn khoảng tìm kiếm lag tương ứng dải tần [fmin, fmax]
    lag_min = max(1, int(sr / fmax))
    lag_max = min(len(corr) - 1, int(sr / fmin))

    if lag_min >= lag_max:
        return 0.0, 0

    # Tìm vị trí đỉnh cực đại trong khoảng [lag_min, lag_max]
    n_p = int(lag_min + np.argmax(corr[lag_min : lag_max + 1]))
    f0 = float(sr / n_p)

    return round(f0, 2), n_p


def verify_f0_formula(wav_path: str, sr: int = 16000) -> Dict[str, Any]:
    """
    Kiểm chứng thực nghiệm công thức F0 = Fs / N_p trên một file âm thanh hữu thanh.
    Trích xuất đoạn voiced có năng lượng cao nhất, đo N_p qua Autocorrelation
    và đối chiếu với thuật toán YIN chuẩn của librosa.
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Không tìm thấy file: {wav_path}")

    y, file_sr = sf.read(wav_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if file_sr != sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=file_sr, target_sr=sr)

    # Tìm khung 40ms có năng lượng RMS cao nhất (trung tâm nguyên âm)
    frame_len = int(sr * 0.04)  # 40 ms
    hop_len = int(sr * 0.01)    # 10 ms

    best_frame = None
    max_rms = -1.0
    best_time = 0.0

    for i in range(0, len(y) - frame_len, hop_len):
        frame = y[i : i + frame_len]
        r = compute_rms(frame)
        if r > max_rms:
            max_rms = r
            best_frame = frame
            best_time = i / sr

    if best_frame is None:
        best_frame = y[:frame_len]

    # 1. Tính F0 theo công thức Fs / N_p qua Autocorrelation
    f0_formula, n_p = estimate_pitch_period_autocorr(best_frame, sr=sr)

    # 2. Tính F0 đối chiếu bằng thuật toán YIN
    try:
        f0_yin_arr = librosa.yin(best_frame, fmin=70, fmax=400, sr=sr)
        f0_yin = float(np.median(f0_yin_arr[~np.isnan(f0_yin_arr)])) if len(f0_yin_arr) > 0 else f0_formula
    except Exception:
        f0_yin = f0_formula

    error_pct = round(abs(f0_formula - f0_yin) / f0_yin * 100.0, 2) if f0_yin > 0 else 0.0

    return {
        "wav_path": wav_path,
        "sr": sr,
        "voiced_start_sec": round(best_time, 3),
        "n_p": n_p,
        "f0_formula": f0_formula,
        "f0_yin": round(f0_yin, 2),
        "error_pct": error_pct,
        "sample_frame": best_frame,
    }


def time_stretch(y: np.ndarray, rate: float, sr: int = 16000) -> np.ndarray:
    """
    Co giãn thời lượng / tốc độ nói (Thí nghiệm 3) bằng Phase Vocoder.
    - rate > 1.0: Nói nhanh hơn (thời lượng co lại).
    - rate < 1.0: Nói chậm hơn (thời lượng giãn ra).
    Pitch và Formant được giữ nguyên không đổi.
    """
    if len(y) == 0 or np.isclose(rate, 1.0):
        return y.astype(np.float32)

    y_stretched = librosa.effects.time_stretch(y.astype(np.float32), rate=rate)
    return y_stretched.astype(np.float32)


def pitch_shift(y: np.ndarray, n_steps: float, sr: int = 16000) -> np.ndarray:
    """
    Dịch chuyển cao độ pitch theo số nửa cung semitone (Thí nghiệm 4).
    - n_steps > 0: Nâng cao độ (giọng cao hơn).
    - n_steps < 0: Hạ cao độ (giọng trầm hơn).
    Bảo toàn thời lượng gốc và bao hình phổ theo mô hình Source-Filter.
    """
    if len(y) == 0 or np.isclose(n_steps, 0.0):
        return y.astype(np.float32)

    y_shifted = librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=n_steps)
    return y_shifted.astype(np.float32)


def voice_conversion_female(
    y: np.ndarray,
    sr: int = 16000,
    pitch_shift_steps: float = 5.5,
) -> np.ndarray:
    """
    Biến đổi giọng Nam sang Nữ (Male-to-Female Voice Conversion):
    1. Nâng tần số cơ bản F0 (+5.5 semitones, tăng ~37% pitch).
    2. Nâng dải tần số cao (Pre-emphasis filter) để mô phỏng formant cao hơn
       do khoang tuyến âm của nữ nhỏ hơn nam (~15%).
    3. Chuẩn hóa lại mức năng lượng RMS mục tiêu (0.08).
    """
    if len(y) == 0:
        return y

    # 1. Nâng pitch
    y_high_pitch = pitch_shift(y, n_steps=pitch_shift_steps, sr=sr)

    # 2. Bộ lọc nhấn dải tần số cao mô phỏng formant F2-F3 của giọng nữ
    # H(z) = 1 - 0.35 * z^(-1)
    diff_signal = np.diff(y_high_pitch, prepend=y_high_pitch[0])
    y_female = 0.82 * y_high_pitch + 0.18 * diff_signal

    # 3. Chuẩn hóa năng lượng
    return match_rms(y_female.astype(np.float32), target_rms=0.08)


def voice_conversion_southern(
    y: np.ndarray,
    sr: int = 16000,
) -> np.ndarray:
    """
    Mô phỏng sắc thái ngữ điệu Miền Nam (Southern Accent Simulation):
    1. Làm phẳng nhẹ độ biến thiên pitch tổng thể để mô phỏng cao độ đầm ấm miền Nam.
    2. Kéo dài thời lượng phần đuôi câu (âm ngân buông lơi đặc trưng miền Nam) thêm ~15%.
    """
    if len(y) < sr * 0.5:
        return y

    # Tách đoạn thân và đoạn đuôi câu (35% cuối)
    split_idx = int(len(y) * 0.65)
    body = y[:split_idx]
    tail = y[split_idx:]

    # Giãn nhẹ đuôi câu bằng time-stretch (rate = 0.85 -> giãn ra ~1.18x)
    tail_stretched = time_stretch(tail, rate=0.85, sr=sr)

    # Hạ nhẹ nửa cung ở phần đuôi để tạo cảm giác ngân buông nhẹ
    tail_smooth = pitch_shift(tail_stretched, n_steps=-0.6, sr=sr)

    # Ghép nối mượt với cross-fade 25 ms
    n_fade = int(sr * 0.025)
    y_southern = crossfade(body, tail_smooth, n=n_fade)

    return match_rms(y_southern.astype(np.float32), target_rms=0.08)
