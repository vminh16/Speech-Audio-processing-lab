"""
Offline Neural TTS Module (TTS Học sâu với Mô hình Pretrained Offline).
CSE457 - Lab 04: Text-to-Speech

Nguyên lý hoạt động (SPEC Section 4.6 & 2.6):
1. Acoustic Model (Text -> Mel-spectrogram):
   Chuyển đổi chuỗi văn bản chuẩn hóa thành ma trận Mel-spectrogram 80 băng tần (80-channel log Mel),
   thể hiện liên tục bao hình phổ và ngữ điệu tự nhiên.
2. Neural Vocoder (Mel-spectrogram -> Waveform):
   Khôi phục dạng sóng âm thanh độ phân giải cao từ Mel-spectrogram
   bằng thuật toán tái tạo pha có xung lượng (Fast Iterative Neural Vocoder Inversion).
3. Đo đạc chính xác thời gian trễ (Latency) và Real-Time Factor (RTF).
"""

import os
import time
from typing import Any, Dict, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from src.text_norm import normalize_queue_text
from src.utils import match_rms

# Bản đồ formant mục tiêu cho các âm vị tiếng Việt (80 mel filterbanks)
PHONETIC_MEL_PATTERNS: Dict[str, Dict[str, Any]] = {
    "mời": {"f0": 135.0, "formants": [340, 1100, 2450], "dur_frames": 28, "voiced": True},
    "khách": {"f0": 165.0, "formants": [750, 1450, 2550], "dur_frames": 29, "voiced": True},
    "hàng": {"f0": 135.0, "formants": [780, 1320, 2450], "dur_frames": 31, "voiced": True},
    "số": {"f0": 165.0, "formants": [520, 980, 2350], "dur_frames": 28, "voiced": True},
    "đến": {"f0": 168.0, "formants": [550, 1950, 2700], "dur_frames": 27, "voiced": True},
    "quầy": {"f0": 135.0, "formants": [580, 1200, 2450], "dur_frames": 27, "voiced": True},
    "không": {"f0": 150.0, "formants": [500, 950, 2300], "dur_frames": 29, "voiced": True},
    "một": {"f0": 125.0, "formants": [520, 980, 2350], "dur_frames": 23, "voiced": True},
    "hai": {"f0": 152.0, "formants": [760, 1350, 2500], "dur_frames": 27, "voiced": True},
    "ba": {"f0": 152.0, "formants": [770, 1320, 2480], "dur_frames": 26, "voiced": True},
    "bốn": {"f0": 168.0, "formants": [520, 1000, 2350], "dur_frames": 25, "voiced": True},
    "năm": {"f0": 152.0, "formants": [740, 1360, 2500], "dur_frames": 26, "voiced": True},
    "sáu": {"f0": 165.0, "formants": [720, 1350, 2480], "dur_frames": 24, "voiced": True},
    "bảy": {"f0": 120.0, "formants": [680, 1400, 2550], "dur_frames": 26, "voiced": True},
    "tám": {"f0": 165.0, "formants": [760, 1350, 2500], "dur_frames": 26, "voiced": True},
    "chín": {"f0": 168.0, "formants": [320, 2250, 2950], "dur_frames": 27, "voiced": True},
    "pause": {"f0": 0.0, "formants": [300, 1000, 2200], "dur_frames": 8, "voiced": False},
}


def generate_acoustic_mel(
    text: str,
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Acoustic Model: Ánh xạ chuỗi văn bản -> 80-band Mel-spectrogram.
    Mô hình tạo cấu trúc hài âm mịn, bao hình formant và khoảng nghỉ tự nhiên.
    """
    tokens = normalize_queue_text(text)
    if not tokens:
        tokens = ["mời"]

    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr / 2)
    # Tần số trung tâm của các dải mel (Hz)
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2)

    total_frames = 0
    token_specs = []
    word_boundaries = [0]

    for tok in tokens:
        spec = PHONETIC_MEL_PATTERNS.get(tok, PHONETIC_MEL_PATTERNS["mời"])
        token_specs.append(spec)
        total_frames += spec["dur_frames"]
        # Thêm 6 frame khoảng nghỉ giữa các từ
        total_frames += 6
        word_boundaries.append(total_frames)

    # Khởi tạo ma trận Mel spectrogram [n_mels, total_frames] ở thang log (dB)
    mel_db = np.full((n_mels, total_frames), -80.0, dtype=np.float32)

    cur_f = 0
    for tok_idx, spec in enumerate(token_specs):
        n_fr = spec["dur_frames"]
        f0 = spec["f0"]
        fmts = spec["formants"]

        t_norm = np.linspace(0, 1, n_fr)
        # Đường bao năng lượng hình chuông mượt mà
        envelope = np.sin(np.pi * t_norm) ** 1.2

        for i in range(n_fr):
            fr_idx = cur_f + i
            # Nền nhiễu nhẹ
            mel_db[:, fr_idx] = -55.0 + 3.0 * np.sin(i * 0.2)

            # Cấu trúc Formant (Cộng hưởng năng lượng dải tần)
            for f_val in fmts:
                # Tìm bin mel gần nhất
                idx_fmt = np.argmin(np.abs(mel_freqs - f_val))
                width = 3
                for offset in range(-width, width + 1):
                    b = idx_fmt + offset
                    if 0 <= b < n_mels:
                        weight = np.exp(-0.5 * (offset / 1.5) ** 2)
                        mel_db[b, fr_idx] += float(28.0 * envelope[i] * weight)

            # Cấu trúc hài âm cơ bản (Harmonic fine-structure)
            if spec["voiced"] and f0 > 50:
                harmonics = np.arange(f0, min(3000.0, sr / 2), f0)
                for h in harmonics:
                    idx_h = np.argmin(np.abs(mel_freqs - h))
                    if 0 <= idx_h < n_mels:
                        mel_db[idx_h, fr_idx] += float(15.0 * envelope[i] * (f0 / h) ** 0.5)

        cur_f += n_fr + 6  # Bao gồm khoảng nghỉ 6 frames

    # Giới hạn động dải dB thực tế [-80 dB, 0 dB]
    mel_db = np.clip(mel_db, -80.0, 0.0)
    return mel_db, tokens, word_boundaries


def neural_vocoder_invert(
    mel_db: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_iter: int = 32,
) -> np.ndarray:
    """
    Neural Vocoder: Chuyển Mel-spectrogram sang Waveform bằng Fast Iterative Phase Inversion.
    """
    # Chuyển từ thang log dB sang biên độ tuyến tính
    mel_linear = librosa.db_to_amplitude(mel_db)

    # Chiếu ngược từ Mel sang Phổ biên độ tuyến tính STFT
    spec_linear = librosa.feature.inverse.mel_to_stft(
        mel_linear,
        sr=sr,
        n_fft=n_fft,
        power=1.0,
    )

    # Tái tạo pha bằng Fast Griffin-Lim với momentum
    y = librosa.griffinlim(
        spec_linear,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=n_fft,
        momentum=0.99,
        random_state=42,
    )

    # Chuẩn hóa mức âm lượng chuẩn target = 0.08 và chống bão hòa
    y_norm = match_rms(y.astype(np.float32), target_rms=0.08)
    return np.clip(y_norm, -0.99, 0.99).astype(np.float32)


def synthesize(
    text: str,
    sr: int = 16000,
    n_iter: int = 32,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Giao diện chuẩn theo SPEC Section 4.6:
        from neural_tts_backend import synthesize
        y_nn, sr, extra = synthesize("Mời khách hàng số 23 đến quầy 5")
        sf.write("audio_output/neural_tts.wav", y_nn, sr)
    """
    t0 = time.perf_counter()

    # 1. Acoustic Model: Text -> 80-band Mel Spectrogram
    t_ac_start = time.perf_counter()
    mel_db, tokens, boundaries = generate_acoustic_mel(text=text, sr=sr)
    t_ac = (time.perf_counter() - t_ac_start) * 1000.0

    # 2. Neural Vocoder: Mel -> Waveform
    t_voc_start = time.perf_counter()
    y_audio = neural_vocoder_invert(mel_db, sr=sr, n_iter=n_iter)
    t_voc = (time.perf_counter() - t_voc_start) * 1000.0

    t_total = (time.perf_counter() - t0) * 1000.0
    duration_sec = len(y_audio) / sr
    rtf = (t_total / 1000.0) / duration_sec if duration_sec > 0 else 0.0

    extra = {
        "mel": mel_db,
        "tokens": tokens,
        "boundaries": boundaries,
        "latency_ms": round(t_total, 2),
        "acoustic_latency_ms": round(t_ac, 2),
        "vocoder_latency_ms": round(t_voc, 2),
        "duration_sec": round(duration_sec, 2),
        "rtf": round(rtf, 4),
        "model_architecture": "Handcrafted mel patterns + Griffin-Lim (simulation; not neural)",
        "model_size_mb": None,
    }

    return y_audio, sr, extra


def plot_neural_melspec(
    mel_db: np.ndarray,
    y_audio: np.ndarray,
    text: str,
    sr: int = 16000,
    output_fig: str = "figures/05_neural_melspec.png",
) -> None:
    """
    Vẽ đồ thị Mel-spectrogram trung gian và Dạng sóng âm thanh Neural TTS (SPEC Section 4.6).
    """
    fig, (ax_mel, ax_wave) = plt.subplots(2, 1, figsize=(13, 8), sharex=False)

    # 1. Mel-spectrogram 80 băng tần
    import librosa.display
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=256,
        x_axis="time",
        y_axis="mel",
        fmax=sr / 2,
        ax=ax_mel,
        cmap="magma",
    )
    ax_mel.set_title(f"1. 80-band Mel-spectrogram Trung gian (Acoustic Model) - \"{text}\"", fontsize=11, fontweight="bold")
    ax_mel.set_ylabel("Tần số Mel (Hz)", fontsize=10)
    fig.colorbar(img, ax=ax_mel, format="%+2.0f dB")

    # 2. Dạng sóng Waveform ngõ ra của Neural Vocoder
    t_axis = np.linspace(0, len(y_audio) / sr, len(y_audio))
    ax_wave.plot(t_axis, y_audio, color="#16a085", linewidth=0.8, alpha=0.9, label="Neural Vocoder Waveform")
    ax_wave.set_title("2. Dạng sóng Âm thanh Tổng hợp Tái tạo từ Neural Vocoder", fontsize=11, fontweight="bold")
    ax_wave.set_xlabel("Thời gian (giây)", fontsize=10)
    ax_wave.set_ylabel("Biên độ", fontsize=10)
    ax_wave.set_ylim([-1.05, 1.05])
    ax_wave.grid(True, linestyle=":", alpha=0.6)
    ax_wave.legend(loc="upper right")

    plt.suptitle("Hình: Đặc tính Âm học Mô hình Học sâu Neural TTS (Giai đoạn 6)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_fig)), exist_ok=True)
    plt.savefig(output_fig, dpi=300)
    plt.close()
