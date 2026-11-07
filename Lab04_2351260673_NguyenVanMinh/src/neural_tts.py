"""Offline Piper/VITS pretrained Vietnamese inference; no synthetic fallback."""
import os
import time
from functools import lru_cache
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import numpy as np
from piper import PiperVoice
from src.text_norm import normalize_queue_text

MODEL = Path(__file__).resolve().parents[1] / 'data/models/piper/vi_VN-vais1000-medium.onnx'

@lru_cache(maxsize=1)
def load_voice():
    if not MODEL.exists():
        raise FileNotFoundError('Run scripts/download_model.py before offline inference')
    return PiperVoice.load(str(MODEL))

def synthesize(text, sr=16000, n_iter=None):
    start = time.perf_counter()
    voice = load_voice()
    tokens = normalize_queue_text(text)
    if not tokens:
        raise ValueError('Empty text')
    chunks = list(voice.synthesize(' '.join(tokens)))
    y = np.concatenate([c.audio_float_array for c in chunks])
    native_sr = chunks[0].sample_rate
    y = librosa.resample(y, orig_sr=native_sr, target_sr=sr)
    elapsed = (time.perf_counter()-start)*1000
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=1024, hop_length=256), ref=np.max)
    return y.astype(np.float32), sr, dict(mel=mel, mel_source='analysis of output waveform; VITS exposes no intermediate mel', latency_ms=elapsed, acoustic_latency_ms=None, vocoder_latency_ms=None, duration_sec=len(y)/sr, rtf=elapsed/1000/(len(y)/sr), model_architecture='Piper VITS pretrained vi_VN-vais1000-medium v1.0.0', model_size_mb=MODEL.stat().st_size/1024**2)

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
    ax_mel.set_title(f"1. 80-band Mel-spectrogram tính từ waveform (không phải latent nội bộ) - \"{text}\"", fontsize=11, fontweight="bold")
    ax_mel.set_ylabel("Tần số Mel (Hz)", fontsize=10)
    fig.colorbar(img, ax=ax_mel, format="%+2.0f dB")

    # 2. Dạng sóng Waveform ngõ ra của Neural Vocoder
    t_axis = np.linspace(0, len(y_audio) / sr, len(y_audio))
    ax_wave.plot(t_axis, y_audio, color="#16a085", linewidth=0.8, alpha=0.9, label="Piper VITS Waveform")
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
