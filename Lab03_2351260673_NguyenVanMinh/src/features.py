"""Tiền xử lý và trích xuất đặc trưng MFCC (+Δ +Δ²) chuẩn hóa CMVN."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np

from .config import FeatureConfig
from .data import Utt
from .noise import add_white_noise

FeatSet = list[tuple[str, np.ndarray]]  # [(label, ma trận T x D), ...]


def load_wav(path: Path | str, cfg: FeatureConfig) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=cfg.sr, mono=True)
    return y


def trim_bounds(y: np.ndarray, cfg: FeatureConfig) -> tuple[int, int]:
    """Chỉ số mẫu ``[start, end)`` của vùng được giữ lại; toàn bộ tín hiệu nếu phần còn lại ngắn hơn một frame."""
    _, (start, end) = librosa.effects.trim(y, top_db=cfg.top_db)
    return (int(start), int(end)) if end - start >= cfg.win_length else (0, len(y))


def trim_silence(y: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Cắt khoảng lặng hai đầu theo ``trim_bounds``."""
    start, end = trim_bounds(y, cfg)
    return y[start:end]


def _odd_width(width: int, n_frames: int) -> int:
    width = min(width, n_frames if n_frames % 2 == 1 else n_frames - 1)
    return max(width, 3)


def _mfcc(y_pre: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    return librosa.feature.mfcc(
        y=y_pre,
        sr=cfg.sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        window="hamming",
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )


def _stack_and_normalize(mfcc: np.ndarray, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    """``(đặc trưng trước CMVN, đặc trưng cuối cùng)``, cả hai có dạng ``T x cfg.dim``."""
    blocks = [mfcc]
    if cfg.use_delta:
        width = _odd_width(cfg.delta_width, mfcc.shape[1])
        blocks.append(librosa.feature.delta(mfcc, width=width, order=1, mode="nearest"))
        blocks.append(librosa.feature.delta(mfcc, width=width, order=2, mode="nearest"))
    raw = np.vstack(blocks).T.astype(np.float64)
    if not cfg.cmvn:
        return raw, raw
    return raw, (raw - raw.mean(axis=0)) / (raw.std(axis=0) + 1e-8)


def features_from_signal(y: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Waveform -> ma trận đặc trưng ``T x cfg.dim``.

    Các bước: trim -> pre-emphasis -> MFCC (Hamming, 25/10 ms, 26 mel) -> Δ, Δ² -> CMVN theo utterance.
    """
    y_pre = librosa.effects.preemphasis(trim_silence(y, cfg), coef=cfg.preemph)
    return _stack_and_normalize(_mfcc(y_pre, cfg), cfg)[1]


def preprocessing_stages(y: np.ndarray, cfg: FeatureConfig) -> dict:
    """Kết quả trung gian của từng bước tiền xử lý, dùng để trực quan hóa (``features_from_signal`` cho cùng đặc trưng cuối)."""
    start, end = trim_bounds(y, cfg)
    trimmed = y[start:end]
    emphasized = librosa.effects.preemphasis(trimmed, coef=cfg.preemph)
    mel = librosa.feature.melspectrogram(
        y=emphasized, sr=cfg.sr, n_fft=cfg.n_fft, win_length=cfg.win_length, hop_length=cfg.hop_length,
        window="hamming", n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax,
    )
    raw, final = _stack_and_normalize(_mfcc(emphasized, cfg), cfg)
    return {
        "raw": y, "bounds": (start, end), "trimmed": trimmed, "emphasized": emphasized,
        "log_mel": librosa.power_to_db(mel), "feat_raw": raw, "feat": final,
    }


def extract_feat(path: Path | str, cfg: FeatureConfig | None = None, use_delta: bool | None = None) -> np.ndarray:
    """Đặc trưng của một file WAV. ``use_delta`` (nếu truyền) ghi đè ``cfg.use_delta``."""
    cfg = cfg or FeatureConfig()
    if use_delta is not None and use_delta != cfg.use_delta:
        cfg = replace(cfg, use_delta=use_delta)
    return features_from_signal(load_wav(path, cfg), cfg)


@lru_cache(maxsize=None)
def _cached_feat(path: str, cfg: FeatureConfig) -> np.ndarray:
    feat = extract_feat(path, cfg)
    feat.setflags(write=False)  # cache dùng chung -> chặn ghi nhầm
    return feat


def build_featset(
    utts: list[Utt],
    cfg: FeatureConfig,
    snr_db: float | None = None,
    seed: int = 0,
) -> FeatSet:
    """Đặc trưng cho cả danh sách utterance; thêm nhiễu trắng nếu ``snr_db`` được cho."""
    if snr_db is None:
        return [(u.label, _cached_feat(str(u.path), cfg)) for u in utts]
    rng = np.random.default_rng(seed)
    out: FeatSet = []
    for u in utts:
        y = add_white_noise(load_wav(u.path, cfg), snr_db, rng)
        out.append((u.label, features_from_signal(y, cfg)))
    return out


def by_label(featset: FeatSet) -> dict[str, list[np.ndarray]]:
    groups: dict[str, list[np.ndarray]] = {}
    for label, feat in featset:
        groups.setdefault(label, []).append(feat)
    return groups
