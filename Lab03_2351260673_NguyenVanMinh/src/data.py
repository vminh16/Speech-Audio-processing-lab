"""Liệt kê, nạp và kiểm tra tính hợp lệ của tập dữ liệu WAV."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .config import LABELS, MISMATCH_DIR, TEST_DIR, TRAIN_DIR


@dataclass(frozen=True)
class Utt:
    """Một utterance: đường dẫn file và nhãn."""

    path: Path
    label: str

    @property
    def name(self) -> str:
        return self.path.name


def list_split(split_dir: Path, labels: tuple[str, ...] = LABELS) -> list[Utt]:
    """Liệt kê ``<split_dir>/<label>/*.wav`` theo thứ tự ổn định (nhãn, tên file)."""
    utts: list[Utt] = []
    for label in labels:
        folder = Path(split_dir) / label
        utts.extend(Utt(p, label) for p in sorted(folder.glob("*.wav")))
    return utts


def group_by_label(utts: list[Utt]) -> dict[str, list[Utt]]:
    groups: dict[str, list[Utt]] = {}
    for u in utts:
        groups.setdefault(u.label, []).append(u)
    return groups


def load_splits() -> tuple[list[Utt], list[Utt], list[Utt]]:
    """Trả về (train, test, test_mismatch). ``test_mismatch`` có thể rỗng."""
    return list_split(TRAIN_DIR), list_split(TEST_DIR), list_split(MISMATCH_DIR)


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def validate_dataset(
    splits: dict[str, list[Utt]],
    sr: int = 16000,
    min_train: int = 1,
    min_test: int = 1,
) -> tuple[pd.DataFrame, list[str]]:
    """Kiểm tra định dạng từng file và rò rỉ train/test.

    Returns:
        (bảng thông tin từng file, danh sách lỗi dạng text). Danh sách lỗi rỗng nghĩa là hợp lệ.
    """
    rows = []
    for split, utts in splits.items():
        for u in utts:
            info = sf.info(str(u.path))
            y, _ = sf.read(str(u.path), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            rows.append(
                {
                    "split": split,
                    "label": u.label,
                    "file": u.name,
                    "sr": info.samplerate,
                    "channels": info.channels,
                    "subtype": info.subtype,
                    "duration_s": round(info.duration, 3),
                    "peak": round(float(np.max(np.abs(y))), 4) if y.size else 0.0,
                    "rms": round(float(np.sqrt(np.mean(y**2))), 4) if y.size else 0.0,
                    "sha1": _sha1(u.path),
                }
            )
    df = pd.DataFrame(rows)
    errors: list[str] = []
    if df.empty:
        return df, ["Không tìm thấy file WAV nào trong dataset/."]

    bad_fmt = df[(df.sr != sr) | (df.channels != 1) | (df.subtype != "PCM_16")]
    for _, r in bad_fmt.iterrows():
        errors.append(f"Sai định dạng ({r.sr} Hz, {r.channels} ch, {r.subtype}): {r.split}/{r.label}/{r.file}")

    for label in LABELS:
        n_train = int(((df.split == "train") & (df.label == label)).sum())
        n_test = int(((df.split == "test") & (df.label == label)).sum())
        if n_train < min_train:
            errors.append(f"Lớp '{label}' có {n_train} file train (< {min_train}).")
        if n_test < min_test:
            errors.append(f"Lớp '{label}' có {n_test} file test (< {min_test}).")

    train_hashes = set(df[df.split == "train"].sha1)
    for _, r in df[df.split != "train"].iterrows():
        if r.sha1 in train_hashes:
            errors.append(f"Rò rỉ dữ liệu: {r.split}/{r.label}/{r.file} trùng nội dung với một file train.")
    return df.drop(columns=["sha1"]), errors
