"""
Concatenative Synthesis and Unit Inventory Loading.
CSE457 - Lab 04: Text-to-Speech
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from src.smoothing import crossfade_sequence

# Bảng ánh xạ token tiếng Việt có dấu sang tên file không dấu trong data/units/
TOKEN_TO_FILENAME: Dict[str, str] = {
    "mời": "moi",
    "khách": "khach",
    "hàng": "hang",
    "số": "so",
    "không": "khong",
    "một": "mot",
    "hai": "hai",
    "ba": "ba",
    "bốn": "bon",
    "năm": "nam",
    "sáu": "sau",
    "bảy": "bay",
    "tám": "tam",
    "chín": "chin",
    "đến": "den",
    "quầy": "quay",
}


def get_unit_filename(token: str, candidate_id: str = "01") -> str:
    """Trả về tên file WAV tương ứng với token và candidate_id."""
    clean_tok = token.strip().lower()
    base_name = TOKEN_TO_FILENAME.get(clean_tok, clean_tok)
    return f"{base_name}_{candidate_id}.wav"


def load_unit(token: str, candidate_id: str = "01", units_dir: str = "data/units") -> Tuple[np.ndarray, int]:
    """
    Nạp mảng waveform và samplerate của một đơn vị âm thanh từ thư mục units_dir.
    """
    fname = get_unit_filename(token, candidate_id)
    file_path = os.path.join(units_dir, fname)

    # Thử tìm file với tên gốc nếu chưa thấy
    if not os.path.exists(file_path):
        alt_path = os.path.join(units_dir, f"{token.lower()}_{candidate_id}.wav")
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file unit '{fname}' trong {units_dir}")

    data, sr = sf.read(file_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    return data.astype(np.float32), int(sr)


def build_baseline_sequence(
    tokens: List[str],
    units_dir: str = "data/units",
    candidate_id: str = "01",
) -> List[np.ndarray]:
    """
    Nạp danh sách các mảng waveform tương ứng với chuỗi token (cố định candidate_id).
    """
    units = []
    for tok in tokens:
        y, _ = load_unit(tok, candidate_id=candidate_id, units_dir=units_dir)
        units.append(y)
    return units


def synthesize_baseline(
    tokens: List[str],
    units_dir: str = "data/units",
    sr: int = 16000,
) -> Tuple[np.ndarray, List[float]]:
    """
    Tổng hợp tiếng nói ghép nối cơ sở (Baseline Concatenation):
    - Chọn cố định candidate '_01'.
    - Ghép nối trực tiếp với xfade_ms = 0.
    """
    units = build_baseline_sequence(tokens, units_dir=units_dir, candidate_id="01")
    return crossfade_sequence(units, sr=sr, xfade_ms=0.0)
