"""Hằng số, đường dẫn và siêu tham số dùng chung cho toàn bộ pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
MISMATCH_DIR = DATASET_DIR / "test_mismatch"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
LEXICON_PATH = ROOT / "lexicon.txt"
CORPUS_PATH = ROOT / "corpus.txt"

LABELS: tuple[str, ...] = ("mo", "dong", "len", "xuong", "dung")
DISPLAY: dict[str, str] = {
    "mo": "mở",
    "dong": "đóng",
    "len": "lên",
    "xuong": "xuống",
    "dung": "dừng",
}

BOS = "<s>"  # ký hiệu bắt đầu câu trong mô hình bigram


@dataclass(frozen=True)
class FeatureConfig:
    """Tham số tiền xử lý và trích xuất đặc trưng (hashable để dùng làm khóa cache)."""

    sr: int = 16000
    top_db: float = 30.0
    preemph: float = 0.97
    n_fft: int = 512
    win_length: int = 400  # 25 ms
    hop_length: int = 160  # 10 ms
    n_mels: int = 26
    fmin: float = 20.0
    fmax: float = 8000.0
    n_mfcc: int = 13
    use_delta: bool = True
    delta_width: int = 9
    cmvn: bool = True

    @property
    def dim(self) -> int:
        return self.n_mfcc * (3 if self.use_delta else 1)


@dataclass(frozen=True)
class HMMConfig:
    """Tham số mô hình Whole-word Left-to-Right Gaussian HMM."""

    n_states: int = 5
    cov_type: str = "diag"
    n_iter: int = 80
    tol: float = 1e-3
    min_covar: float = 1e-3
    self_loop: float = 0.7
    init: str = "uniform"  # "uniform" (chia đều theo thời gian) | "kmeans" (init_params="mc")
    random_state: int = 0


@dataclass(frozen=True)
class LMConfig:
    """Tham số mô hình ngôn ngữ bigram và re-ranking."""

    k: float = 0.5
    lambdas: tuple[float, ...] = (0.0, 0.1, 0.3, 1.0)
