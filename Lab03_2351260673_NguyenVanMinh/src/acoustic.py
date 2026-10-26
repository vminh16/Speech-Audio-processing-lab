"""Mô hình âm học: Whole-word Left-to-Right Gaussian HMM (khởi tạo, huấn luyện, kiểm tra, lưu/nạp)."""
from __future__ import annotations

import logging
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from hmmlearn.hmm import GaussianHMM

from .config import HMMConfig
from .utils import Timer

Models = dict[str, GaussianHMM]

# hmmlearn báo "not converging" cho cả dao động ~1e-7 do sai số số học; trạng thái hội tụ được kiểm tra riêng
# qua ``monitor_.converged`` nên tắt log này để không làm nhiễu đầu ra.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


def left_right_matrices(n_states: int, self_loop: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """``(startprob, transmat)`` của topology trái-sang-phải: chỉ self-loop hoặc sang trạng thái kế tiếp."""
    startprob = np.zeros(n_states)
    startprob[0] = 1.0
    transmat = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        transmat[i, i] = self_loop
        transmat[i, i + 1] = 1.0 - self_loop
    transmat[-1, -1] = 1.0
    return startprob, transmat


def make_lr_hmm(
    n_states: int = 5,
    cov_type: str = "diag",
    n_iter: int = 80,
    tol: float = 1e-3,
    min_covar: float = 1e-3,
    self_loop: float = 0.7,
    init: str = "uniform",
    random_state: int = 0,
) -> GaussianHMM:
    """Tạo GaussianHMM trái-sang-phải với ``startprob_`` và ``transmat_`` đã đặt sẵn.

    ``params="tmc"`` cho phép EM cập nhật A, mean, cov; ``startprob`` được giữ cố định.
    Các phần tử bằng 0 của A không bao giờ bị phá vỡ vì Baum-Welch nhân với giá trị khởi tạo.
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=cov_type,
        n_iter=n_iter,
        tol=tol,
        min_covar=min_covar,
        init_params="mc" if init == "kmeans" else "",
        params="tmc",
        random_state=random_state,
    )
    model.startprob_, model.transmat_ = left_right_matrices(n_states, self_loop)
    return model


def uniform_segment_init(model: GaussianHMM, seqs: list[np.ndarray]) -> None:
    """Khởi tạo mean/cov mỗi trạng thái từ việc chia đều mỗi utterance thành N đoạn theo thời gian.

    Phù hợp với topology trái-sang-phải hơn K-means ngẫu nhiên (K-means không gắn trạng thái với thứ tự thời gian).
    """
    n = model.n_components
    dim = seqs[0].shape[1]
    per_state: list[list[np.ndarray]] = [[] for _ in range(n)]
    for seq in seqs:
        for s, idx in enumerate(np.array_split(np.arange(len(seq)), n)):
            if len(idx):
                per_state[s].append(seq[idx])
    all_frames = np.vstack(seqs)
    g_mean, g_var = all_frames.mean(axis=0), all_frames.var(axis=0) + model.min_covar

    means = np.zeros((n, dim))
    covars: list[np.ndarray] = []
    for s in range(n):
        frames = np.vstack(per_state[s]) if per_state[s] else all_frames
        means[s] = frames.mean(axis=0)
        if model.covariance_type == "diag":
            covars.append(frames.var(axis=0) + model.min_covar if len(frames) > 1 else g_var)
        else:  # "full": co rút về đường chéo toàn cục để khả nghịch khi ít frame
            cov = np.cov(frames.T) if len(frames) > 1 else np.diag(g_var)
            covars.append(0.7 * cov + 0.3 * np.diag(g_var))
    model.means_ = means
    model.covars_ = np.array(covars)


@dataclass
class TrainInfo:
    label: str
    n_seqs: int
    n_frames: int
    converged: bool
    n_iter_run: int
    train_ll_per_frame: float
    train_time_s: float


def fit_word_model(seqs: list[np.ndarray], cfg: HMMConfig) -> tuple[GaussianHMM, TrainInfo]:
    """Huấn luyện (Baum-Welch) một HMM cho một từ từ danh sách ma trận đặc trưng."""
    if not seqs:
        raise ValueError("Không có dữ liệu huấn luyện.")
    model = make_lr_hmm(
        cfg.n_states, cfg.cov_type, cfg.n_iter, cfg.tol, cfg.min_covar, cfg.self_loop, cfg.init, cfg.random_state
    )
    short = [len(s) for s in seqs if len(s) < cfg.n_states]
    if short:
        warnings.warn(f"{len(short)} utterance ngắn hơn số trạng thái ({cfg.n_states}) - không thể đi hết các trạng thái.")
    if cfg.init == "uniform":
        uniform_segment_init(model, seqs)
    X, lengths = np.vstack(seqs), [len(s) for s in seqs]
    try:
        with Timer() as t, warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # log(0) trên các phần tử 0 của A là chủ ý
            model.fit(X, lengths)
        train_ll = float(model.score(X, lengths) / X.shape[0])
    except ValueError as exc:  # hmmlearn báo NaN trong transmat_ khi một trạng thái không nhận được frame nào
        raise RuntimeError(
            f"EM suy biến (init='{cfg.init}', random_state={cfg.random_state}): có trạng thái không nhận frame nào. "
            "Thử init='uniform' hoặc giảm n_states."
        ) from exc
    info = TrainInfo(
        label="",
        n_seqs=len(seqs),
        n_frames=int(X.shape[0]),
        converged=bool(model.monitor_.converged),
        n_iter_run=len(model.monitor_.history),
        train_ll_per_frame=train_ll,
        train_time_s=t.elapsed,
    )
    return model, info


def train_models(groups: dict[str, list[np.ndarray]], cfg: HMMConfig) -> tuple[Models, dict[str, TrainInfo]]:
    """Huấn luyện một HMM cho mỗi nhãn trong ``groups``."""
    models: Models = {}
    infos: dict[str, TrainInfo] = {}
    for label, seqs in groups.items():
        models[label], info = fit_word_model(seqs, cfg)
        info.label = label
        infos[label] = info
    return models, infos


def check_topology(model: GaussianHMM, atol: float = 1e-12) -> dict[str, bool]:
    """Kiểm tra hậu huấn luyện: cấu trúc trái-sang-phải còn nguyên và các hàng của A chuẩn hóa."""
    A, pi = model.transmat_, model.startprob_
    n = A.shape[0]
    upper_band = np.triu(np.ones((n, n), dtype=bool), k=2)  # nhảy cóc >= 2 trạng thái
    return {
        "converged": bool(model.monitor_.converged),
        "no_backward": bool(np.all(np.tril(A, k=-1) <= atol)),
        "no_skip": bool(np.all(A[upper_band] <= atol)),
        "rows_sum_to_1": bool(np.allclose(A.sum(axis=1), 1.0)),
        "starts_in_state_0": bool(np.isclose(pi[0], 1.0)),
    }


def save_models(models: Models, directory: Path) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for label, model in models.items():
        path = directory / f"hmm_{label}.pkl"
        with path.open("wb") as f:
            pickle.dump(model, f)
        paths.append(path)
    return paths


def load_models(directory: Path, labels: tuple[str, ...]) -> Models:
    """Nạp mô hình từ ``hmm_<label>.pkl``. Chỉ nạp file do chính pipeline này sinh ra (pickle không an toàn)."""
    models: Models = {}
    for label in labels:
        path = Path(directory) / f"hmm_{label}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Chưa có {path}. Hãy chạy bước 'train' trước.")
        with path.open("rb") as f:
            models[label] = pickle.load(f)
    return models


def load_meta(directory: Path) -> dict:
    """Đọc ``meta.json`` do bước train ghi (cấu hình đặc trưng/HMM, thời gian huấn luyện, ...)."""
    from .utils import load_json

    path = Path(directory) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}. Hãy chạy bước 'train' trước.")
    return load_json(path)
