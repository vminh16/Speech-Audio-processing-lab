"""Các hàm vẽ biểu đồ. Mỗi hàm trả về ``Figure``; việc lưu/hiển thị do nơi gọi quyết định."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DISPLAY


def _name(label: str) -> str:
    return f"{DISPLAY.get(label, label)} ({label})" if label in DISPLAY else label


def save(fig: plt.Figure, path: Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_mfcc_comparison(feats: dict[str, np.ndarray], n_mfcc: int = 13) -> plt.Figure:
    """Heatmap MFCC / Δ / Δ² (đặc trưng 39-D sau CMVN) của một vài utterance, mỗi utterance một cột."""
    names = list(feats)
    titles = ["MFCC (c0-c12)", "Δ MFCC", "Δ² MFCC"]
    fig, axes = plt.subplots(3, len(names), figsize=(5.5 * len(names), 8), squeeze=False, sharey=True, layout="constrained")
    for col, name in enumerate(names):
        F = feats[name]
        for row in range(3):
            block = F[:, row * n_mfcc : (row + 1) * n_mfcc]
            im = axes[row, col].imshow(block.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
            axes[row, col].set_title(f"{name} - {titles[row]}", fontsize=10)
            axes[row, col].set_ylabel("Hệ số")
            if row == 2:
                axes[row, col].set_xlabel("Frame (10 ms)")
        fig.colorbar(im, ax=axes[:, col].tolist(), shrink=0.6, label="Giá trị sau CMVN")
    return fig


def plot_preprocessing(stages: dict, cfg) -> plt.Figure:
    """Các bước tiền xử lý của một utterance: trim, pre-emphasis, log-mel, MFCC và tác dụng của CMVN."""
    sr, hop = cfg.sr, cfg.hop_length
    start, end = stages["bounds"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), layout="constrained")

    ax = axes[0, 0]
    ax.plot(np.arange(len(stages["raw"])) / sr, stages["raw"], lw=0.5)
    ax.axvspan(start / sr, end / sr, color="tab:green", alpha=0.2, label=f"giữ lại (top_db={cfg.top_db:g})")
    ax.set(title="1. Waveform và vùng sau trim", xlabel="Thời gian (s)", ylabel="Biên độ")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for name, key in (("trước pre-emphasis", "trimmed"), (f"sau pre-emphasis (α={cfg.preemph})", "emphasized")):
        spec = np.abs(np.fft.rfft(stages[key] * np.hanning(len(stages[key]))))
        ax.plot(np.fft.rfftfreq(len(stages[key]), 1 / sr), 20 * np.log10(spec + 1e-9), lw=0.7, label=name)
    ax.set(title="2. Phổ trung bình: pre-emphasis nâng tần số cao", xlabel="Tần số (Hz)", ylabel="Biên độ (dB)")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    im = ax.imshow(stages["log_mel"], origin="lower", aspect="auto", cmap="magma", extent=[0, stages["log_mel"].shape[1] * hop / sr, 0, cfg.n_mels])
    ax.set(title=f"3. Log-Mel spectrogram ({cfg.n_mels} băng)", xlabel="Thời gian (s)", ylabel="Băng mel")
    fig.colorbar(im, ax=ax, label="dB")

    t_end = len(stages["feat"]) * hop / sr
    ax = axes[1, 0]
    im = ax.imshow(stages["feat_raw"][:, : cfg.n_mfcc].T, origin="lower", aspect="auto", cmap="coolwarm", extent=[0, t_end, 0, cfg.n_mfcc])
    ax.set(title="4. 13 MFCC (trước CMVN)", xlabel="Thời gian (s)", ylabel="Hệ số")
    fig.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(stages["feat"].T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-3, vmax=3, extent=[0, t_end, 0, stages["feat"].shape[1]])
    ax.set(title=f"5. Đặc trưng {stages['feat'].shape[1]}-D sau CMVN", xlabel="Thời gian (s)", ylabel="Chiều")
    fig.colorbar(im, ax=ax)

    ax = axes[1, 2]
    dims = np.arange(stages["feat"].shape[1])
    ax.semilogy(dims, stages["feat_raw"].std(axis=0), "o-", ms=3, label="std trước CMVN")
    ax.semilogy(dims, stages["feat"].std(axis=0), "s-", ms=3, label="std sau CMVN")
    ax.set(title="6. CMVN: mean -> 0 và std -> 1 ở mọi chiều", xlabel="Chiều đặc trưng", ylabel="Độ lệch chuẩn (thang log)")
    ax.text(0.03, 0.05, f"max |mean| sau CMVN = {np.abs(stages['feat'].mean(axis=0)).max():.1e}", transform=ax.transAxes, fontsize=8)
    ax.legend(fontsize=8)
    return fig


def plot_noise_trim(signals: dict[str, tuple[np.ndarray, tuple[int, int]]], sr: int) -> plt.Figure:
    """Waveform ở các mức SNR kèm vùng được giữ lại sau trim, cho thấy trim mất tác dụng khi có nhiễu."""
    fig, axes = plt.subplots(len(signals), 1, figsize=(10, 1.9 * len(signals)), sharex=True, sharey=True, layout="constrained")
    for ax, (name, (y, (start, end))) in zip(np.atleast_1d(axes), signals.items()):
        ax.plot(np.arange(len(y)) / sr, y, lw=0.5)
        ax.axvspan(start / sr, end / sr, color="tab:green", alpha=0.2)
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
        ax.text(0.995, 0.85, f"giữ lại {(end - start) / sr:.2f}/{len(y) / sr:.2f} s", transform=ax.transAxes, ha="right", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("Thời gian (s) - vùng xanh là phần giữ lại sau trim")
    return fig


def plot_transmats(transmats: dict[str, np.ndarray]) -> plt.Figure:
    """Ma trận chuyển trạng thái A của từng mô hình (kiểm tra cấu trúc trái-sang-phải)."""
    fig, axes = plt.subplots(1, len(transmats), figsize=(3.2 * len(transmats), 3.4), squeeze=False)
    for ax, (label, A) in zip(axes[0], transmats.items()):
        im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] > 0.005:
                    ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if A[i, j] > 0.5 else "black")
        ax.set_title(_name(label), fontsize=10)
        ax.set_xlabel("Trạng thái đích")
        ax.set_ylabel("Trạng thái nguồn")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="a_ij")
    return fig


def plot_viterbi_paths(paths: dict[str, np.ndarray], n_states: int) -> plt.Figure:
    """Đường đi trạng thái Viterbi theo frame (đồ thị bậc thang)."""
    fig, axes = plt.subplots(1, len(paths), figsize=(5.5 * len(paths), 3.6), squeeze=False, sharey=True)
    for ax, (title, states) in zip(axes[0], paths.items()):
        ax.step(np.arange(len(states)), states, where="post", lw=2)
        ax.set_yticks(range(n_states))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Frame (10 ms)")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Trạng thái ẩn q*")
    return fig


def plot_confusion(cm: np.ndarray, labels: tuple[str, ...], title: str = "Ma trận nhầm lẫn") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ticks = [_name(l) for l in labels]
    ax.set_xticks(range(len(labels)), ticks, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), ticks)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thực")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


def plot_score_distribution(pos: np.ndarray, neg: np.ndarray, threshold: float | None = None, test_pos: np.ndarray | None = None) -> plt.Figure:
    """Phân bố log-likelihood theo frame: mô hình đúng lớp vs sai lớp, kèm ngưỡng từ chối."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 40)
    ax.hist(neg, bins=bins, alpha=0.6, label="Mô hình sai lớp (LOO train)", color="tab:red")
    ax.hist(pos, bins=bins, alpha=0.6, label="Mô hình đúng lớp (LOO train)", color="tab:green")
    if test_pos is not None:
        ax.plot(test_pos, np.full_like(test_pos, -0.5), "k|", ms=14, label="Test (mô hình đúng lớp)")
    if threshold is not None:
        ax.axvline(threshold, color="k", ls="--", label=f"Ngưỡng từ chối = {threshold:.1f}")
    ax.set_xlabel("log P(X|λ) / T")
    ax.set_ylabel("Số utterance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def plot_grouped_bars(df: pd.DataFrame, x: str, y: str, hue: str | None = None, yerr: str | None = None, title: str = "", ylabel: str = "Accuracy") -> plt.Figure:
    """Cột nhóm theo ``x`` (và ``hue``); ``yerr`` là tên cột độ lệch chuẩn nếu có."""
    xs = list(dict.fromkeys(df[x]))
    hues = list(dict.fromkeys(df[hue])) if hue else [None]
    width = 0.8 / len(hues)
    fig, ax = plt.subplots(figsize=(max(5.5, 1.3 * len(xs) * len(hues)), 4))
    for k, h in enumerate(hues):
        sub = df if h is None else df[df[hue] == h]
        sub = sub.set_index(x).loc[xs]
        ax.bar(np.arange(len(xs)) + (k - (len(hues) - 1) / 2) * width, sub[y], width, yerr=sub[yerr] if yerr else None, capsize=3, label=h)
    ax.set_xticks(range(len(xs)), [str(v) for v in xs])
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.08)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    if hue:
        ax.legend(title=hue, fontsize=8)
    return fig


def plot_lambda_curves(df: pd.DataFrame) -> plt.Figure:
    """Accuracy theo λ_LM, mỗi đường là một điều kiện nhiễu."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond, sub in df.groupby("condition", sort=False):
        ax.plot(sub["lambda_lm"], sub["accuracy"], marker="o", label=cond)
    ax.set_xlabel("λ_LM")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig
