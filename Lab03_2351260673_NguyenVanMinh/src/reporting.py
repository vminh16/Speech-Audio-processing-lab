"""Dựng hình và bảng Markdown từ các file kết quả trong ``results/`` (dùng chung cho script và notebook)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plots
from .config import DISPLAY, LABELS, RESULTS_DIR, TEST_DIR, FeatureConfig
from .features import extract_feat, load_wav, preprocessing_stages, trim_bounds
from .noise import add_white_noise
from .utils import load_json


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """DataFrame -> bảng Markdown (không cần thư viện tabulate)."""
    def fmt(v: object) -> str:
        text = floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
        return text.replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")  # `<s>` không bị hiểu là thẻ HTML

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _read_csv(results: Path, name: str) -> pd.DataFrame | None:
    path = results / name
    return pd.read_csv(path) if path.exists() else None


def pick_viterbi_files(vit: dict, k: int = 2) -> list[str]:
    """Chọn file test cuối cùng của ``k`` lớp đầu tiên để minh họa."""
    chosen = []
    for label in LABELS[:k]:
        files = sorted(f for f, v in vit.items() if v["label"] == label)
        if files:
            chosen.append(files[-1])
    return chosen


def build_figures(results: Path = RESULTS_DIR, fcfg: FeatureConfig | None = None) -> dict[str, plt.Figure]:
    """Dựng mọi hình có đủ dữ liệu đầu vào; hình thiếu dữ liệu bị bỏ qua."""
    fcfg = fcfg or FeatureConfig()
    figs: dict[str, plt.Figure] = {}

    vit_path = results / "viterbi_paths.json"
    if vit_path.exists():
        vit = load_json(vit_path)
        chosen = pick_viterbi_files(vit)
        if chosen:
            n_states = int(max(max(vit[f]["states"]) for f in chosen)) + 1
            n_states = max(n_states, load_json(results / "metrics_hmm.json").get("n_states", n_states))
            figs["viterbi_paths"] = plots.plot_viterbi_paths(
                {f"{f} (log P = {vit[f]['logprob']:.1f})": np.array(vit[f]["states"]) for f in chosen}, n_states
            )
            feats = {f: extract_feat(TEST_DIR / vit[f]["label"] / f, fcfg) for f in chosen}
            last = chosen[-1]
            figs["preprocessing"] = plots.plot_preprocessing(
                preprocessing_stages(load_wav(TEST_DIR / vit[last]["label"] / last, fcfg), fcfg), fcfg
            )
            if fcfg.use_delta:
                figs["mfcc_comparison"] = plots.plot_mfcc_comparison(feats, fcfg.n_mfcc)

    sample = TEST_DIR / LABELS[2] / f"{LABELS[2]}_07.wav"
    if sample.exists():  # minh họa trim mất tác dụng khi có nhiễu (nguyên nhân chính của E5/E6)
        y, rng = load_wav(sample, fcfg), np.random.default_rng(0)
        noisy = {"sạch": y} | {f"SNR {snr} dB": add_white_noise(y, snr, rng) for snr in (20, 10, 0)}
        figs["noise_trim"] = plots.plot_noise_trim({k: (v, trim_bounds(v, fcfg)) for k, v in noisy.items()}, fcfg.sr)

    for tag, fname in (("hmm", "confusion_matrix"), ("dtw", "confusion_matrix_dtw")):
        cm = _read_csv(results, f"confusion_matrix_{tag}.csv")
        if cm is not None:
            figs[fname] = plots.plot_confusion(cm.set_index(cm.columns[0]).to_numpy(), LABELS, f"Ma trận nhầm lẫn - {tag.upper()}")

    rej = _read_csv(results, "reject_scores.csv")
    if rej is not None:
        meta = load_json(results / "metrics_hmm.json")
        figs["score_distribution"] = plots.plot_score_distribution(
            rej[rej.kind == "loo_correct"].score.to_numpy(),
            rej[rej.kind == "loo_wrong"].score.to_numpy(),
            meta.get("reject_threshold"),
            rej[rej.kind == "test_correct"].score.to_numpy(),
        )

    for name, x in (("e1_states", "n_states"), ("e2_features", "features"), ("e3_covariance", "cov_type")):
        df = _read_csv(results, f"{name}.csv")
        if df is not None:
            long = df.melt(id_vars=[x], value_vars=["train_acc", "test_acc"], var_name="tập", value_name="accuracy")
            figs[name] = plots.plot_grouped_bars(long, x, "accuracy", hue="tập", title=name.upper()[:2] + ": " + name[3:])

    e4 = _read_csv(results, "e4_data_size.csv")
    if e4 is not None:
        figs["e4_data_size"] = plots.plot_grouped_bars(e4, "n_train", "mean_acc", hue="method", yerr="std_acc", title="E4: số mẫu train / lớp")
    e5 = _read_csv(results, "e5_robustness.csv")
    if e5 is not None:
        figs["e5_robustness"] = plots.plot_grouped_bars(e5, "condition", "mean_acc", hue="method", yerr="std_acc", title="E5: độ bền vững với nhiễu")
    e6 = _read_csv(results, "e6_lm.csv")
    if e6 is not None:
        figs["e6_lm"] = plots.plot_lambda_curves(e6)
    return figs


def build_summary(results: Path = RESULTS_DIR) -> str:
    """Gộp các bảng số liệu thành một file Markdown để dán vào README / báo cáo."""
    parts = ["# Tổng hợp kết quả Lab 3 (sinh tự động bởi `scripts/report.py`)\n"]

    hmm_p, dtw_p = results / "metrics_hmm.json", results / "metrics_dtw.json"
    if hmm_p.exists() and dtw_p.exists():
        h, d = load_json(hmm_p), load_json(dtw_p)
        fmt_pairs = lambda m: ", ".join(f"{a}→{b} ({n})" for a, b, n in m["most_confused"]) or "không có"
        rows = [
            ("Accuracy (%)", f"{100 * d['accuracy']:.1f}", f"{100 * h['accuracy']:.1f}"),
            ("Số mẫu train / lớp", d["train_per_class"], h["train_per_class"]),
            ("Thời gian train (ms)", f"{d['train_time_ms']:.2f}", f"{h['train_time_ms']:.1f}"),
            ("Thời gian test / trial (ms)", f"{d['infer_ms_per_trial']:.2f}", f"{h['infer_ms_per_trial']:.2f}"),
            ("Cặp từ dễ nhầm nhất", fmt_pairs(d), fmt_pairs(h)),
            ("Cơ chế alignment", "Warping path", "Viterbi state path"),
        ]
        parts += ["## So sánh DTW và Gaussian HMM\n", md_table(pd.DataFrame(rows, columns=["Chỉ số", "DTW (Lab 2)", "Gaussian HMM (Lab 3)"])), ""]

    scores = _read_csv(results, "scores_test.csv")
    if scores is not None:
        cols = ["file", "label", *[f"ll_{l}" for l in LABELS], "pred", "margin"]
        first = scores.groupby("label", sort=False).tail(1)[cols]
        parts += ["## Bảng log-likelihood (mỗi lớp một file test)\n", md_table(first, "{:.1f}"), ""]

    titles = {
        "e1_states": "E1 - Số trạng thái",
        "e2_features": "E2 - Chiều đặc trưng",
        "e3_covariance": "E3 - Kiểu covariance",
        "e4_data_size": "E4 - Số mẫu train / lớp",
        "e5_robustness": "E5 - Độ bền vững với nhiễu",
        "e6_lm": "E6 - Trọng số LM",
    }
    for name, title in titles.items():
        df = _read_csv(results, f"{name}.csv")
        if df is not None:
            parts += [f"## {title}\n", md_table(df), ""]

    lm = _read_csv(results, "bigram_table.csv")
    if lm is not None:
        parts += ["## Bảng P(cur | prev) của bigram LM\n", md_table(lm, "{:.3f}"), ""]
    parts.append("Tên hiển thị: " + ", ".join(f"`{k}` = {v}" for k, v in DISPLAY.items()) + "\n")
    return "\n".join(parts)
