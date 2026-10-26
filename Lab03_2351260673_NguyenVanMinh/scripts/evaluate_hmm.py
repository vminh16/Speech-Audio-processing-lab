"""Bước 3 - Đánh giá HMM trên tập test: bảng log-likelihood, ma trận nhầm lẫn, đường Viterbi, ngưỡng từ chối."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.acoustic import load_meta, load_models
from src.config import LABELS, MODELS_DIR, RESULTS_DIR, TEST_DIR, TRAIN_DIR, FeatureConfig, HMMConfig
from src.data import list_split
from src.decode import recognize, state_durations, viterbi_path
from src.features import build_featset, by_label
from src.metrics import accuracy, confusion, most_confused, per_class_accuracy
from src.rejection import loo_scores, youden_threshold
from src.utils import Timer, get_logger, save_json

log = get_logger()


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--no-reject", action="store_true", help="bỏ qua bước chọn ngưỡng từ chối (leave-one-out)")
    args = parser.parse_args(argv)

    meta = load_meta(args.models_dir)
    fcfg, hcfg = FeatureConfig(**meta["feature_config"]), HMMConfig(**meta["hmm_config"])
    models = load_models(args.models_dir, LABELS)

    test = list_split(TEST_DIR)
    if not test:
        raise SystemExit(f"Không có dữ liệu test trong {TEST_DIR}")
    featset = build_featset(test, fcfg)

    recognize(featset[0][1], models)  # khởi động để thời gian đo không dính chi phí lần gọi đầu
    with Timer() as t_inf:
        results = [recognize(f, models) for _, f in featset]
    infer_ms = 1000 * t_inf.elapsed / len(featset)

    y_true, y_pred = [l for l, _ in featset], [r.pred for r in results]
    rows, vit = [], {}
    for utt, (label, feat), r in zip(test, featset, results):
        rows.append(
            {
                "file": utt.name, "label": label, "n_frames": r.n_frames,
                **{f"ll_{w}": r.scores[w] for w in LABELS},
                "pred": r.pred, "correct": r.pred == label, "margin": r.margin,
                "best_ll_per_frame": r.per_frame[r.pred],
            }
        )
        logprob, states = viterbi_path(feat, models[label])  # gióng hàng với mô hình của từ đúng
        vit[utt.name] = {
            "label": label, "logprob": logprob, "states": states.tolist(),
            "durations": state_durations(states, hcfg.n_states).tolist(),
        }
    table = pd.DataFrame(rows)

    cm = confusion(y_true, y_pred, LABELS)
    acc = accuracy(y_true, y_pred)
    log.info("Accuracy (Forward/ML) = %.1f%% trên %d utterance test", 100 * acc, len(test))
    log.info("Margin Top1-Top2: min=%.1f, trung bình=%.1f", table.margin.min(), table.margin.mean())
    for _, row in table[~table.correct].iterrows():
        log.warning("Sai: %s (thực=%s, đoán=%s, margin=%.1f)", row.file, row.label, row.pred, row.margin)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "scores_test.csv", index=False)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).rename_axis("true").to_csv(RESULTS_DIR / "confusion_matrix_hmm.csv")
    save_json(vit, RESULTS_DIR / "viterbi_paths.json")

    metrics = {
        "accuracy": acc, "n_test": len(test), "n_states": hcfg.n_states,
        "per_class_accuracy": per_class_accuracy(cm, LABELS),
        "most_confused": most_confused(cm, LABELS),
        "mean_margin": float(table.margin.mean()), "min_margin": float(table.margin.min()),
        "train_per_class": int(np.min(list(meta["train_per_class"].values()))),
        "train_time_ms": 1000 * meta["train_time_s"],
        "infer_ms_per_trial": infer_ms,
    }

    if not args.no_reject:
        pos, neg = loo_scores(by_label(build_featset(list_split(TRAIN_DIR), fcfg)), hcfg)
        threshold = youden_threshold(pos, neg)
        test_pos = np.array([r.per_frame[l] for r, l in zip(results, y_true)])
        metrics |= {
            "reject_threshold": threshold,
            "loo_tpr": float(np.mean(pos >= threshold)), "loo_fpr": float(np.mean(neg >= threshold)),
            "test_accept_rate": float(np.mean(test_pos >= threshold)),
        }
        pd.DataFrame(
            {"kind": ["loo_correct"] * len(pos) + ["loo_wrong"] * len(neg) + ["test_correct"] * len(test_pos),
             "score": np.concatenate([pos, neg, test_pos])}
        ).to_csv(RESULTS_DIR / "reject_scores.csv", index=False)
        log.info("Ngưỡng từ chối (LOO, Youden) = %.2f | TPR=%.2f FPR=%.2f | test được chấp nhận: %.0f%%",
                 threshold, metrics["loo_tpr"], metrics["loo_fpr"], 100 * metrics["test_accept_rate"])

    save_json(metrics, RESULTS_DIR / "metrics_hmm.json")
    return {"table": table, "confusion": cm, "metrics": metrics, "viterbi": vit}


if __name__ == "__main__":
    main()
