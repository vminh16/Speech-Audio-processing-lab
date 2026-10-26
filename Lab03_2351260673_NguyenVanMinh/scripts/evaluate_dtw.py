"""Bước 4 - Baseline DTW (Lab 2) trên đúng cùng đặc trưng, cùng tập train/test với HMM."""
from __future__ import annotations

import argparse

import pandas as pd

from src.acoustic import load_meta
from src.config import LABELS, MODELS_DIR, RESULTS_DIR, TEST_DIR, TRAIN_DIR, FeatureConfig
from src.data import list_split
from src.dtw import DTWRecognizer
from src.features import build_featset, by_label
from src.metrics import accuracy, confusion, most_confused, per_class_accuracy
from src.utils import Timer, get_logger, save_json

log = get_logger()


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help="nơi chứa meta.json để dùng chung cấu hình đặc trưng với HMM")
    args = parser.parse_args(argv)

    fcfg = FeatureConfig(**load_meta(args.models_dir)["feature_config"])
    train_fs, test_fs = build_featset(list_split(TRAIN_DIR), fcfg), build_featset(list_split(TEST_DIR), fcfg)
    test = list_split(TEST_DIR)

    with Timer() as t_fit:  # DTW không có pha học: chỉ lưu template
        rec = DTWRecognizer().fit(by_label(train_fs))
    with Timer() as t_inf:
        outs = [rec.predict(f) for _, f in test_fs]

    y_true, y_pred = [l for l, _ in test_fs], [p for p, _ in outs]
    rows = []
    for utt, (pred, dist) in zip(test, outs):
        ranked = sorted(dist.values())
        rows.append({"file": utt.name, "label": utt.label, **{f"dist_{w}": dist[w] for w in LABELS},
                     "pred": pred, "correct": pred == utt.label, "margin": ranked[1] - ranked[0]})
    table = pd.DataFrame(rows)

    cm, acc = confusion(y_true, y_pred, LABELS), accuracy(y_true, y_pred)
    log.info("Accuracy (DTW 1-NN) = %.1f%% trên %d utterance test", 100 * acc, len(test_fs))
    for _, row in table[~table.correct].iterrows():
        log.warning("Sai: %s (thực=%s, đoán=%s)", row.file, row.label, row.pred)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "scores_dtw.csv", index=False)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).rename_axis("true").to_csv(RESULTS_DIR / "confusion_matrix_dtw.csv")
    metrics = {
        "accuracy": acc, "n_test": len(test_fs),
        "per_class_accuracy": per_class_accuracy(cm, LABELS),
        "most_confused": most_confused(cm, LABELS),
        "train_per_class": min(len(v) for v in by_label(train_fs).values()),
        "train_time_ms": 1000 * t_fit.elapsed,
        "infer_ms_per_trial": 1000 * t_inf.elapsed / len(test_fs),
    }
    save_json(metrics, RESULTS_DIR / "metrics_dtw.json")
    return {"table": table, "confusion": cm, "metrics": metrics}


if __name__ == "__main__":
    main()
