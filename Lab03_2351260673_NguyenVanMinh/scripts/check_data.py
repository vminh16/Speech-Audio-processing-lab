"""Bước 1 - Kiểm tra dataset: định dạng WAV, số file mỗi lớp, rò rỉ train/test."""
from __future__ import annotations

import argparse
import sys

from src.config import LABELS, RESULTS_DIR
from src.data import load_splits, validate_dataset
from src.utils import get_logger

log = get_logger()
RECOMMENDED_TRAIN, RECOMMENDED_TEST = 6, 2


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr", type=int, default=16000, help="tần số lấy mẫu yêu cầu (Hz)")
    args = parser.parse_args(argv)

    train, test, mismatch = load_splits()
    df, errors = validate_dataset({"train": train, "test": test, "test_mismatch": mismatch}, sr=args.sr)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "data_report.csv", index=False)

    counts = df.pivot_table(index="label", columns="split", values="file", aggfunc="count", fill_value=0).reindex(LABELS)
    log.info("Số file theo lớp:\n%s", counts.to_string())
    log.info("Thời lượng trung bình: %.2f s | peak lớn nhất: %.3f", df.duration_s.mean(), df.peak.max())

    warnings = []
    for label in LABELS:
        n_tr, n_te = int(counts.loc[label].get("train", 0)), int(counts.loc[label].get("test", 0))
        if n_tr < RECOMMENDED_TRAIN or n_te < RECOMMENDED_TEST:
            warnings.append(f"Lớp '{label}': {n_tr} train / {n_te} test (khuyến nghị {RECOMMENDED_TRAIN}/{RECOMMENDED_TEST}).")
    for w in warnings:
        log.warning(w)
    for e in errors:
        log.error(e)
    log.info("Kiểm tra dữ liệu: %s", "OK" if not errors else f"{len(errors)} lỗi")
    return {"errors": errors, "warnings": warnings, "report": df}


if __name__ == "__main__":
    sys.exit(1 if main()["errors"] else 0)
