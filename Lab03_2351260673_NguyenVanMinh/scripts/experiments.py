"""Bước 6 - Thí nghiệm khảo sát E1-E5 (E6 nằm ở ``scripts.lm_rerank``)."""
from __future__ import annotations

import argparse

from src import experiments as ex
from src.acoustic import load_meta, load_models
from src.config import LABELS, MISMATCH_DIR, MODELS_DIR, RESULTS_DIR, TEST_DIR, TRAIN_DIR, FeatureConfig, HMMConfig
from src.data import list_split
from src.dtw import DTWRecognizer
from src.features import build_featset, by_label
from src.utils import Timer, get_logger

log = get_logger()
ALL = ("E1", "E2", "E3", "E4", "E5")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=ALL, default=list(ALL), help="chỉ chạy một số thí nghiệm")
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--repeats", type=int, default=5, help="số lần lấy mẫu con ngẫu nhiên cho E4")
    parser.add_argument("--noise-repeats", type=int, default=3, help="số lần lặp nhiễu cho mỗi mức SNR trong E5")
    args = parser.parse_args(argv)

    meta = load_meta(args.models_dir)
    fcfg, hcfg = FeatureConfig(**meta["feature_config"]), HMMConfig(**meta["hmm_config"])
    train, test, mismatch = list_split(TRAIN_DIR), list_split(TEST_DIR), list_split(MISMATCH_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    jobs = {
        "E1": ("e1_states", lambda: ex.e1_states(train, test, fcfg, hcfg)),
        "E2": ("e2_features", lambda: ex.e2_features(train, test, fcfg, hcfg)),
        "E3": ("e3_covariance", lambda: ex.e3_covariance(train, test, fcfg, hcfg)),
        "E4": ("e4_data_size", lambda: ex.e4_data_size(train, test, fcfg, hcfg, repeats=args.repeats)),
        "E5": ("e5_robustness", lambda: ex.e5_robustness(
            load_models(args.models_dir, LABELS),
            DTWRecognizer().fit(by_label(build_featset(train, fcfg))),
            test, mismatch, fcfg, repeats=args.noise_repeats)),
    }
    out = {}
    for key in args.only:
        name, job = jobs[key]
        with Timer() as t:
            df = job()
        df.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
        out[key] = df
        log.info("%s xong trong %.1f s -> results/%s.csv\n%s", key, t.elapsed, name, df.round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    main()
