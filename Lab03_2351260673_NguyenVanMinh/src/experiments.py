"""Các thí nghiệm khảo sát E1-E6. Mỗi hàm trả về ``pandas.DataFrame`` để script lưu CSV và notebook vẽ."""
from __future__ import annotations

from dataclasses import replace
from math import comb

import numpy as np
import pandas as pd

from .acoustic import Models, train_models
from .config import FeatureConfig, HMMConfig
from .data import Utt, group_by_label
from .decode import score_all
from .dtw import DTWRecognizer
from .features import FeatSet, build_featset, by_label
from .lm import BigramLM, rerank
from .metrics import accuracy
from .utils import Timer

SNR_CONDITIONS: tuple[float | None, ...] = (None, 20.0, 10.0, 5.0, 0.0)


def condition_name(snr_db: float | None) -> str:
    return "clean" if snr_db is None else f"SNR {snr_db:g} dB"


def hmm_accuracy(models: Models, featset: FeatSet) -> float:
    preds = [max(s := score_all(f, models), key=s.get) for _, f in featset]
    return accuracy([l for l, _ in featset], preds)


def dtw_accuracy(rec: DTWRecognizer, featset: FeatSet) -> float:
    return accuracy([l for l, _ in featset], [rec.predict(f)[0] for _, f in featset])


def mean_ll_per_frame(models: Models, featset: FeatSet) -> float:
    """Trung bình log-likelihood theo frame của mô hình *đúng lớp*."""
    return float(np.mean([models[l].score(f) / len(f) for l, f in featset]))


def _fit_and_eval(trf: FeatSet, tef: FeatSet, cfg: HMMConfig) -> dict:
    models, infos = train_models(by_label(trf), cfg)
    return {
        "train_acc": hmm_accuracy(models, trf),
        "test_acc": hmm_accuracy(models, tef),
        "converged": f"{sum(i.converged for i in infos.values())}/{len(infos)}",
        "mean_iters": float(np.mean([i.n_iter_run for i in infos.values()])),
        "train_ll": mean_ll_per_frame(models, trf),
        "test_ll": mean_ll_per_frame(models, tef),
        "train_time_s": float(sum(i.train_time_s for i in infos.values())),
    }


def e1_states(train: list[Utt], test: list[Utt], fcfg: FeatureConfig, hcfg: HMMConfig, states=(3, 5, 7)) -> pd.DataFrame:
    """E1: số trạng thái N."""
    trf, tef = build_featset(train, fcfg), build_featset(test, fcfg)
    return pd.DataFrame([{"n_states": n, **_fit_and_eval(trf, tef, replace(hcfg, n_states=n))} for n in states])


def e2_features(train: list[Utt], test: list[Utt], fcfg: FeatureConfig, hcfg: HMMConfig) -> pd.DataFrame:
    """E2: 13-D (MFCC tĩnh) so với 39-D (MFCC + Δ + Δ²)."""
    rows = []
    for use_delta in (False, True):
        c = replace(fcfg, use_delta=use_delta)
        res = _fit_and_eval(build_featset(train, c), build_featset(test, c), hcfg)
        rows.append({"features": f"{c.dim}-D", **res})
    return pd.DataFrame(rows)


def e3_covariance(train: list[Utt], test: list[Utt], fcfg: FeatureConfig, hcfg: HMMConfig) -> pd.DataFrame:
    """E3: ma trận hiệp phương sai ``diag`` so với ``full``; ``ll_gap`` = train_ll - test_ll đo mức overfit."""
    trf, tef = build_featset(train, fcfg), build_featset(test, fcfg)
    rows = []
    for cov in ("diag", "full"):
        res = _fit_and_eval(trf, tef, replace(hcfg, cov_type=cov))
        rows.append({"cov_type": cov, **res, "ll_gap": res["train_ll"] - res["test_ll"]})
    return pd.DataFrame(rows)


def e4_data_size(
    train: list[Utt],
    test: list[Utt],
    fcfg: FeatureConfig,
    hcfg: HMMConfig,
    sizes=(1, 2, 3, 4, 6),
    repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """E4: độ chính xác của HMM và DTW khi giảm số mẫu train mỗi lớp (lấy ngẫu nhiên có seed)."""
    tef = build_featset(test, fcfg)
    groups = group_by_label(train)
    max_n = min(len(g) for g in groups.values())
    rng = np.random.default_rng(seed)
    rows = []
    for n in sorted({min(s, max_n) for s in sizes}):
        n_subsets = comb(max_n, n) ** len(groups)
        runs = repeats if n_subsets > 1 else 1
        accs: dict[str, list[float]] = {"HMM": [], "DTW": []}
        for _ in range(runs):
            subset = [u for g in groups.values() for u in (g[i] for i in rng.choice(len(g), size=n, replace=False))]
            trf = build_featset(subset, fcfg)
            models, _ = train_models(by_label(trf), hcfg)
            accs["HMM"].append(hmm_accuracy(models, tef))
            accs["DTW"].append(dtw_accuracy(DTWRecognizer().fit(by_label(trf)), tef))
        rows += [{"n_train": n, "method": m, "mean_acc": np.mean(v), "std_acc": np.std(v), "runs": runs} for m, v in accs.items()]
    return pd.DataFrame(rows)


def e5_robustness(
    models: Models,
    dtw_rec: DTWRecognizer,
    test: list[Utt],
    mismatch: list[Utt],
    fcfg: FeatureConfig,
    snrs=SNR_CONDITIONS,
    repeats: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """E5: mô hình huấn luyện trên dữ liệu sạch, kiểm thử với nhiễu trắng theo SNR (và tập test_mismatch nếu có)."""
    conditions: list[tuple[str, list[FeatSet]]] = []
    for snr in snrs:
        n_rep = 1 if snr is None else repeats
        conditions.append((condition_name(snr), [build_featset(test, fcfg, snr, seed + r) for r in range(n_rep)]))
    if mismatch:
        conditions.append(("test_mismatch", [build_featset(mismatch, fcfg)]))

    rows = []
    for name, featsets in conditions:
        for method, fn in (("HMM", lambda fs: hmm_accuracy(models, fs)), ("DTW", lambda fs: dtw_accuracy(dtw_rec, fs))):
            accs = [fn(fs) for fs in featsets]
            rows.append({"condition": name, "method": method, "mean_acc": np.mean(accs), "std_acc": np.std(accs), "runs": len(accs)})
    return pd.DataFrame(rows)


def e6_lm(
    models: Models,
    lm: BigramLM,
    test: list[Utt],
    fcfg: FeatureConfig,
    lambdas=(0.0, 0.1, 0.3, 1.0),
    snrs=(None, 10.0, 5.0, 0.0),
    repeats: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """E6: ảnh hưởng của trọng số LM. Mỗi utterance được thử với mọi từ đứng trước nó trong corpus
    (kể cả ``<s>``), mô phỏng việc nhận dạng từ cuối của một câu lệnh có ngữ cảnh."""
    rows = []
    for snr in snrs:
        n_rep = 1 if snr is None else repeats
        correct = {lam: 0 for lam in lambdas}
        total = 0
        for r in range(n_rep):
            for label, feat in build_featset(test, fcfg, snr, seed + r):
                per_frame = {w: s / len(feat) for w, s in score_all(feat, models).items()}
                for prev in lm.predecessors(label):
                    total += 1
                    for lam in lambdas:
                        correct[lam] += rerank(per_frame, prev, lm, lam)[0] == label
        rows += [{"condition": condition_name(snr), "lambda_lm": lam, "accuracy": correct[lam] / total, "trials": total} for lam in lambdas]
    return pd.DataFrame(rows)
