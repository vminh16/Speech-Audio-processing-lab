"""Prepare reproducible listening material and summarize actual ratings."""
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from src.benchmark import DEFAULT_BENCHMARK_SENTENCES
from src.hmm_tts import synthesize_hmm
from src.neural_tts import synthesize as synthesize_mel_demo
from src.smoothing import crossfade_sequence
from src.text_norm import normalize_queue_text
from src.unit_selection import compare_candidate_strategies
from src.utils import match_rms


def prepare_evaluation(output_dir="evaluation", units_dir="data/units"):
    """Create nine RMS-matched samples and a blank rating sheet; no ratings are invented."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, sentence in enumerate(DEFAULT_BENCHMARK_SENTENCES, 1):
        for method in ("concat", "hmm", "neural"):
            start = time.perf_counter()
            if method == "concat":
                candidates = compare_candidate_strategies(normalize_queue_text(sentence), units_dir=units_dir)
                y, _ = crossfade_sequence([u["audio"] for u in candidates["optimal_viterbi"]["units"]], sr=16000, xfade_ms=15)
                sr = 16000
            elif method == "hmm":
                y, sr, _ = synthesize_hmm(sentence)
            else:
                y, sr, _ = synthesize_mel_demo(sentence)
            elapsed = (time.perf_counter() - start) * 1000
            y = np.clip(match_rms(y, target_rms=0.08), -0.99, 0.99)
            sf.write(out / f"{method}_{i}.wav", y, sr, subtype="PCM_16")
            results.append({"sentence_id": i, "method": method, "latency_ms": round(elapsed, 2), "duration_s": round(len(y)/sr, 3), "rtf": round(elapsed / 1000 / (len(y)/sr), 4)})
    pd.DataFrame(results).to_csv(out / "measurements.csv", index=False)
    models = Path(__file__).resolve().parents[1] / 'data/models'
    storage = {'concat':sum(p.stat().st_size for p in Path(units_dir).glob('*.wav')),
               'hmm':(models/'hmm.json').stat().st_size,
               'neural':sum(p.stat().st_size for p in (models/'piper').glob('*'))}
    summary = pd.DataFrame(results).groupby('method')[['latency_ms','rtf','duration_s']].mean()
    summary['storage_mb'] = [storage[m]/1024**2 for m in summary.index]
    summary['mos'] = np.nan
    summary['intelligibility_pct'] = np.nan
    summary['continuity'] = np.nan
    summary.to_csv(out/'summary.csv')
    with (out / "ratings_template.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["listener_id", "sentence_id", "method", "naturalness_score", "intelligibility_correct", "intelligibility_total", "continuity_score"])
    plot_methods_comparison(results, str(out / "latency_comparison.png"))
    import random, shutil
    rng = random.Random(42)
    mapping = []
    for sentence_id in range(1,4):
        methods = ['concat','hmm','neural']
        rng.shuffle(methods)
        for j, method in enumerate(methods):
            sample = f"S{sentence_id}_{chr(65+j)}"
            shutil.copyfile(out / f"{method}_{sentence_id}.wav", out / f"{sample}.wav")
            mapping.append(dict(sample_id=sample,sentence_id=sentence_id,method=method))
    pd.DataFrame(mapping).to_csv(out / 'organizer_key.csv',index=False)
    return results


def summarize_ratings(path):
    """Aggregate real listener responses; intelligibility is correct words / total words."""
    df = pd.read_csv(path)
    required = {"listener_id", "sentence_id", "method", "naturalness_score", "intelligibility_correct", "intelligibility_total", "continuity_score"}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError("Ratings must contain actual listener responses and all required columns")
    if df[list(required)].isna().any().any() or not df["naturalness_score"].between(1, 5).all() or not df["continuity_score"].between(1, 5).all():
        raise ValueError("Missing or invalid rating")
    if ((df.intelligibility_total <= 0) | (df.intelligibility_correct < 0) | (df.intelligibility_correct > df.intelligibility_total)).any():
        raise ValueError("Invalid word counts")
    grouped = df.groupby("method")[["naturalness_score", "continuity_score", "intelligibility_correct", "intelligibility_total"]]
    return grouped.apply(lambda g: pd.Series({"mos": g.naturalness_score.mean(), "continuity": g.continuity_score.mean(), "intelligibility_pct": 100 * g.intelligibility_correct.sum() / g.intelligibility_total.sum(), "responses": len(g)}))


def plot_methods_comparison(results, output_fig="evaluation/latency_comparison.png"):
    df = pd.DataFrame(results)
    means = df.groupby("method")["latency_ms"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    means.plot.bar(ax=ax, color=["#2a9d8f", "#e9c46a", "#e76f51"])
    ax.set_ylabel("Mean latency (ms), 3 sentences")
    ax.set_xlabel("")
    ax.set_title("Three synthesis backends")
    fig.tight_layout()
    fig.savefig(output_fig, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    prepare_evaluation()
