import pandas as pd
import pytest
from src.evaluation import summarize_ratings, plot_methods_comparison


def test_summarize_real_responses(tmp_path):
    path = tmp_path / "ratings.csv"
    pd.DataFrame([
        dict(listener_id="A", sentence_id=1, method="concat", naturalness_score=4, intelligibility_correct=8, intelligibility_total=10, continuity_score=3),
        dict(listener_id="B", sentence_id=1, method="concat", naturalness_score=2, intelligibility_correct=9, intelligibility_total=10, continuity_score=5),
    ]).to_csv(path, index=False)
    row = summarize_ratings(path).loc["concat"]
    assert row.mos == 3
    assert row.intelligibility_pct == 85
    assert row.continuity == 4


def test_empty_ratings_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("listener_id,sentence_id,method,naturalness_score,intelligibility_correct,intelligibility_total,continuity_score\n")
    with pytest.raises(ValueError):
        summarize_ratings(path)


def test_plot_measured_latency(tmp_path):
    path = tmp_path / "plot.png"
    plot_methods_comparison([{"method": "concat", "latency_ms": 10}], str(path))
    assert path.exists()
