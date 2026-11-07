"""
Unit test for TTS benchmark module.
Following Test-Driven Development (TDD).
"""

import pytest
from src.benchmark import run_tts_benchmark


def test_run_tts_benchmark():
    """Kiểm tra benchmark chạy thành công trên các câu test mẫu."""
    test_sentences = [
        "Mời khách hàng số 23 đến quầy 5.",
        "Mời khách hàng số 08 đến quầy 10.",
    ]
    results = run_tts_benchmark(sentences=test_sentences, units_dir="data/units", verbose=False)

    assert "sentences" in results
    assert len(results["sentences"]) == 2
    for item in results["sentences"]:
        assert item["latency_ms"] > 0
        assert item["audio_duration_sec"] > 0
        assert item["rtf"] > 0
        assert item["d_spec_raw"] >= item["d_spec_smooth"]  # cross-fade phải giảm D_spec
