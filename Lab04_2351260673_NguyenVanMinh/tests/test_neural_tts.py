"""
Unit tests for Offline Neural TTS module (Phase 6).
"""

import numpy as np
import pytest
from src.neural_tts import synthesize, plot_neural_melspec


def test_neural_tts_synthesis():
    text = "Mời khách hàng số 23 đến quầy 5."
    y_nn, sr, extra = synthesize(text, sr=16000)

    assert sr == 16000
    assert isinstance(y_nn, np.ndarray)
    assert y_nn.ndim == 1
    assert len(y_nn) > sr * 1.0  # Duration > 1.0s
    assert np.max(np.abs(y_nn)) > 0.05
    assert not np.isnan(y_nn).any()

    assert "mel" in extra
    assert "latency_ms" in extra
    assert "rtf" in extra
    assert extra["mel"].shape[0] == 80  # 80-band Mel spectrogram
    assert extra["latency_ms"] > 0


def test_neural_tts_three_test_sentences():
    sentences = [
        "Mời khách hàng số 23 đến quầy 5.",
        "Mời khách hàng số 10 đến quầy 2.",
        "Khách hàng số 49 đến quầy 8.",
    ]
    for s in sentences:
        y, sr, extra = synthesize(s, sr=16000)
        assert len(y) > 0
        assert extra["rtf"] > 0


def test_plot_neural_melspec(tmp_path):
    text = "Mời khách hàng số 23 đến quầy 5."
    y_nn, sr, extra = synthesize(text, sr=16000)
    fig_path = tmp_path / "05_neural_melspec.png"
    plot_neural_melspec(extra["mel"], y_audio=y_nn, text=text, sr=sr, output_fig=str(fig_path))

    assert fig_path.exists()
