"""
End-to-End test for Phase 6 Pretrained Neural TTS pipeline.
"""

import pytest
import soundfile as sf
from pathlib import Path


def test_run_phase6_end_to_end(tmp_path):
    from src.run_phase6 import run_phase6

    out_audio = tmp_path / "audio_output" / "neural_tts.wav"
    out_fig = tmp_path / "figures" / "05_neural_melspec.png"

    res = run_phase6(
        output_audio=str(out_audio),
        output_figure=str(out_fig),
        sr=16000,
    )

    assert res["status"] == "success"
    assert out_audio.exists()
    assert out_fig.exists()
    assert len(res["benchmarks"]) == 3
    for b in res["benchmarks"]:
        assert b["latency_ms"] > 0
        assert b["rtf"] > 0
