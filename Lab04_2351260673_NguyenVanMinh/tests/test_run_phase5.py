"""
End-to-End test for Phase 5 HMM-based synthesis pipeline.
"""

import pytest
import soundfile as sf
from pathlib import Path


def test_run_phase5_end_to_end(tmp_path):
    from src.run_phase5 import run_phase5

    out_audio = tmp_path / "audio_output" / "hmm_tts.wav"
    out_fig = tmp_path / "figures" / "04_hmm_trajectories.png"

    res = run_phase5(
        text="Mời khách hàng số 23 đến quầy 5.",
        output_audio=str(out_audio),
        output_figure=str(out_fig),
        sr=16000,
    )

    assert res["status"] == "success"
    assert out_audio.exists()
    assert out_fig.exists()
    assert res["duration_sec"] > 1.0
    assert "mlpg_smoothing_reduction" in res
