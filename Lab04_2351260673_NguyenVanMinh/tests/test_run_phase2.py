"""
End-to-End test for Phase 2 pipeline.
"""

import numpy as np
import pytest
import soundfile as sf
from src.run_phase2 import run_phase2


def test_run_phase2_end_to_end(tmp_path):
    units_dir = tmp_path / "units"
    units_dir.mkdir()
    out_audio = tmp_path / "audio_output"
    out_fig = tmp_path / "figures" / "02_boundary_spec.png"

    # Tạo mock units cho các từ trong câu: "Mời khách hàng số 23 đến quầy 5."
    # Tokens: ['moi', 'khach', 'hang', 'so', 'hai', 'ba', 'den', 'quay', 'nam']
    words = ["moi", "khach", "hang", "so", "hai", "ba", "den", "quay", "nam"]
    sr = 16000
    for i, w in enumerate(words):
        freq = 200 + i * 50
        t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
        y = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        sf.write(str(units_dir / f"{w}_01.wav"), y, sr, subtype="PCM_16")

    res = run_phase2(
        text="Mời khách hàng số 23 đến quầy 5.",
        units_dir=str(units_dir),
        output_audio_dir=str(out_audio),
        output_figure=str(out_fig),
        sr=sr,
    )

    assert res["status"] == "success"
    assert (out_audio / "raw_concat.wav").exists()
    assert (out_audio / "best_concat.wav").exists()
    assert out_fig.exists()
    assert len(res["experiment_results"]) == 4
