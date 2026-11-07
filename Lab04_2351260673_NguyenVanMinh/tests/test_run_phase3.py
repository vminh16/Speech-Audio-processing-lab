"""
End-to-End test for Phase 3 Unit Selection pipeline.
"""

import numpy as np
import pytest
import soundfile as sf
from src.run_phase3 import run_phase3


def test_run_phase3_end_to_end(tmp_path):
    units_dir = tmp_path / "units"
    units_dir.mkdir()
    out_audio = tmp_path / "audio_output" / "best_concat.wav"
    out_fig = tmp_path / "figures" / "03_unit_selection.png"

    # Tạo kho unit giả lập với 2 candidates cho các từ
    words = ["moi", "khach", "hang", "so", "hai", "ba", "den", "quay", "nam"]
    sr = 16000
    for w in words:
        for cid in ["01", "02"]:
            freq = 300 if cid == "01" else 350
            t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
            y = 0.08 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            sf.write(str(units_dir / f"{w}_{cid}.wav"), y, sr, subtype="PCM_16")

    res = run_phase3(
        text="Mời khách hàng số 23 đến quầy 5.",
        units_dir=str(units_dir),
        output_audio=str(out_audio),
        output_figure=str(out_fig),
        sr=sr,
    )

    assert res["status"] == "success"
    assert out_audio.exists()
    assert out_fig.exists()
    assert len(res["optimal_path"]) == 9
    assert res["cost_reduction_pct"] >= 0.0
