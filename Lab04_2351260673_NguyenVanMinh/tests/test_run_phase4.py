"""
End-to-End test for Phase 4 Prosody Modification and Voice Conversion.
"""

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path


def test_run_phase4_end_to_end(tmp_path):
    from src.run_phase4 import run_phase4

    sr = 16000
    # Create sample synthetic audio
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    # Sine wave with fundamental 150Hz
    audio = 0.1 * np.sin(2 * np.pi * 150.0 * t).astype(np.float32)
    in_wav = tmp_path / "best_concat.wav"
    sf.write(str(in_wav), audio, sr, subtype="PCM_16")

    sample_unit = tmp_path / "ba_01.wav"
    sf.write(str(sample_unit), audio[: int(sr * 0.4)], sr, subtype="PCM_16")

    out_dir = tmp_path / "audio_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = tmp_path / "figures" / "04_prosody_analysis.png"

    res = run_phase4(
        input_audio=str(in_wav),
        sample_unit=str(sample_unit),
        output_dir=str(out_dir),
        output_figure=str(fig_path),
        sr=sr,
    )

    assert res["status"] == "success"
    assert (out_dir / "speed_0.8x.wav").exists()
    assert (out_dir / "speed_1.2x.wav").exists()
    assert (out_dir / "pitch_minus2.wav").exists()
    assert (out_dir / "pitch_plus2.wav").exists()
    assert (out_dir / "voice_female_sim.wav").exists()
    assert (out_dir / "voice_southern_sim.wav").exists()
    assert fig_path.exists()
    assert "f0_verification" in res
    assert res["f0_verification"]["error_pct"] < 5.0
