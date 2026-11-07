"""
Unit tests for HMM-based Parametric Speech Synthesis (Phase 5).
"""

import numpy as np
import pytest
from src.hmm_tts import (
    generate_hmm_trajectories,
    synthesize_hmm,
    apply_mlpg_smoothing,
)


def test_generate_hmm_trajectories():
    text = "Mời khách hàng số 23"
    tracks = generate_hmm_trajectories(text, sr=16000, frame_shift_ms=10)

    assert "f0" in tracks
    assert "f1" in tracks
    assert "f2" in tracks
    assert "energy" in tracks
    assert "time" in tracks
    assert "states" in tracks
    assert len(tracks["f0"]) == len(tracks["time"])
    assert len(tracks["f1"]) == len(tracks["time"])
    assert np.all(tracks["f0"] >= 0)


def test_apply_mlpg_smoothing():
    # Step-like static mean values
    step_vals = np.array([120.0] * 20 + [160.0] * 20 + [140.0] * 20)
    smoothed = apply_mlpg_smoothing(step_vals, delta_weight=0.5)

    assert len(smoothed) == len(step_vals)
    # Smoothed trajectory should eliminate step jump
    assert abs(smoothed[19] - smoothed[20]) < abs(step_vals[19] - step_vals[20])


def test_synthesize_hmm():
    text = "Mời khách hàng số 23 đến quầy 5"
    y, sr, tracks = synthesize_hmm(text, sr=16000, speed=1.0, pitch_scale=1.0)

    assert sr == 16000
    assert isinstance(y, np.ndarray)
    assert y.ndim == 1
    assert len(y) > sr * 1.0  # Duration > 1 second
    assert np.max(np.abs(y)) > 0.01  # Non-silent
    assert not np.isnan(y).any()


def test_synthesize_hmm_parameter_control():
    text = "Mời số 5"
    y_normal, sr, _ = synthesize_hmm(text, sr=16000, speed=1.0, pitch_scale=1.0)
    y_fast, _, _ = synthesize_hmm(text, sr=16000, speed=1.3, pitch_scale=1.0)

    # Fast duration should be shorter
    assert len(y_fast) < len(y_normal)
