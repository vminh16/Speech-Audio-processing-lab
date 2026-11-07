"""Check that production entry points use learned parameters and offline weights."""
import json
import numpy as np
import pytest
from src.neural_tts import MODEL, load_voice, synthesize
from src.hmm_tts import generate_hmm_trajectories


def test_piper_uses_onnx_and_reports_provenance():
    assert MODEL.stat().st_size > 1_000_000
    voice = load_voice()
    assert voice.session.get_providers() == ['CPUExecutionProvider']
    _, _, extra = synthesize('Mời khách hàng số 23 đến quầy 5')
    assert 'VITS pretrained' in extra['model_architecture']
    assert 'output waveform' in extra['mel_source']
    assert extra['model_size_mb'] > 1
    assert extra['acoustic_latency_ms'] is None


def test_hmm_model_is_trained_and_controls_are_validated():
    data = json.loads((MODEL.parents[1] / 'hmm.json').read_text())
    assert len(data['models']) == 16
    for item in data['models'].values():
        assert np.asarray(item['means']).shape == (3, 10)
        assert (np.asarray(item['variance']) > 0).all()
        assert np.allclose(np.sum(item['transition'], axis=1), 1)
    with pytest.raises(ValueError):
        generate_hmm_trajectories('Mời', speed=0)
