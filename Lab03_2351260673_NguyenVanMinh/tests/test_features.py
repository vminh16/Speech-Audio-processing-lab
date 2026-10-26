import numpy as np

from src.config import FeatureConfig
from src.features import features_from_signal
from src.metrics import accuracy, confusion, most_confused
from src.noise import add_white_noise
from src.rejection import youden_threshold


def _tone(sr=16000, dur=0.8):
    t = np.arange(int(sr * dur)) / sr
    rng = np.random.default_rng(0)
    return 0.5 * np.sin(2 * np.pi * 440 * t) * np.hanning(t.size) + 1e-4 * rng.normal(size=t.size)


def test_feature_shapes_and_cmvn():
    y = _tone()
    f39 = features_from_signal(y, FeatureConfig())
    f13 = features_from_signal(y, FeatureConfig(use_delta=False))
    assert f39.shape[1] == 39 and f13.shape[1] == 13 and f39.shape[0] == f13.shape[0]
    assert np.allclose(f39.mean(axis=0), 0.0, atol=1e-6)


def test_noise_hits_requested_snr():
    rng = np.random.default_rng(0)
    y = np.sin(np.linspace(0, 200, 16000))
    noise = add_white_noise(y, 10.0, rng) - y
    measured = 10 * np.log10(np.mean(y**2) / np.mean(noise**2))
    assert abs(measured - 10.0) < 1.0


def test_metrics():
    labels = ("a", "b")
    cm = confusion(["a", "a", "b"], ["a", "b", "b"], labels)
    assert accuracy(["a", "a", "b"], ["a", "b", "b"]) == 2 / 3
    assert most_confused(cm, labels) == [("a", "b", 1)]


def test_youden_threshold_separates_clusters():
    pos, neg = np.array([-1.0, -1.2, -0.8]), np.array([-5.0, -4.5, -6.0])
    t = youden_threshold(pos, neg)
    assert np.all(pos >= t) and np.all(neg < t)


def test_preprocessing_stages_match_final_features():
    from src.features import preprocessing_stages

    y, cfg = _tone(), FeatureConfig()
    stages = preprocessing_stages(y, cfg)
    assert np.allclose(stages["feat"], features_from_signal(y, cfg))
    assert stages["log_mel"].shape[0] == cfg.n_mels and stages["feat_raw"].shape == stages["feat"].shape
