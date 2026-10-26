import numpy as np
import pytest

from src.dtw import DTWRecognizer, dtw_distance


def test_identical_sequences_have_zero_distance():
    a = np.random.default_rng(0).normal(size=(20, 5))
    assert dtw_distance(a, a) == pytest.approx(0.0)


def test_symmetric_and_handles_time_warp():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(15, 4))
    stretched = np.repeat(a, 2, axis=0)  # cùng nội dung nhưng chậm gấp đôi
    other = rng.normal(size=(30, 4))
    assert dtw_distance(a, other) == pytest.approx(dtw_distance(other, a))
    assert dtw_distance(a, stretched) < dtw_distance(a, other)


def test_path_is_monotone_and_spans_both_sequences():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(8, 3)), rng.normal(size=(11, 3))
    _, path = dtw_distance(a, b, return_path=True)
    assert path[0] == (0, 0) and path[-1] == (7, 10)
    assert all(i2 >= i1 and j2 >= j1 for (i1, j1), (i2, j2) in zip(path, path[1:]))


def test_recognizer_picks_nearest_class():
    rng = np.random.default_rng(3)
    proto = {"x": rng.normal(size=(12, 4)), "y": rng.normal(size=(12, 4)) + 5}
    rec = DTWRecognizer().fit({k: [v + 0.01 * rng.normal(size=v.shape)] for k, v in proto.items()})
    assert rec.predict(proto["y"] + 0.02)[0] == "y"
