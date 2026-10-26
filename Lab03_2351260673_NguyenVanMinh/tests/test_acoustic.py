import numpy as np

from src.acoustic import check_topology, fit_word_model, left_right_matrices, make_lr_hmm
from src.config import HMMConfig
from src.decode import recognize, state_durations, viterbi_path


def _utterances(rng, offsets, n=4, frames=40, dim=6):
    """Mỗi utterance gồm các đoạn có mean khác nhau theo thời gian (giống phát âm nhiều âm tiết)."""
    seqs = []
    for _ in range(n):
        parts = [rng.normal(loc=o, scale=0.3, size=(frames // len(offsets), dim)) for o in offsets]
        seqs.append(np.vstack(parts))
    return seqs


def test_left_right_matrices():
    pi, A = left_right_matrices(4, self_loop=0.6)
    assert pi.tolist() == [1, 0, 0, 0]
    assert np.allclose(A.sum(axis=1), 1.0)
    assert A[3, 3] == 1.0 and A[0, 1] == 0.4 and np.triu(A, 2).sum() == 0


def test_make_lr_hmm_sets_start_and_transitions():
    model = make_lr_hmm(n_states=3)
    assert model.startprob_[0] == 1.0 and model.transmat_[2, 2] == 1.0


def test_training_preserves_topology_and_viterbi_is_monotone():
    rng = np.random.default_rng(0)
    cfg = HMMConfig(n_states=3)
    model, info = fit_word_model(_utterances(rng, [0.0, 3.0, -3.0]), cfg)
    topo = check_topology(model)
    assert topo["no_backward"] and topo["no_skip"] and topo["rows_sum_to_1"] and info.converged
    _, states = viterbi_path(_utterances(rng, [0.0, 3.0, -3.0], n=1)[0], model)
    assert np.all(np.diff(states) >= 0) and states[0] == 0
    assert state_durations(states, 3).sum() == len(states)


def test_recognize_separates_two_words():
    rng = np.random.default_rng(1)
    cfg = HMMConfig(n_states=3)
    models = {
        "w1": fit_word_model(_utterances(rng, [0.0, 3.0, -3.0]), cfg)[0],
        "w2": fit_word_model(_utterances(rng, [-3.0, 0.0, 3.0]), cfg)[0],
    }
    r = recognize(_utterances(rng, [-3.0, 0.0, 3.0], n=1)[0], models)
    assert r.pred == "w2" and r.margin > 0
