"""Instantaneous two-microphone FastICA; reference matching is evaluation only."""
import warnings
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning


def _pair(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=2 or x.shape[0]<4 or not np.isfinite(x).all():
        raise ValueError('Expected finite (samples, 2) audio from two channels.')
    if np.any(np.std(x,axis=0)==0):
        raise ValueError('Both channels must have nonzero variance.')
    return x


def mix_sources(sources, matrix):
    sources=_pair(sources)
    matrix=np.asarray(matrix,dtype=float)
    if matrix.shape!=(2,2) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix)<2:
        raise ValueError('Mixing matrix must be finite, 2×2 and invertible.')
    return sources@matrix.T


def separate_ica(mixtures, random_state=0, return_diagnostics=False):
    """Accept only two observations, never A, clean references or speaker labels.

    No fixed passband, time masks, denoising or reference-guided selection.
    Assumes two independent non-Gaussian sources, instantaneous stable mixing.
    """
    mixtures=_pair(mixtures)
    centered=mixtures-mixtures.mean(axis=0)
    if np.linalg.matrix_rank(centered)<2:
        raise ValueError('Two distinct microphone channels required; copied mono is rank deficient.')
    model=FastICA(n_components=2,whiten='unit-variance',random_state=random_state,
                  max_iter=2000,tol=1e-5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        estimates=model.fit_transform(mixtures)
    info=dict(converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught),
              iterations=int(model.n_iter_),input_correlation=float(np.corrcoef(mixtures.T)[0,1]),
              input_condition=float(np.linalg.cond(centered)),
              warnings=[str(w.message) for w in caught],random_state=random_state)
    return (estimates,info) if return_diagnostics else estimates


def match_sources(references, estimates):
    """Evaluation-only permutation/sign correction; never changes ICA fitting.

    Absolute zero-lag Pearson correlations before and after one-to-one matching.
    Do not treat close-talk real microphones as perfect ground truth.
    """
    references,estimates=_pair(references),_pair(estimates)
    if references.shape!=estimates.shape:
        raise ValueError('References and estimates must be aligned and have equal lengths.')
    signed=np.corrcoef(references.T,estimates.T)[:2,2:]
    rows,columns=linear_sum_assignment(-np.abs(signed))
    order=columns[np.argsort(rows)]
    aligned=estimates[:,order].copy()
    for i,j in enumerate(order):
        if signed[i,j]<0:aligned[:,i]*=-1
    after=np.abs(np.corrcoef(references.T,aligned.T)[:2,2:])
    return aligned,np.abs(signed),after
