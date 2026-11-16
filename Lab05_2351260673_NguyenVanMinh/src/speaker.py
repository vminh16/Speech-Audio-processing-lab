"""MFCC centroid/cosine baseline for Lab 5 speaker identification/verification."""
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import lfilter, stft, savgol_filter
from .enhancement import _signal


def mfcc_features(x, sr=16000, n_mfcc=20, n_mels=40, cmn=False):
    """HTK Mel triangles (unit row sum), log10 power, orthonormal DCT-II.

    Includes coefficient c0; 25ms Hann frames, 10ms hop, 512-point FFT at 16k.
    No learned parameters, VAD, pitch model, or speaker-specific preprocessing.
    """
    x = _signal(x)
    if np.mean(x*x) <= 1e-12 or not 1 <= n_mfcc <= n_mels:
        raise ValueError('Need non-silent audio and 1 <= n_mfcc <= n_mels.')
    win, hop = round(sr*.025), round(sr*.010)
    n_fft = max(512, 2**int(np.ceil(np.log2(win))))
    x = lfilter([1.,-.97],[1.],x)
    x = np.pad(x,(0,max(0,win-len(x))))
    _, _, spectrum = stft(x,fs=sr,window='hann',nperseg=win,
                          noverlap=win-hop,nfft=n_fft,boundary=None,padded=False)
    mel_edges = np.linspace(0,2595*np.log10(1+sr/2/700),n_mels+2)
    hz = 700*(10**(mel_edges/2595)-1)
    frequencies = np.fft.rfftfreq(n_fft,1/sr)
    up = (frequencies[None,:]-hz[:-2,None])/(hz[1:-1]-hz[:-2])[:,None]
    down = (hz[2:,None]-frequencies[None,:])/(hz[2:]-hz[1:-1])[:,None]
    filters = np.maximum(0,np.minimum(up,down))
    filters /= np.maximum(filters.sum(axis=1,keepdims=True),np.finfo(float).tiny)
    mel_power = filters @ (np.abs(spectrum)**2)
    coefficients = dct(10*np.log10(np.maximum(mel_power,1e-12)),type=2,norm='ortho',axis=0)[:n_mfcc]
    if cmn:
        coefficients -= coefficients.mean(axis=1,keepdims=True)
    return coefficients


def speaker_embedding(x, sr=16000, n_mfcc=20, use_delta=True, cmn=False):
    mfcc = mfcc_features(x,sr,n_mfcc=n_mfcc,cmn=cmn)
    statistics = [mfcc.mean(axis=1),mfcc.std(axis=1)]
    if use_delta:
        width = min(9,mfcc.shape[1] if mfcc.shape[1]%2 else mfcc.shape[1]-1)
        delta = (savgol_filter(mfcc,width,1,deriv=1,axis=1,mode='interp')
                 if width >= 3 else np.zeros_like(mfcc))
        statistics.extend([delta.mean(axis=1),delta.std(axis=1)])
    e = np.concatenate(statistics)
    return e/max(np.linalg.norm(e),np.finfo(float).tiny)


def enroll_speakers(embeddings, labels, n_enroll=None):
    """L2-normalize each enrollment vector, then average and normalize per speaker."""
    embeddings = np.asarray(embeddings,dtype=float)
    labels = np.asarray(labels)
    if embeddings.ndim!=2 or not len(labels) or len(labels)!=len(embeddings) or not np.isfinite(embeddings).all():
        raise ValueError('Need finite enrollment vectors and matching labels.')
    lengths = np.linalg.norm(embeddings,axis=1,keepdims=True)
    if np.any(lengths==0) or (n_enroll is not None and n_enroll<1):
        raise ValueError('Enrollment vectors and count must be nonzero.')
    embeddings = embeddings/lengths
    models = {}
    for speaker in sorted(set(labels)):
        examples = embeddings[labels==speaker]
        if n_enroll is not None:
            if len(examples)<n_enroll:
                raise ValueError(f'Insufficient enrollment for {speaker}.')
            examples = examples[:n_enroll]
        model = examples.mean(axis=0)
        if np.linalg.norm(model)==0:
            raise ValueError('Enrollment centroid is zero.')
        models[str(speaker)] = model/np.linalg.norm(model)
    return models


def cosine_scores(embeddings, models):
    embeddings = np.atleast_2d(np.asarray(embeddings,dtype=float))
    if not models or not np.isfinite(embeddings).all():
        raise ValueError('Need models and finite query vectors.')
    speakers = sorted(models)
    lengths = np.linalg.norm(embeddings,axis=1,keepdims=True)
    if np.any(lengths==0):
        raise ValueError('Query vectors must be nonzero.')
    matrix = np.stack([models[s] for s in speakers])
    matrix = matrix/np.linalg.norm(matrix,axis=1,keepdims=True)
    return np.clip((embeddings/lengths)@matrix.T,-1.,1.), speakers


def verification_metrics(genuine, impostor, threshold):
    genuine, impostor = np.asarray(genuine),np.asarray(impostor)
    if not genuine.size or not impostor.size or not np.isfinite(threshold):
        raise ValueError('Need genuine and impostor trials and a finite threshold.')
    if not np.isfinite(genuine).all() or not np.isfinite(impostor).all():
        raise ValueError('Trial scores must be finite.')
    false_accepts = int(np.sum(impostor>=threshold))
    false_rejects = int(np.sum(genuine<threshold))
    return dict(far=false_accepts/len(impostor),frr=false_rejects/len(genuine),
                false_accepts=false_accepts,false_rejects=false_rejects,
                genuine_trials=len(genuine),impostor_trials=len(impostor))


def choose_threshold(genuine, impostor):
    """Choose nearest FAR=FRR on DEV only; exact score midpoints avoid coarse grids."""
    verification_metrics(genuine,impostor,0.)
    scores = np.unique(np.r_[genuine,impostor])
    thresholds = np.r_[np.nextafter(scores[0],-np.inf),
                       (scores[:-1]+scores[1:])/2,np.nextafter(scores[-1],np.inf)]
    curve = pd.DataFrame([dict(threshold=t,**verification_metrics(genuine,impostor,t)) for t in thresholds])
    index = min(curve.index,key=lambda i:(abs(curve.loc[i,'far']-curve.loc[i,'frr']),
                                         curve.loc[i,'far']+curve.loc[i,'frr']))
    return float(curve.loc[index,'threshold']),curve
