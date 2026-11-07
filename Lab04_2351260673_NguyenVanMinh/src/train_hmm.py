"""Train small left-to-right Gaussian HMMs from measured word-unit features."""
import json
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from hmmlearn.hmm import GaussianHMM

MODEL = Path(__file__).resolve().parents[1] / 'data/models/hmm.json'


def train():
    root = MODEL.parents[1] / 'units'
    models = {}
    for label in sorted({p.stem.rsplit('_', 1)[0] for p in root.glob('*.wav')}):
        sequences = []
        for path in sorted(root.glob(label + '_*.wav')):
            y, sr = sf.read(path)
            frames = librosa.util.frame(np.pad(y, (320, 320)), frame_length=640, hop_length=160).T
            f0 = librosa.yin(y, fmin=65, fmax=400, sr=sr, frame_length=640, hop_length=160)
            values = []
            for i, frame in enumerate(frames[:len(f0)]):
                rms = np.sqrt(np.mean(frame**2))
                try:
                    a = librosa.lpc(frame * np.hanning(len(frame)) + 1e-9, order=16)
                    roots = np.roots(a)
                    freq = np.angle(roots) * sr / (2*np.pi)
                    bw = -np.log(np.maximum(np.abs(roots), 1e-9)) * sr / np.pi
                    formants = sorted(freq[(freq > 200) & (freq < 4000) & (bw > 0) & (bw < 700)])
                except (FloatingPointError, np.linalg.LinAlgError):
                    formants = []
                if len(formants) < 3:
                    continue
                values.append([f0[i], *formants[:3], rms])
            x = np.asarray(values)
            delta = np.vstack([np.zeros(5), np.diff(x, axis=0)])
            sequences.append(np.hstack([x, delta]))
        data = np.concatenate(sequences)
        scale = np.maximum(data.std(axis=0), 1e-3)
        scaled = data / scale
        model = GaussianHMM(3, covariance_type='diag', n_iter=40, random_state=42, init_params='', params='tmc', min_covar=0.01)
        model.startprob_ = np.array([1.,0.,0.])
        model.transmat_ = np.array([[.9,.1,0],[0,.9,.1],[0,0,1.]])
        groups = [np.concatenate([np.array_split(x / scale, 3)[i] for x in sequences]) for i in range(3)]
        model.means_ = np.array([g.mean(axis=0) for g in groups])
        model.covars_ = np.array([g.var(axis=0) + .05 for g in groups])
        model.fit(scaled, [len(x) for x in sequences])
        durations = []
        for x in sequences:
            states = model.predict(x / scale)
            durations.append([max(2, (states == i).sum()) for i in range(3)])
        models[label] = dict(means=(model.means_*scale).tolist(), variance=(model._covars_*scale**2).tolist(), duration_frames=np.mean(durations,axis=0).tolist(), transition=model.transmat_.tolist(), start=model.startprob_.tolist())
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    MODEL.write_text(json.dumps(dict(feature_order=['f0','f1','f2','f3','rms'], frame_ms=10, models=models), indent=2))

if __name__ == '__main__':
    train()
