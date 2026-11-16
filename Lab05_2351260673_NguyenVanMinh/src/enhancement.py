"""Parts 4.1–4.3: additive noise, STFT enhancement, metrics, and PCM export."""
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def _signal(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty, finite mono signal.')
    return x


def power(x):
    return float(np.mean(_signal(x) ** 2))


def rms(x):
    return float(np.sqrt(power(x)))


def load_audio(path, target_sr=16000):
    """Decode as float, downmix channels, and resample when necessary."""
    x, sr = sf.read(path, dtype='float64', always_2d=True)
    x = _signal(x.mean(axis=1))
    if sr != target_sr:
        divisor = gcd(sr, target_sr)
        x = resample_poly(x, target_sr // divisor, sr // divisor)
    return x, target_sr


def match_noise_length(noise, length):
    """Repeat short noise and crop long noise; never add silence to the noise."""
    noise = _signal(noise)
    if length <= 0:
        raise ValueError('Length must be positive.')
    return np.tile(noise, int(np.ceil(length / len(noise))))[:length]


def add_noise_at_snr(clean, noise, target_db):
    """Return (noisy, actual_added_noise, alpha), without normalization/clipping."""
    clean = _signal(clean)
    noise = match_noise_length(noise, len(clean))
    ps, pv = power(clean), power(noise)
    if ps <= 0 or pv <= 0 or not np.isfinite(target_db):
        raise ValueError('Speech/noise must have positive power and SNR must be finite.')
    alpha = float(np.sqrt(ps / (pv * 10 ** (target_db / 10))))
    scaled = alpha * noise
    return clean + scaled, scaled, alpha


def snr_db(reference, observed):
    """Reference-based SNR; inputs must already be aligned and equally long."""
    reference, observed = _signal(reference), _signal(observed)
    if reference.shape != observed.shape or power(reference) <= 0:
        raise ValueError('Use aligned, equal-length signals and a nonzero reference.')
    error_power = power(observed - reference)
    return float(10 * np.log10(power(reference) / error_power)) if error_power else float('inf')


def save_pcm16(path, x, sr=16000):
    """Reject out-of-range samples rather than silently clipping PCM audio."""
    x = _signal(x)
    if np.max(np.abs(x)) >= 1:
        raise ValueError('Apply a shared headroom gain before PCM16 export (peak < 1).')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, sr, subtype='PCM_16')


def _spectra(noisy, n_fft, win_length, hop_length):
    from scipy.signal import stft, check_NOLA
    if not (0 < hop_length <= win_length <= n_fft):
        raise ValueError('Require 0 < hop_length <= win_length <= n_fft.')
    if not check_NOLA('hann', win_length, win_length - hop_length):
        raise ValueError('Window/hop must support overlap-add reconstruction.')
    noisy = _signal(noisy)
    padded = np.pad(noisy, (0, max(0, win_length-len(noisy))))
    _, _, Y = stft(padded, window='hann', nperseg=win_length,
                   noverlap=win_length-hop_length, nfft=n_fft,
                   boundary='zeros', padded=True)
    return Y, len(noisy)


def estimate_noise_psd(spectrum):
    """Offline blind estimate from the lowest-energy 20% of observed frames.

    No initial silence, clean signal, original noise, or target SNR is supplied.
    Boundary frames padded by STFT are excluded. This assumes some low-speech
    frames and roughly stationary noise; continuous speech can bias the estimate.
    """
    py = np.abs(spectrum)**2
    interior = py[:, 1:-1] if py.shape[1] > 2 else py
    energies = interior.sum(axis=0)
    count = max(1, int(np.ceil(interior.shape[1]*.2)))
    quiet = np.argsort(energies)[:count]
    # Compensate modest downward selection bias; independent of target SNR.
    pn = 1.2*np.mean(interior[:, quiet], axis=1, keepdims=True)
    return pn


def _smooth_power(py, coefficient=.8):
    from scipy.signal import lfilter
    from scipy.ndimage import uniform_filter1d
    # Smooth forward/backward for offline use; no frame-dependent noise spikes.
    smoothed = uniform_filter1d(py, size=3, axis=0, mode='nearest')
    for reverse in [False, True]:
        if reverse:
            smoothed = smoothed[:, ::-1]
        smoothed, _ = lfilter([1-coefficient], [1, -coefficient], smoothed, axis=1,
                              zi=(coefficient*smoothed[:, :1]))
        if reverse:
            smoothed = smoothed[:, ::-1]
    return smoothed


def _reconstruct(spectrum, length, n_fft, win_length, hop_length):
    from scipy.signal import istft
    _, x = istft(spectrum, window='hann', nperseg=win_length,
                 noverlap=win_length-hop_length, nfft=n_fft, boundary=True)
    return x[:length]


def spectral_subtraction(noisy, beta=1.5, floor=0.01,
                         n_fft=512, win_length=400, hop_length=160,
                         return_diagnostics=False):
    """Blind power spectral subtraction with smoothed power and noisy phase.

    Input: only noisy audio. floor is a power floor relative to estimated noise.
    Offline smoothing reduces musical noise; it is not a separate denoiser.
    """
    if not np.isfinite(beta) or beta < 1 or not np.isfinite(floor) or floor < 0:
        raise ValueError('Require finite beta >= 1 and floor >= 0.')
    Y, length = _spectra(noisy, n_fft, win_length, hop_length)
    from scipy.ndimage import uniform_filter1d
    # Match frequency smoothing on observation and PSD to preserve units/peaks.
    pn = uniform_filter1d(estimate_noise_psd(Y), size=3, axis=0, mode='nearest')
    py = _smooth_power(np.abs(Y)**2)
    estimated = np.maximum(py-beta*pn, floor*pn)
    gain = np.sqrt(np.divide(estimated, py, out=np.ones_like(py), where=py>0))
    gain = np.clip(gain, 0, 1)
    output = _reconstruct(gain*Y, length, n_fft, win_length, hop_length)
    info = dict(noise_psd=pn, gain=gain)
    return (output, info) if return_diagnostics else output


def wiener_filter(noisy, n_fft=512, win_length=400, hop_length=160,
                  return_diagnostics=False):
    """Blind Wiener gain xi/(1+xi) with decision-directed speech SNR.

    Uses previous estimated speech power, not clean speech. No noise clip or
    mixing metadata is accepted. Offline, same blind noise PSD estimate as SS.
    """
    from scipy.ndimage import uniform_filter1d
    Y, length = _spectra(noisy, n_fft, win_length, hop_length)
    pn = estimate_noise_psd(Y)
    py = np.abs(Y)**2
    epsilon = max(float(py.max())*1e-12, np.finfo(float).tiny)
    posterior = py/np.maximum(pn, epsilon)
    gain = np.ones_like(py)
    previous = np.maximum(posterior[:, 0]-1, 0)
    for t in range(py.shape[1]):
        xi = .9*previous + .1*np.maximum(posterior[:, t]-1, 0)
        current = xi/(1+xi)
        current = np.minimum(current, uniform_filter1d(current, size=3, mode='nearest'))
        gain[:, t] = np.maximum(current, .05)
        previous = gain[:, t]**2 * posterior[:, t]
    if not np.any(pn):
        gain[:] = 1
    output = _reconstruct(gain*Y, length, n_fft, win_length, hop_length)
    info = dict(noise_psd=pn, gain=gain)
    return (output, info) if return_diagnostics else output
