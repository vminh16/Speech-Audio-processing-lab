"""Frequency audit: ground-truth spectra and exact frozen-mask attribution.

Clean is used only for diagnostics; it never enters the enhancement algorithm.
Run from project root: MPLCONFIGDIR=/tmp/lab5-mpl .venv/bin/python tools/analyze_enhancement.py
"""
from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft, istft, welch
from src.enhancement import load_audio, add_noise_at_snr, spectral_subtraction, wiener_filter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/diagnostics'
SR = 16000
BANDS = [(0, 300), (300, 1000), (1000, 2000), (2000, 4000), (4000, 8001)]
KW = dict(window='hann', nperseg=400, noverlap=240, nfft=512, boundary='zeros', padded=True)


def spectrum(x):
    return welch(x, fs=SR, window='hann', nperseg=1024, noverlap=512, scaling='density')


def band_power(x, low=0, high=8001):
    f, p = spectrum(x)
    return float(np.sum(p[(f >= low) & (f < high)]) * (f[1]-f[0]))


def ratio(a, b):
    return float(10*np.log10(max(a, 1e-30)/max(b, 1e-30)))


def inverse(z, length):
    return istft(z, window='hann', nperseg=400, noverlap=240, nfft=512,
                 boundary=True)[1][:length]


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/'results/audio/enhancement/config.json').read_text())
    best_ss = config['ss']
    clean, _ = load_audio(ROOT/'data/clean'/config['experiment']['clean'])
    f, times, S = stft(clean, fs=SR, **KW)
    rows, total_rows = [], []
    for noise_name in ['rain', 'vacuum_cleaner']:
        raw, _ = load_audio(ROOT/f'data/noise/{noise_name}.wav')
        for target in [0, 5, 10]:
            noisy, noise, _ = add_noise_at_snr(clean, raw, target)
            _, _, Y = stft(noisy, fs=SR, **KW)
            _, _, V = stft(noise, fs=SR, **KW)
            ss_output, ss_info = spectral_subtraction(noisy, **best_ss, return_diagnostics=True)
            w_output, w_info = wiener_filter(noisy, return_diagnostics=True)
            ss_gain, wiener_gain = ss_info['gain'], w_info['gain']
            outputs = {'SS': ss_output, 'Wiener': w_output}
            decomposed = {}
            for method, gain in [('SS', ss_gain), ('Wiener', wiener_gain)]:
                # Freeze G estimated from noisy; linear decomposition G*(S+V).
                passed_speech = inverse(gain*S, len(clean))
                passed_noise = inverse(gain*V, len(clean))
                distortion = passed_speech-clean
                error = outputs[method]-clean
                reconstruction_error = np.max(abs(passed_speech+passed_noise-outputs[method]))
                assert reconstruction_error < 1e-10
                decomposed[method] = (passed_speech, passed_noise, distortion, error, gain)
                for low, high in BANDS:
                    original_noise = band_power(noise, low, high)
                    rows.append(dict(noise=noise_name, input_snr=target, method=method,
                        band_hz=f'{low}-{min(high,8000)}',
                        input_band_snr_db=ratio(band_power(clean,low,high), original_noise),
                        noise_reduction_db=ratio(original_noise,band_power(passed_noise,low,high)),
                        error_reduction_db=ratio(original_noise,band_power(error,low,high)),
                        output_to_clean_db=ratio(band_power(outputs[method],low,high),band_power(clean,low,high)),
                        speech_distortion_db=ratio(band_power(distortion,low,high),band_power(clean,low,high))))
                total_rows.append(dict(noise=noise_name,input_snr=target,method=method,
                    noise_reduction_db=ratio(band_power(noise),band_power(passed_noise)),
                    speech_distortion_db=ratio(band_power(distortion),band_power(clean)),
                    error_reduction_below4k_db=ratio(band_power(noise,0,4000),band_power(error,0,4000)),
                    noise_power_above4k_pct=100*band_power(noise,4000,8001)/band_power(noise),
                    speech_power_above4k_pct=100*band_power(clean,4000,8001)/band_power(clean),
                    decomposition_error=reconstruction_error))
            if target != 5:
                continue
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True, sharex=True)
            for label, x in {'Clean':clean, 'Added noise':noise,'Noisy':noisy, **outputs}.items():
                freq, p = spectrum(x)
                axes[0].plot(freq,10*np.log10(np.maximum(p,1e-16)),label=label,lw=1)
            axes[0].set(title=f'{noise_name}, input 5 dB — blind enhancement — absolute Welch PSD',
                        ylabel='PSD (dBFS/Hz)',ylim=(-110,-20))
            axes[0].legend(ncol=5)
            freq, pv = spectrum(noise)
            axes[1].plot(freq,10*np.log10(np.maximum(pv,1e-16)),label='Original noise',color='gray')
            for method, (_,passed_noise,distortion,error,_) in decomposed.items():
                for suffix,x,style in [('remaining noise',passed_noise,'-'),('speech distortion',distortion,':')]:
                    _,p=spectrum(x)
                    axes[1].plot(freq,10*np.log10(np.maximum(p,1e-16)),style,label=f'{method}: {suffix}',lw=1)
                _, pe=spectrum(error)
                axes[2].plot(freq,10*np.log10(np.maximum(pv,1e-16)/np.maximum(pe,1e-16)),label=method,lw=1)
            axes[1].set(ylabel='PSD (dBFS/Hz)',ylim=(-110,-20),title='Frozen-mask attribution: residual noise vs damage to speech')
            axes[1].legend(ncol=3)
            axes[2].axhline(0,color='gray',ls='--')
            axes[2].set(ylabel='Error reduction (dB)',xlabel='Frequency (Hz)',ylim=(-10,20),
                        title='Positive = less error than noisy; negative = worse')
            axes[2].legend()
            for ax in axes:
                ax.axvline(4000,color='black',ls='--',alpha=.4)
                ax.set_xlim(0,8000)
                ax.grid(alpha=.2)
            fig.savefig(OUT/f'{noise_name}_frequency_audit.png',dpi=160)
            plt.close(fig)
            # Shows isolated surviving spectral peaks that long-time PSD hides.
            fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,sharey=True,constrained_layout=True)
            reference=np.max(abs(V)**2)
            for ax,(label,z) in zip(axes,[('Added noise',V),('SS remaining noise',ss_gain*V),('Wiener remaining noise',wiener_gain*V)]):
                img=ax.pcolormesh(times,f,10*np.log10(np.maximum(abs(z)**2/reference,1e-10)),
                                  shading='auto',cmap='magma',vmin=-60,vmax=0)
                ax.set(title=label,ylabel='Hz',ylim=(0,4000))
            axes[-1].set_xlabel('Time (s)')
            fig.colorbar(img,ax=list(axes),label='Power dB, shared noise reference')
            fig.savefig(OUT/f'{noise_name}_residual_noise.png',dpi=160)
            plt.close(fig)
    bands=pd.DataFrame(rows)
    totals=pd.DataFrame(total_rows)
    bands.to_csv(OUT/'frequency_bands.csv',index=False)
    totals.to_csv(OUT/'frequency_totals.csv',index=False)
    print('BANDS at 5 dB:\n',bands[bands.input_snr==5].round(2).to_string(index=False))
    print('\nTOTALS:\n',totals.round(4).to_string(index=False))
    return bands,totals

if __name__=='__main__':
    audit()
