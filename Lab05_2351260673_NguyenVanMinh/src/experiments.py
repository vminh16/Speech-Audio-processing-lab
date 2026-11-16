"""Experiment orchestration; clean/noise are used for synthesis and scoring only."""
from pathlib import Path
import numpy as np
import pandas as pd
from .enhancement import (load_audio, add_noise_at_snr, spectral_subtraction,
                          wiener_filter, snr_db, save_pcm16)


def select_ss_config(dev_paths, noise_paths, snrs, betas, floors, stft):
    """Choose ONE global SS configuration by mean development-set output SNR."""
    if not dev_paths or any('_dev_' not in Path(p).stem for p in dev_paths):
        raise ValueError('Supply nonempty development paths only, separate from test.')
    rows = []
    for path in sorted(dev_paths):
        clean, sr = load_audio(path)
        for noise_path in noise_paths:
            noise, _ = load_audio(noise_path, sr)
            for target in snrs:
                noisy, _, _ = add_noise_at_snr(clean, noise, target)
                for beta in betas:
                    for floor in floors:
                        # The denoiser receives only the observation, never clean/noise.
                        enhanced = spectral_subtraction(noisy, beta=beta, floor=floor, **stft)
                        rows.append(dict(file=Path(path).name, noise=Path(noise_path).stem,
                                         input_snr=target, beta=beta, floor=floor,
                                         snr_out=snr_db(clean, enhanced)))
    trials = pd.DataFrame(rows)
    summary = trials.groupby(['beta', 'floor'], as_index=False).agg(
        mean_snr_out=('snr_out', 'mean'), trials=('snr_out', 'size'))
    summary = summary.sort_values(['mean_snr_out', 'beta', 'floor'], ascending=[False, True, False])
    best = summary.iloc[0]
    return dict(beta=float(best.beta), floor=float(best.floor)), summary, trials


def run_enhancement_cases(clean_path, noise_paths, snrs, ss_config, stft, output_dir):
    """Run frozen settings, export shared-gain WAVs, measure from exported audio."""
    clean, sr = load_audio(clean_path)
    cases, alphas = {}, {}
    for noise_path in noise_paths:
        noise, _ = load_audio(noise_path, sr)
        for target in snrs:
            noisy, _, alpha = add_noise_at_snr(clean, noise, target)
            key = (Path(noise_path).stem, target)
            cases[key] = {'Clean': clean, 'Noisy': noisy,
                          'SS': spectral_subtraction(noisy, **ss_config, **stft),
                          'Wiener': wiener_filter(noisy, **stft)}
            alphas[key] = alpha
    peak = max(np.max(np.abs(x)) for case in cases.values() for x in case.values())
    gain = min(1., .95/peak)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pcm16(output_dir/'clean_reference.wav', clean*gain, sr)
    reference, _ = load_audio(output_dir/'clean_reference.wav', sr)
    rows = []
    for (name, target), signals in cases.items():
        signals['Clean'] = reference
        for method in ['Noisy', 'SS', 'Wiener']:
            path = output_dir/f'{name}_{target}dB_{method.lower()}.wav'
            save_pcm16(path, signals[method]*gain, sr)
            signals[method], _ = load_audio(path, sr)
        measured_input = snr_db(reference, signals['Noisy'])
        if abs(measured_input-target) > .2:
            raise ValueError('Exported noisy WAV missed target SNR tolerance.')
        for method in ['Noisy', 'SS', 'Wiener']:
            measured_output = snr_db(reference, signals[method])
            rows.append(dict(noise=name, target_db=target, method=method,
                             alpha=alphas[(name, target)], shared_gain=gain,
                             snr_in=measured_input, snr_out=measured_output,
                             delta_snr=measured_output-measured_input,
                             peak=float(np.max(np.abs(signals[method])))))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir/'enhancement_metrics.csv', index=False)
    return cases, metrics, sr
