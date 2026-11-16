"""Real 23s microphones and controlled ICA experiments from the same conversation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .enhancement import load_audio, save_pcm16
from .separation import mix_sources, separate_ica, match_sources


def _load_pair(root,names):
    arrays=[load_audio(Path(root)/name) for name in names]
    if arrays[0][1]!=arrays[1][1] or arrays[0][0].shape!=arrays[1][0].shape:
        raise ValueError('Microphone channels must already be synchronized and equally long.')
    return np.column_stack([x for x,sr in arrays]),arrays[0][1]


def _export_pair(folder,names,signals,sr):
    # Common peak gain within each pair; ICA has no absolute scale.
    gain=min(1.,.95/np.max(np.abs(signals)))
    for i,name in enumerate(names):
        save_pcm16(Path(folder)/name,signals[:,i]*gain,sr)


def _envelopes(signals,sr):
    frame=round(.02*sr)
    return np.sqrt(np.mean(signals[:len(signals)//frame*frame].reshape(-1,frame,2)**2,axis=1))


def run_real_separation(data_dir,output_dir,random_state=0):
    mixtures,sr=_load_pair(data_dir,['Array1-01.wav','Array1-05.wav'])
    references,_=_load_pair(data_dir,['Headset-0.wav','Headset-1.wav'])
    # Only these two observations are passed into ICA.
    estimates,info=separate_ica(mixtures,random_state=random_state,return_diagnostics=True)
    aligned,before,after=match_sources(references,estimates)
    _,mic_cor,_=match_sources(references,mixtures)
    ref_env,est_env=_envelopes(references,sr),_envelopes(aligned,sr)
    envelope_corr=np.corrcoef(ref_env.T,est_env.T)[:2,2:]
    folder=Path(output_dir)/'real_23s'
    folder.mkdir(parents=True,exist_ok=True)
    _export_pair(folder,['mix_mic1.wav','mix_mic2.wav'],mixtures,sr)
    _export_pair(folder,['headset_reference_1.wav','headset_reference_2.wav'],references,sr)
    _export_pair(folder,['separated_1.wav','separated_2.wav'],aligned,sr)
    for name,matrix in [('reference_vs_mic',mic_cor),('correlation_before_matching',before),
                        ('correlation_after_matching',after),('rms_envelope_correlation',envelope_corr)]:
        pd.DataFrame(matrix,index=['headset_A','headset_B'],columns=['channel_1','channel_2']).to_csv(folder/f'{name}.csv')
    summary=dict(**info,duration_s=len(mixtures)/sr,
        reference_type='real close-talk headset microphones; not clean ground truth',
        mean_abs_reference_correlation=float(np.diag(after).mean()),
        minimum_abs_reference_correlation=float(np.diag(after).min()),
        rms_envelope_correlations=np.diag(envelope_corr).tolist(),
        matching='permutation/sign only, after blind ICA; no lag or gain fitted to references')
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return dict(mixtures=mixtures,references=references,estimates=aligned,sr=sr,
                before=before,after=after,mic_cor=mic_cor,summary=summary,kind='real',
                intervals=json.loads((Path(data_dir)/'intervals.json').read_text()))


def run_controlled_separation(data_dir,output_dir,noise_snr_db=20.,random_state=0):
    sources,sr=_load_pair(data_dir,['source_A.wav','source_B.wav'])
    sources=sources/np.sqrt(np.mean(sources**2,axis=0,keepdims=True))*.1
    matrices={'well':np.array([[1.,.6],[.4,1.]]),
              'near':np.array([[1.,.98],[.98,1.]])}
    rng=np.random.default_rng(random_state)
    noise=rng.normal(size=sources.shape)
    cases,rows={},[]
    for name,matrix in matrices.items():
        base=mix_sources(sources,matrix)
        for noisy in [False,True]:
            case=name+('_noisy' if noisy else '_clean')
            mixtures=base.copy()
            if noisy:
                # Synthesis only; no noise/SNR/matrix/reference enters ICA.
                scale=np.sqrt(np.mean(base**2,axis=0)/(np.mean(noise**2,axis=0)*10**(noise_snr_db/10)))
                mixtures+=noise*scale
            estimates,info=separate_ica(mixtures,random_state=random_state,return_diagnostics=True)
            aligned,before,after=match_sources(sources,estimates)
            folder=Path(output_dir)/'controlled'/case
            folder.mkdir(parents=True,exist_ok=True)
            for names,signals in [(['source_A.wav','source_B.wav'],sources),
                 (['mix_mic1.wav','mix_mic2.wav'],mixtures),(['separated_1.wav','separated_2.wav'],aligned)]:
                _export_pair(folder,names,signals,sr)
            for label,values in [('before',before),('after',after)]:
                pd.DataFrame(values,index=['source_A','source_B'],columns=['estimate_1','estimate_2']).to_csv(folder/f'correlation_{label}.csv')
            record=dict(case=case,determinant=float(np.linalg.det(matrix)),
                condition=float(np.linalg.cond(matrix)),added_noise_snr_db=noise_snr_db if noisy else None,
                mean_abs_correlation=float(np.diag(after).mean()),min_abs_correlation=float(np.diag(after).min()),
                **info)
            rows.append(record)
            cases[case]=dict(mixtures=mixtures,references=sources,estimates=aligned,sr=sr,
                before=before,after=after,summary=record,kind='controlled')
    frame=pd.DataFrame(rows)
    frame.to_csv(Path(output_dir)/'controlled_metrics.csv',index=False)
    (Path(output_dir)/'config.json').write_text(json.dumps(dict(
        matrices={k:v.tolist() for k,v in matrices.items()},noise_snr_db=noise_snr_db,random_state=random_state,
        note='Solo crops from different times of the 23s preview, overlaid artificially; not original overlap.'),indent=2)+'\n')
    return cases,frame
