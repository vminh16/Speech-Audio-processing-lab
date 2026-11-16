"""Frozen speaker-ID/verification before and after blind SS/Wiener enhancement."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .enhancement import load_audio,add_noise_at_snr,spectral_subtraction,wiener_filter,save_pcm16,snr_db
from .speaker import speaker_embedding,cosine_scores


def run_integrated_enhancement(dataset,conversation,speaker_result,ss_config,stft,
                             noise_paths,snrs,output_dir):
    """Denoise full files, then score the SAME known speaker intervals.

    All features/models/threshold/SS parameters are frozen. Original AMI is an
    ambient recording, not clean ground truth; SNR is reported only when noise
    is explicitly added. No labels/boundaries/noise clips enter the denoisers.
    """
    output=Path(output_dir)
    models=speaker_result['models']
    options=speaker_result['summary']['features']
    threshold=speaker_result['summary']['threshold']
    frozen={k:v.copy() for k,v in models.items()}
    samples=[dict(scope='test',file=Path(r['file']).stem,audio=r['x'],sr=r['sr'],
                  turns=[dict(turn=1,start=0.,end=len(r['x'])/r['sr'],true=r['speaker'])])
             for r in dataset if r['split']=='test']
    samples.append(dict(scope='conversation',file='conversation_23s',audio=conversation['audio'],
                        sr=conversation['sr'],turns=conversation['predictions'].to_dict('records')))
    noises={Path(p).stem:load_audio(p)[0] for p in noise_paths}
    prepared=[]
    for sample in samples:
        variants=[('original',None,sample['audio'])]
        for name,noise in noises.items():
            for target in snrs:
                y,_,_=add_noise_at_snr(sample['audio'],noise,target)
                variants.append((name,target,y))
        for name,target,y in variants:
            signals={'Raw':y,'SS':spectral_subtraction(y,**ss_config,**stft),
                     'Wiener':wiener_filter(y,**stft)}
            prepared.append(dict(sample=sample,noise=name,target=target,signals=signals))
    # ONE gain for all methods, samples and clean references, before PCM export.
    peak=max(np.max(np.abs(x)) for case in prepared for x in case['signals'].values())
    gain=min(1.,.95/peak)
    references={}
    for sample in samples:
        path=output/'audio'/sample['scope']/f"{sample['file']}_reference.wav"
        save_pcm16(path,sample['audio']*gain,sample['sr'])
        references[(sample['scope'],sample['file'])]=load_audio(path,sample['sr'])[0]
    predictions,trials,waves=[],[],[]
    conversation_cases={}
    for case in prepared:
        sample=case['sample'];sr=sample['sr']
        reference=references[(sample['scope'],sample['file'])]
        suffix='original' if case['target'] is None else f"{case['noise']}_{case['target']:g}dB"
        for method,x in case['signals'].items():
            path=output/'audio'/sample['scope']/f"{sample['file']}_{suffix}_{method.lower()}.wav"
            save_pcm16(path,x*gain,sr)
            case['signals'][method]=load_audio(path,sr)[0]
        noisy=case['signals']['Raw']
        snr_in=snr_db(reference,noisy) if case['target'] is not None else None
        if snr_in is not None and abs(snr_in-case['target'])>.2:
            raise ValueError('Integrated noisy WAV misses SNR target.')
        if sample['scope']=='conversation':
            conversation_cases[(case['noise'],case['target'])]=case['signals']
        for method,x in case['signals'].items():
            snr_out=snr_db(reference,x) if case['target'] is not None else None
            common=dict(scope=sample['scope'],file=sample['file'],noise=case['noise'],
                        target_db=case['target'],method=method)
            waves.append(dict(**common,snr_in=snr_in,snr_out=snr_out,peak=float(np.max(np.abs(x)))))
            for turn in sample['turns']:
                start,end=round(turn['start']*sr),round(turn['end']*sr)
                embedding=speaker_embedding(x[start:end],sr,**options)
                scores,speakers=cosine_scores(embedding,models)
                scores=scores[0]
                true_index=speakers.index(turn['true'])
                predicted=speakers[int(scores.argmax())]
                rival=np.max(np.delete(scores,true_index))
                predictions.append(dict(**common,turn=turn['turn'],start=turn['start'],end=turn['end'],
                    true=turn['true'],predicted=predicted,correct=predicted==turn['true'],
                    margin=float(scores[true_index]-rival),snr_in=snr_in,snr_out=snr_out))
                for j,claim in enumerate(speakers):
                    genuine=j==true_index
                    accepted=bool(scores[j]>=threshold)
                    trials.append(dict(**common,turn=turn['turn'],true=turn['true'],claim=claim,
                        score=float(scores[j]),genuine=genuine,accepted=accepted,
                        false_accept=not genuine and accepted,false_reject=genuine and not accepted))
    predictions,trials,waves=pd.DataFrame(predictions),pd.DataFrame(trials),pd.DataFrame(waves)
    groups=['scope','noise','target_db','method']
    summary=predictions.groupby(groups,dropna=False,as_index=False).agg(
        accuracy=('correct','mean'),correct=('correct','sum'),count=('correct','size'),mean_margin=('margin','mean'))
    rates=trials.groupby(groups,dropna=False).apply(lambda f:pd.Series(dict(
        far=f.false_accept.sum()/(~f.genuine).sum(),frr=f.false_reject.sum()/f.genuine.sum())),include_groups=False).reset_index()
    summary=summary.merge(rates,on=groups).merge(waves.groupby(groups,dropna=False,as_index=False)[['snr_in','snr_out']].mean(),on=groups)
    for name,frame in [('predictions',predictions),('trials',trials),('waveforms',waves),('summary',summary)]:
        frame.to_csv(output/f'{name}.csv',index=False)
    config=dict(ss=ss_config,stft=stft,features=options,verification_threshold=threshold,
                shared_export_gain=gain,snrs=list(snrs),noise_names=list(noises),
                note='All parameters frozen before integration. Whole-file enhancement before known-turn cropping. '
                     'Original AMI contains room noise; SNR only measures explicitly added noise relative to original.')
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    assert all(np.array_equal(frozen[k],models[k]) for k in models),'Enrollment models were modified.'
    return dict(predictions=predictions,trials=trials,waveforms=waves,summary=summary,
                conversation_cases=conversation_cases,conversation=conversation,config=config,sr=conversation['sr'])
