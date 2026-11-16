"""Single reproducible entry point for all Lab 5 experiments and figures."""
import json
from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt
from .experiments import select_ss_config,run_enhancement_cases
from .speaker_experiments import load_ami_speakers,run_speaker_experiment,evaluate_conversation
from .separation_experiments import run_real_separation,run_controlled_separation
from .integration import run_integrated_enhancement
from .visualization import (plot_dev_selection,plot_snr_summary,plot_comparison,plot_speaker_data,
                           plot_speaker_pipeline,plot_speaker_identification,plot_speaker_verification,
                           plot_conversation_demo)
from .separation_visualization import (plot_ica_pipeline,plot_separation_spectrograms,
                                      plot_correlation_matching,plot_controlled_summary)
from .integration_visualization import plot_integration_summary,plot_integration_snr,plot_integrated_conversation


DEFAULT_CONFIG=dict(stft=dict(n_fft=512,win_length=400,hop_length=160),
    noise_names=['rain','vacuum_cleaner'],snrs=[0,5,10],ss_betas=[1.,1.5,2.],ss_floors=[0.,.01,.05],
    enhancement_reference='spk01_test_01.wav',n_mfcc=20,min_turn_seconds=1.,
    ica_noise_snr_db=20.,random_state=0)


def run_full_lab(root, config=None):
    config=deepcopy(DEFAULT_CONFIG if config is None else config)
    root=Path(root).resolve()
    data,results=root/'data',root/'results'
    stft=config['stft']
    noise_paths=[data/'noise'/f'{name}.wav' for name in config['noise_names']]
    snrs=config['snrs']
    figures={}
    def save(name,fig):
        path=results/'figures'/f'{name}.png'
        path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=150)
        plt.close(fig)
        figures[name]=path
    best,dev,dev_trials=select_ss_config(list((data/'clean').glob('*_dev_*.wav')),noise_paths,
                                        snrs,config['ss_betas'],config['ss_floors'],stft)
    eout=results/'audio/enhancement'
    eout.mkdir(parents=True,exist_ok=True)
    dev.to_csv(eout/'dev_selection.csv',index=False)
    dev_trials.to_csv(eout/'dev_trials.csv',index=False)
    enhancement_config=dict(experiment=dict(clean=config['enhancement_reference'],noises=config['noise_names'],
             snrs=snrs,betas=config['ss_betas'],floors=config['ss_floors'],demo_snr=5),stft=stft,ss=best,
             wiener=dict(decision_directed=.9,amplitude_floor=.05),selection='maximum mean dev SNR; frozen before test')
    (eout/'config.json').write_text(json.dumps(enhancement_config,indent=2)+'\n')
    enhancement,enh_metrics,sr=run_enhancement_cases(data/'clean'/config['enhancement_reference'],noise_paths,snrs,best,stft,eout)
    save('enhancement/dev_selection',plot_dev_selection(dev))
    save('enhancement/snr_comparison',plot_snr_summary(enh_metrics))
    save('enhancement/rain_spectrogram',plot_comparison(enhancement[('rain',5)],sr,**stft))
    ami=data/'speaker_recognition/ami'
    dataset,inventory=load_ami_speakers(ami)
    speaker=run_speaker_experiment(dataset,results/'speaker_recognition',n_mfcc=config['n_mfcc'])
    inventory.to_csv(results/'speaker_recognition/data_inventory.csv',index=False)
    conversation=evaluate_conversation(ami,speaker['models'],speaker['summary']['features'],results/'speaker_recognition',min_duration=config['min_turn_seconds'])
    samples={s:(next(r['x'] for r in dataset if r['speaker']==s and r['split']=='enroll'),16000)
             for s in speaker['speakers']}
    save('speaker_recognition/data_overview',plot_speaker_data(inventory,samples))
    example,sr=next(iter(samples.values()))
    save('speaker_recognition/method_pipeline',plot_speaker_pipeline(example,sr,speaker['summary']['features']))
    save('speaker_recognition/identification',plot_speaker_identification(speaker))
    save('speaker_recognition/verification',plot_speaker_verification(speaker))
    save('speaker_recognition/conversation_23s',plot_conversation_demo(conversation))
    separation_data=data/'separation/ami23s'
    real=run_real_separation(separation_data,results/'separation',random_state=config['random_state'])
    controlled,ica_metrics=run_controlled_separation(separation_data,results/'separation',noise_snr_db=config['ica_noise_snr_db'],random_state=config['random_state'])
    save('separation/pipeline',plot_ica_pipeline())
    save('separation/real_23s_spectrogram',plot_separation_spectrograms(real))
    save('separation/real_23s_correlation',plot_correlation_matching(real))
    save('separation/controlled_summary',plot_controlled_summary(ica_metrics))
    save('separation/well_clean_spectrogram',plot_separation_spectrograms(controlled['well_clean']))
    save('separation/well_clean_correlation',plot_correlation_matching(controlled['well_clean']))
    integrated=run_integrated_enhancement(dataset,conversation,speaker,best,stft,noise_paths,snrs,results/'integration')
    save('integration/speaker_id_comparison',plot_integration_summary(integrated))
    save('integration/snr_comparison',plot_integration_snr(integrated))
    save('integration/conversation_timeline',plot_integrated_conversation(integrated))
    original=integrated['conversation_cases'][('original',None)]
    save('integration/original_enhancement',plot_comparison(original,integrated['sr'],**stft))
    noisy=integrated['conversation_cases'][('rain',5)]
    save('integration/rain5_enhancement',plot_comparison(
        {'Original target (ambient)':original['Raw'],**noisy},integrated['sr'],**stft))
    summary=dict(config=config,enhancement=dict(ss=best,reference='FSDD spk01_test_01',sr=16000),
         speaker=speaker['summary'],conversation=conversation['summary'],
         real_ica=real['summary'],controlled_ica=json.loads(ica_metrics.to_json(orient='records')),
         integration=integrated['summary'].where(integrated['summary'].notna(),None).to_dict('records'),
         note='SNR in integration references original ambient AMI; all model/threshold selection uses dev only.')
    # Pandas float columns retain NaN under where(None); JSON roundtrip makes nulls.
    summary['integration']=json.loads(integrated['summary'].to_json(orient='records'))
    (results/'lab_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    return dict(enhancement=enh_metrics,speaker=speaker,conversation=conversation,
                real_ica=real,controlled_ica=ica_metrics,integration=integrated,figures=figures,summary=summary)
