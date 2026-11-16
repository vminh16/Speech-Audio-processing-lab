"""Data loading, dev-only ablations, frozen test evaluation and saved models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .enhancement import load_audio, rms
from .speaker import (speaker_embedding, enroll_speakers, cosine_scores,
                      choose_threshold, verification_metrics)


def load_ami_speakers(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    dataset, seen = [],set()
    for row in manifest:
        if row['split']=='preview':
            continue
        if row['path'] in seen or row['split'] not in {'enroll','dev','test'}:
            raise ValueError('Duplicate file or invalid dataset split.')
        seen.add(row['path'])
        x,sr = load_audio(root/row['path'])
        dataset.append(dict(file=Path(row['path']).name,speaker=row['speaker'],
                            split=row['split'],x=x,sr=sr,duration=len(x)/sr,
                            rms=rms(x),source_start=row['start']))
    inventory = pd.DataFrame([{k:v for k,v in row.items() if k!='x'} for row in dataset])
    if inventory.empty or set(inventory.split)!={'enroll','dev','test'}:
        raise ValueError('Dataset must contain enrollment, development and test.')
    return dataset, inventory


def _embed(rows, options):
    return np.stack([speaker_embedding(r['x'],r['sr'],**options) for r in rows])


def _identify(rows, vectors, models):
    scores,speakers = cosine_scores(vectors,models)
    truth = np.array([speakers.index(r['speaker']) for r in rows])
    predicted = scores.argmax(axis=1)
    rival = scores.copy()
    rival[np.arange(len(rows)),truth] = -np.inf
    margins = scores[np.arange(len(rows)),truth]-rival.max(axis=1)
    predictions = pd.DataFrame(dict(file=[r['file'] for r in rows],
        true=[r['speaker'] for r in rows],predicted=[speakers[i] for i in predicted],
        correct=truth==predicted,margin=margins))
    trials = pd.DataFrame([dict(file=rows[i]['file'],true=rows[i]['speaker'],claim=speaker,
                               score=float(scores[i,j]),genuine=bool(truth[i]==j))
                          for i in range(len(rows)) for j,speaker in enumerate(speakers)])
    return scores,predictions,trials,speakers


def run_speaker_experiment(dataset, output_dir, n_mfcc=20):
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    train = [r for r in dataset if r['split']=='enroll']
    dev = [r for r in dataset if r['split']=='dev']
    test = [r for r in dataset if r['split']=='test']
    if not train or not dev or not test:
        raise ValueError('Need separate enrollment/dev/test data.')
    labels = [r['speaker'] for r in train]
    ablation, candidates = [],[]
    for use_delta in [False,True]:
        for cmn in [False,True]:
            options = dict(n_mfcc=n_mfcc,use_delta=use_delta,cmn=cmn)
            train_vectors,dev_vectors = _embed(train,options),_embed(dev,options)
            models = enroll_speakers(train_vectors,labels,n_enroll=4)
            _,predictions,_,_ = _identify(dev,dev_vectors,models)
            name = 'MFCC'+('+Δ' if use_delta else '')+('+CMN' if cmn else '')
            ablation.append(dict(feature=name,use_delta=use_delta,cmn=cmn,
                dimensions=train_vectors.shape[1],dev_accuracy=float(predictions.correct.mean()),
                mean_margin=float(predictions.margin.mean())))
            candidates.append((options,train_vectors,dev_vectors,models))
    # Only dev labels/scores determine the feature choice; no test scores here.
    best = max(range(len(ablation)),key=lambda i:(ablation[i]['dev_accuracy'],
               ablation[i]['mean_margin'],-ablation[i]['dimensions']))
    options,train_vectors,dev_vectors,models = candidates[best]
    dev_scores,dev_predictions,dev_trials,speakers = _identify(dev,dev_vectors,models)
    threshold,curve = choose_threshold(dev_trials[dev_trials.genuine].score.to_numpy(),
                                       dev_trials[~dev_trials.genuine].score.to_numpy())
    enrollment_rows = []
    for n in [1,2,4]:
        small_models = enroll_speakers(train_vectors,labels,n_enroll=n)
        _,p,_,_ = _identify(dev,dev_vectors,small_models)
        enrollment_rows.append(dict(n_enroll=n,dev_accuracy=float(p.correct.mean()),mean_margin=float(p.margin.mean())))
    # Freeze features, 4 enrollment tokens and dev threshold before test evaluation.
    test_scores,test_predictions,test_trials,_ = _identify(test,_embed(test,options),models)
    verification = verification_metrics(test_trials[test_trials.genuine].score.to_numpy(),
                                        test_trials[~test_trials.genuine].score.to_numpy(),threshold)
    confusion = pd.crosstab(test_predictions['true'],test_predictions.predicted).reindex(
        index=speakers,columns=speakers,fill_value=0).fillna(0).astype(int)
    ablation = pd.DataFrame(ablation)
    enrollment = pd.DataFrame(enrollment_rows)
    for name,frame in [('dev_ablation',ablation),('dev_enrollment',enrollment),
        ('dev_predictions',dev_predictions),('dev_trials',dev_trials),('verification_curve',curve),
        ('test_predictions',test_predictions),('test_trials',test_trials),
        ('test_scores',pd.DataFrame(test_scores,index=test_predictions.file,columns=speakers))]:
        frame.to_csv(output/f'{name}.csv',index=name=='test_scores')
    confusion.to_csv(output/'confusion_matrix.csv')
    np.savez(output/'speaker_models.npz',speakers=np.array(speakers),vectors=np.stack([models[s] for s in speakers]))
    summary = dict(features=options,feature_name=str(ablation.iloc[best].feature),
        enrollment_per_speaker=4,threshold=threshold,selection='dev accuracy, then mean true-vs-rival margin',
        dev_accuracy=float(dev_predictions.correct.mean()),test_accuracy=float(test_predictions.correct.mean()),
        test_count=len(test),verification=verification)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    return dict(summary=summary,ablation=ablation,enrollment=enrollment,models=models,
                test_scores=test_scores,test_predictions=test_predictions,confusion=confusion,
                dev_trials=dev_trials,test_trials=test_trials,curve=curve,speakers=speakers)


def conversation_intervals(annotations, duration):
    """Partition manual labels into exclusive speakers, overlap and silence."""
    if duration <= 0:
        raise ValueError('Conversation duration must be positive.')
    boundaries = sorted({0.,float(duration)} | {
        min(duration,max(0.,r[k])) for r in annotations for k in ['start','end']})
    intervals = []
    for start,end in zip(boundaries[:-1],boundaries[1:]):
        active = {r['speaker'] for r in annotations if r['start'] < end and r['end'] > start}
        speaker = next(iter(active)) if len(active)==1 else ('overlap' if active else 'silence')
        if intervals and intervals[-1]['speaker']==speaker:
            intervals[-1]['end'] = end
        else:
            intervals.append(dict(start=start,end=end,speaker=speaker))
    return intervals


def evaluate_conversation(root, models, options, output_dir, min_duration=1.):
    """Known manual turn boundaries + frozen models; NOT automatic diarization.

    Exclude overlap, short turns, silence and speakers absent from enrollment.
    This supplementary demo never updates feature selection, models or threshold.
    """
    from xml.etree import ElementTree as ET
    root,output = Path(root),Path(output_dir)
    if min_duration <= 0:
        raise ValueError('Minimum turn duration must be positive.')
    manifest = json.loads((root/'manifest.json').read_text())
    preview = next(r for r in manifest if r['split']=='preview')
    # Do not quietly evaluate on any enrollment/dev/test interval.
    if any(r['split']!='preview' and r['meeting']==preview['meeting'] and
           r['start']<preview['end'] and r['end']>preview['start'] for r in manifest):
        raise ValueError('Conversation overlaps data used for model selection/evaluation.')
    identities = {r['ami_agent']:r['speaker'] for r in manifest if 'ami_agent' in r}
    audio,sr = load_audio(root/preview['path'])
    duration = len(audio)/sr
    annotations = []
    for path in sorted((root/'metadata').glob(f"{preview['meeting']}.*.segments.xml")):
        agent = path.name.split('.')[1]
        for segment in ET.parse(path).getroot():
            start = max(0.,float(segment.attrib['transcriber_start'])-preview['start'])
            end = min(duration,float(segment.attrib['transcriber_end'])-preview['start'])
            if end>start:
                annotations.append(dict(start=start,end=end,speaker=identities.get(agent,f'unknown_{agent}')))
    if not annotations:
        raise ValueError('No manual annotations found for preview.')
    intervals = conversation_intervals(annotations,duration)
    eligible = [r for r in intervals if r['speaker'] in models and r['end']-r['start']>=min_duration]
    if not eligible:
        raise ValueError('No sufficiently long single-speaker intervals to evaluate.')
    embeddings = np.stack([speaker_embedding(audio[round(r['start']*sr):round(r['end']*sr)],sr,**options)
                           for r in eligible])
    scores,speakers = cosine_scores(embeddings,models)
    rows = []
    for index,turn in enumerate(eligible):
        order = np.argsort(scores[index])[::-1]
        predicted = speakers[order[0]]
        rows.append(dict(turn=index+1,start=turn['start'],end=turn['end'],
                         duration=turn['end']-turn['start'],true=turn['speaker'],predicted=predicted,
                         correct=predicted==turn['speaker'],score=float(scores[index,order[0]]),
                         margin=float(scores[index,order[0]]-scores[index,order[1]])))
    predictions = pd.DataFrame(rows)
    coverage = float(predictions.duration.sum())
    summary = dict(duration_s=duration,evaluated_turns=len(predictions),
        correct_turns=int(predictions.correct.sum()),accuracy=float(predictions.correct.mean()),
        evaluated_seconds=coverage,coverage=coverage/duration,
        duration_weighted_accuracy=float(predictions.loc[predictions.correct,'duration'].sum()/coverage),
        overlap_seconds=sum(r['end']-r['start'] for r in intervals if r['speaker']=='overlap'),
        min_duration=min_duration,mode='manual boundaries, frozen enrollment models; supplementary demo only')
    output.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(output/'conversation_predictions.csv',index=False)
    pd.DataFrame(scores,index=predictions.turn,columns=speakers).to_csv(output/'conversation_scores.csv')
    pd.DataFrame(annotations).to_csv(output/'conversation_annotations.csv',index=False)
    pd.DataFrame(intervals).to_csv(output/'conversation_intervals.csv',index=False)
    (output/'conversation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return dict(audio=audio,sr=sr,annotations=annotations,intervals=intervals,
                predictions=predictions,scores=scores,speakers=speakers,summary=summary)
