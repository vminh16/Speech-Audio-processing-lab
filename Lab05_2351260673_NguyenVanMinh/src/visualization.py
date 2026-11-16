"""Plots share axes/color scales so clean/noisy comparisons are meaningful."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


def plot_comparison(signals, sr=16000, n_fft=512, win_length=400, hop_length=160):
    """Waveform + STFT power in dB relative to ONE reference across all panels."""
    spectra = []
    for x in signals.values():
        f, t, z = stft(x, fs=sr, window='hann', nperseg=win_length,
                       noverlap=win_length-hop_length, nfft=n_fft)
        spectra.append((f, t, np.abs(z)**2))
    reference = max(p.max() for _, _, p in spectra)
    peak = max(np.max(np.abs(x)) for x in signals.values()) * 1.05
    fig, axes = plt.subplots(len(signals), 2, figsize=(14, 2.6*len(signals)), squeeze=False,
                             constrained_layout=True)
    for row, ((label, x), (f, t, p)) in enumerate(zip(signals.items(), spectra)):
        axes[row, 0].plot(np.arange(len(x))/sr, x, lw=.55)
        axes[row, 0].set(title=label, xlabel='Time (s)', ylabel='Amplitude',
                         ylim=(-peak, peak), xlim=(0, len(x)/sr))
        db = 10*np.log10(np.maximum(p/reference, 1e-10))
        mesh = axes[row, 1].pcolormesh(t, f, db, shading='auto', cmap='magma', vmin=-80, vmax=0)
        axes[row, 1].set(xlabel='Time (s)', ylabel='Frequency (Hz)',
                         ylim=(0, sr/2), xlim=(0, len(x)/sr))
    fig.colorbar(mesh, ax=axes[:, 1], label='Power (dB relative to shared maximum)')
    return fig


def plot_dev_selection(summary):
    table = summary.pivot(index='beta', columns='floor', values='mean_snr_out')
    fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
    mesh = ax.imshow(table, origin='lower', aspect='auto', cmap='viridis')
    ax.set(xticks=range(len(table.columns)), xticklabels=table.columns,
           yticks=range(len(table.index)), yticklabels=table.index,
           xlabel='γ (power floor)', ylabel='β', title='SS: mean output SNR on dev')
    best = table.to_numpy().argmax()
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            star = ' ★' if i*len(table.columns)+j == best else ''
            ax.text(j, i, f'{table.iloc[i,j]:.2f}{star}', ha='center', va='center',
                    color='white', bbox=dict(facecolor='black', alpha=.25, edgecolor='none'))
    fig.colorbar(mesh, ax=ax, label='SNR out (dB)')
    return fig


def plot_snr_summary(metrics):
    noises = list(metrics.noise.unique())
    fig, axes = plt.subplots(1, len(noises), figsize=(6*len(noises), 3.7),
                             squeeze=False, constrained_layout=True, sharey=True)
    for ax, name in zip(axes[0], noises):
        table = metrics[metrics.noise==name].pivot(index='target_db', columns='method', values='snr_out')
        x = np.arange(len(table.index))
        for offset, (method, color) in enumerate([('Noisy','gray'), ('SS','tab:blue'), ('Wiener','tab:orange')]):
            bars = ax.bar(x+(offset-1)*.25, table[method], width=.24, label=method, color=color)
            ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
        ax.set(xticks=x, xticklabels=table.index, xlabel='Input SNR (dB)',
               ylabel='Output SNR (dB)', title=name)
        ax.margins(y=.15)
        ax.legend()
    return fig


def plot_spectrum_comparison(signals, sr=16000, title=''):
    """Absolute Welch PSD, no separate normalization; right panel is reference error."""
    from scipy.signal import welch
    settings = dict(fs=sr, window='hann', nperseg=min(1024,len(signals['Clean'])), scaling='density')
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    colors = {'Clean':'black', 'Noisy':'gray', 'SS':'tab:blue', 'Wiener':'tab:orange'}
    for label, x in signals.items():
        f, p = welch(x, **settings)
        axes[0].plot(f, 10*np.log10(np.maximum(p, 1e-15)), label=label,
                     color=colors[label], lw=1.6 if label=='Clean' else 1, alpha=.85)
        if label != 'Clean':
            _, error = welch(x-signals['Clean'], **settings)
            axes[1].plot(f, 10*np.log10(np.maximum(error,1e-15)), label=f'{label} − Clean',
                         color=colors[label], lw=1, alpha=.85)
    for ax in axes:
        ax.set(xlim=(0,sr/2), ylim=(-115,-20), xlabel='Frequency (Hz)', ylabel='PSD (dBFS/Hz)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_title(f'{title} — spectrum vs ground truth')
    axes[1].set_title('Residual error spectrum (lower is better)')
    return fig


def plot_speaker_data(inventory, samples):
    """Counts/durations plus one enrollment waveform per speaker, same amplitude scale."""
    fig = plt.figure(figsize=(13,6),constrained_layout=True)
    grid = fig.add_gridspec(2,6)
    count_ax, duration_ax = fig.add_subplot(grid[0,:3]),fig.add_subplot(grid[0,3:])
    speakers = sorted(inventory.speaker.unique())
    counts = inventory.groupby(['speaker','split']).size().unstack(fill_value=0)
    bottom = np.zeros(len(speakers))
    for split,color in [('enroll','tab:blue'),('dev','tab:orange'),('test','tab:green')]:
        values = counts.reindex(speakers)[split].to_numpy()
        count_ax.bar(speakers,values,bottom=bottom,label=split,color=color)
        bottom += values
    count_ax.set(title='Data split per speaker',ylabel='Clips')
    count_ax.legend(ncol=3,fontsize=8)
    for offset,split in enumerate(['enroll','dev','test']):
        frame = inventory[inventory.split==split]
        duration_ax.scatter([speakers.index(s)+(offset-1)*.12 for s in frame.speaker],
                            frame.duration,label=split,alpha=.75)
    duration_ax.set(xticks=range(len(speakers)),xticklabels=speakers,
                    ylabel='Duration (s)',title='Clip lengths',ylim=(0,4.2))
    duration_ax.legend(ncol=3,fontsize=8)
    peak = max(np.max(np.abs(x)) for x,sr in samples.values())*1.05
    for i,(speaker,(x,sr)) in enumerate(samples.items()):
        ax = fig.add_subplot(grid[1,2*i:2*i+2])
        ax.plot(np.arange(len(x))/sr,x,lw=.55)
        ax.set(title=f'{speaker}: enrollment example',xlabel='Time (s)',ylabel='Amplitude',ylim=(-peak,peak))
    return fig


def plot_speaker_pipeline(x, sr, options):
    from .speaker import mfcc_features, speaker_embedding
    M = mfcc_features(x,sr,n_mfcc=options['n_mfcc'],cmn=options['cmn'])
    e = speaker_embedding(x,sr,**options)
    fig = plt.figure(figsize=(13,4.7),constrained_layout=True)
    grid = fig.add_gridspec(2,2,height_ratios=[1,3])
    ax = fig.add_subplot(grid[0,:])
    ax.set_axis_off()
    labels = ['Audio\n25ms / 10ms','MFCC\n20 × frames',
              'Mean / std\n'+('MFCC + Δ' if options['use_delta'] else 'MFCC'),
              f'L2 vector\n{len(e)} dimensions','Cosine\nenroll models','ID: argmax\nVerify: score ≥ θ']
    positions = np.linspace(.07,.93,len(labels))
    for index,(position,label) in enumerate(zip(positions,labels)):
        ax.text(position,.5,label,ha='center',va='center',fontsize=9,transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=.45',facecolor='#e9f2fa',edgecolor='#5781a3'))
        if index:
            ax.annotate('',xy=(position-.06,.5),xytext=(positions[index-1]+.06,.5),
                        xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#5781a3'))
    ax = fig.add_subplot(grid[1,0])
    im = ax.imshow(M[1:],origin='lower',aspect='auto',extent=[0,len(x)/sr,.5,M.shape[0]-.5],cmap='coolwarm')
    ax.set(xlabel='Time (s)',ylabel='MFCC coefficient',title='MFCC c1–c19 (c0 omitted from this plot only)')
    fig.colorbar(im,ax=ax,label='Coefficient value')
    ax = fig.add_subplot(grid[1,1])
    ax.bar(np.arange(len(e)),e,width=1)
    ax.set(xlabel='Embedding dimension',ylabel='L2-normalized value',title='Utterance representation (includes c0)')
    return fig


def plot_speaker_identification(result):
    scores = result['test_scores']
    predictions, speakers = result['test_predictions'],result['speakers']
    fig, axes = plt.subplots(1,3,figsize=(15,4),constrained_layout=True,
                              gridspec_kw={'width_ratios':[1.3,1,1]})
    im = axes[0].imshow(scores,aspect='auto',cmap='viridis')
    axes[0].set(xticks=range(len(speakers)),xticklabels=speakers,
                yticks=range(len(predictions)),yticklabels=predictions.file,
                title='Test cosine scores (not probabilities)',xlabel='Enrollment model')
    axes[0].tick_params(axis='y',labelsize=8)
    for i,row in enumerate(predictions.itertuples()):
        for j,speaker in enumerate(speakers):
            label = f'{scores[i,j]:.6f}'+(' ✓' if speaker==row.true else '')
            axes[0].text(j,i,label,ha='center',va='center',fontsize=7,color='white',
                         bbox=dict(facecolor='black',alpha=.2,edgecolor='none'))
    fig.colorbar(im,ax=axes[0],fraction=.04,format='%.4f')
    cm = result['confusion'].to_numpy()
    axes[1].imshow(cm,cmap='Blues',vmin=0,vmax=max(1,cm.max()))
    axes[1].set(xticks=range(len(speakers)),xticklabels=speakers,
                yticks=range(len(speakers)),yticklabels=speakers,
                xlabel='Predicted',ylabel='True',title=f'Test accuracy: {result["summary"]["test_accuracy"]:.1%}')
    for i in range(len(speakers)):
        for j in range(len(speakers)):
            axes[1].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    frame = result['ablation']
    bars = axes[2].bar(np.arange(len(frame)),frame.dev_accuracy*100,color='tab:blue')
    axes[2].bar_label(bars,fmt='%.0f%%',fontsize=8,padding=2)
    axes[2].set(xticks=range(len(frame)),xticklabels=frame.feature,ylim=(0,115),
                ylabel='Accuracy (%)',title='Feature ablation — DEV only')
    axes[2].tick_params(axis='x',rotation=35,labelsize=8)
    return fig


def plot_speaker_verification(result):
    fig,axes = plt.subplots(1,3,figsize=(14,3.5),constrained_layout=True)
    trials,curve = result['dev_trials'],result['curve']
    threshold = result['summary']['threshold']
    bins = np.linspace(trials.score.min()-1e-6,trials.score.max()+1e-6,12)
    for genuine,label in [(True,'Genuine'),(False,'Impostor')]:
        axes[0].hist(trials[trials.genuine==genuine].score,bins=bins,alpha=.55,label=label)
    axes[0].axvline(threshold,color='black',ls='--',label=f'θ={threshold:.6f}')
    axes[0].set(title='DEV score distributions',xlabel='Cosine score',ylabel='Trials')
    axes[0].legend(fontsize=8)
    for metric in ['far','frr']:
        axes[1].step(curve.threshold,curve[metric]*100,where='mid',label=metric.upper())
    axes[1].axvline(threshold,color='black',ls='--')
    axes[1].set(title='Threshold selected on DEV',xlabel='Threshold θ',ylabel='Error rate (%)',ylim=(-3,103))
    axes[1].legend()
    v = result['summary']['verification']
    values = [v['far']*100,v['frr']*100]
    bars=axes[2].bar(['FAR','FRR'],values,color=['tab:red','tab:orange'])
    labels=[f'{values[0]:.1f}%\n{v["false_accepts"]}/{v["impostor_trials"]}',
            f'{values[1]:.1f}%\n{v["false_rejects"]}/{v["genuine_trials"]}']
    axes[2].bar_label(bars,labels=labels,padding=3,fontsize=9)
    axes[2].set(title='Independent TEST — frozen θ',ylabel='Error rate (%)',ylim=(0,110))
    for ax in axes[:2]:
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        ax.tick_params(axis='x',rotation=25,labelsize=8)
    return fig


def plot_conversation_demo(result):
    """Manual truth vs frozen-model predictions; no invented labels over overlap/gaps."""
    from matplotlib.patches import Patch
    audio,sr = result['audio'],result['sr']
    speakers = result['speakers']
    colors = dict(zip(speakers,['tab:blue','tab:orange','tab:green']))
    fig,axes = plt.subplots(4,1,figsize=(14,8),sharex=True,constrained_layout=True,
                            gridspec_kw={'height_ratios':[1,1,1,1.4]})
    axes[0].plot(np.arange(len(audio))/sr,audio,lw=.5,color='#4c5965')
    axes[0].set(title='AMI conversation — 23.28s',ylabel='Amplitude')
    for row in result['annotations']:
        if row['speaker'] in speakers:
            y = speakers.index(row['speaker'])
            axes[1].broken_barh([(row['start'],row['end']-row['start'])],(y-.32,.64),
                                 facecolors=colors[row['speaker']])
    for row in result['predictions'].itertuples():
        y = speakers.index(row.predicted)
        axes[2].broken_barh([(row.start,row.end-row.start)],(y-.32,.64),
             facecolors=colors[row.predicted],edgecolors='black' if row.correct else 'red',
             linewidth=1 if row.correct else 2.5)
        axes[2].text((row.start+row.end)/2,y,f'T{row.turn} '+('✓' if row.correct else '×'),
                     ha='center',va='center',fontsize=8,color='white')
        for j,speaker in enumerate(speakers):
            axes[3].plot([row.start,row.end],[result['scores'][row.Index,j]]*2,
                          color=colors[speaker],lw=1.6,label=speaker if row.Index==0 else None)
    evaluated = [(r.start,r.end) for r in result['predictions'].itertuples()]
    for row in result['intervals']:
        if row['speaker']=='overlap':
            color,alpha = '#ed8989',.22
        elif not any(abs(row['start']-a)<1e-8 and abs(row['end']-b)<1e-8 for a,b in evaluated):
            color,alpha = '#b4b4b4',.22
        else:
            continue
        for ax in axes:
            ax.axvspan(row['start'],row['end'],color=color,alpha=alpha,zorder=0)
    for ax,title in zip(axes[1:3],['Ground truth (manual annotations)',
                                  'Identification — red border = wrong; blank = not evaluated']):
        ax.set(yticks=range(len(speakers)),yticklabels=speakers,ylim=(len(speakers)-.5,-.5),title=title)
    axes[0].legend(handles=[Patch(facecolor='#ed8989',alpha=.5,label='Overlap: excluded'),
                            Patch(facecolor='#b4b4b4',alpha=.5,label='Silence / short turn: excluded')],
                   loc='upper right',fontsize=8,ncol=2)
    axes[3].set(title='Cosine per turn vs all 3 enrolled models (not probabilities)',
                ylabel='Cosine score',xlabel='Time in preview (s)')
    axes[3].legend(loc='lower left',ncol=3,fontsize=8)
    axes[3].ticklabel_format(axis='y',style='plain',useOffset=False)
    for ax in axes:
        ax.set_xlim(0,len(audio)/sr)
        ax.grid(axis='x',alpha=.2)
    return fig
