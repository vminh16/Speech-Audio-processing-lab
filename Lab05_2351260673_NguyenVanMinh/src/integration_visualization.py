"""Before/after speaker-ID charts using fixed models and identical time intervals."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


METHODS=['Raw','SS','Wiener']
COLORS=['gray','tab:blue','tab:orange']


def plot_integration_summary(result):
    data=result['summary']
    noises=[n for n in data.noise.unique() if n!='original']
    fig,axes=plt.subplots(2,len(noises),figsize=(12,6),sharey=True,squeeze=False,constrained_layout=True)
    for row,scope in enumerate(['test','conversation']):
        original=data[(data.scope==scope)&(data.noise=='original')&(data.method=='Raw')].accuracy.iloc[0]
        for col,noise in enumerate(noises):
            ax=axes[row,col]
            table=data[(data.scope==scope)&(data.noise==noise)].pivot(index='target_db',columns='method',values='accuracy')
            x=np.arange(len(table))
            for i,(method,color) in enumerate(zip(METHODS,COLORS)):
                bars=ax.bar(x+(i-1)*.25,table[method]*100,width=.24,label=method,color=color)
                ax.bar_label(bars,fmt='%.0f',fontsize=7,padding=2)
            ax.axhline(original*100,color='black',ls='--',lw=.8,label='Original raw')
            ax.set(xticks=x,xticklabels=[f'{v:g}' for v in table.index],xlabel='Added-noise input SNR (dB)',
                   ylabel='Speaker-ID accuracy (%)',ylim=(0,115),title=f'{scope}: {noise}')
            ax.legend(ncol=4,fontsize=7,loc='upper right')
    return fig


def plot_integration_snr(result):
    data=result['summary']
    noises=[n for n in data.noise.unique() if n!='original']
    fig,axes=plt.subplots(1,len(noises),figsize=(12,3.4),sharey=True,squeeze=False,constrained_layout=True)
    for ax,noise in zip(axes[0],noises):
        for method,color,marker in zip(METHODS,COLORS,['o','s','^']):
            frame=data[(data.scope=='conversation')&(data.noise==noise)&(data.method==method)].sort_values('target_db')
            ax.plot(frame.target_db,frame.snr_out,marker+'-',color=color,label=method)
        ax.set(xlabel='Added-noise SNR (dB)',ylabel='Output SNR vs original AMI (dB)',title=noise)
        ax.legend()
        ax.grid(alpha=.2)
    return fig


def plot_integrated_conversation(result):
    predictions=result['predictions']
    predictions=predictions[(predictions.scope=='conversation')&(predictions.noise=='original')]
    speakers=sorted(result['conversation']['speakers'])
    colors=dict(zip(speakers,['tab:blue','tab:orange','tab:green']))
    fig,axes=plt.subplots(1,2,figsize=(14,3.5),constrained_layout=True,gridspec_kw={'width_ratios':[3,1]})
    truth=predictions[predictions.method=='Raw']
    for row in truth.itertuples():
        axes[0].broken_barh([(row.start,row.end-row.start)],(-.3,.6),facecolors=colors[row.true])
    accuracies=[]
    for i,method in enumerate(METHODS,1):
        frame=predictions[predictions.method==method]
        accuracies.append(frame.correct.mean()*100)
        for row in frame.itertuples():
            axes[0].broken_barh([(row.start,row.end-row.start)],(i-.3,.6),facecolors=colors[row.predicted],
                                edgecolors='black' if row.correct else 'red',linewidth=1 if row.correct else 2)
    for interval in result['conversation']['intervals']:
        if interval['speaker']=='overlap':axes[0].axvspan(interval['start'],interval['end'],color='red',alpha=.12)
    axes[0].set(yticks=range(4),yticklabels=['GT (scored turns)',*METHODS],ylim=(3.5,-.5),
                xlim=(0,len(result['conversation']['audio'])/result['sr']),xlabel='Time (s)',
                title='Original 23s → whole-file denoise → same speaker turns')
    axes[0].legend(handles=[Patch(facecolor=c,label=s) for s,c in colors.items()]+[
        Patch(facecolor='white',edgecolor='red',label='Wrong / overlap excluded')],ncol=4,fontsize=8,loc='upper right')
    bars=axes[1].bar(METHODS,accuracies,color=COLORS)
    axes[1].bar_label(bars,fmt='%.0f%%',padding=3)
    axes[1].set(ylim=(0,115),ylabel='Accuracy (%)',title='5 held-out solo turns')
    return fig
