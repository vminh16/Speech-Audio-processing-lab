"""ICA plots: distinguish true synthetic sources from approximate real references."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


def listening_audio(x, rms=.08):
    """Level for playback only; ICA does not identify absolute output scale."""
    x=np.asarray(x,dtype=float)
    if x.ndim==1:x=x[:,None]
    level=np.sqrt(np.mean(x*x,axis=0,keepdims=True))
    scaled=x/np.maximum(level,1e-12)*rms
    scaled*=min(1.,.95/np.max(np.abs(scaled)))
    return scaled.T


def plot_ica_pipeline():
    fig,ax=plt.subplots(figsize=(13,1.6),constrained_layout=True)
    ax.axis('off')
    labels=['2 synchronized mics\nX: samples × 2','Center / whiten\ninside FastICA',
            'Independent components\nW estimated from X only','2 outputs\nunknown order / sign',
            'Evaluate afterward\nmatch references; no refit']
    xs=np.linspace(.08,.92,len(labels))
    for i,(x,label) in enumerate(zip(xs,labels)):
        ax.text(x,.5,label,ha='center',va='center',fontsize=9,transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=.5',facecolor='#e9f2fa',edgecolor='#5781a3'))
        if i:ax.annotate('',xy=(x-.075,.5),xytext=(xs[i-1]+.075,.5),xycoords='axes fraction',
                         arrowprops=dict(arrowstyle='->',color='#5781a3'))
    return fig


def plot_separation_spectrograms(result, time_range=None):
    sr=result['sr']
    begin,end=(0,len(result['mixtures'])/sr) if time_range is None else time_range
    first,last=round(begin*sr),round(end*sr)
    kinds=[('Headset reference (approx.)' if result['kind']=='real' else 'True source',result['references']),
           ('Real array mic' if result['kind']=='real' else 'Simulated mic',result['mixtures']),
           ('ICA output (matched order/sign)',result['estimates'])]
    fig,axes=plt.subplots(2,3,figsize=(14,5.3),sharex=True,sharey=True,constrained_layout=True)
    spectra=[]
    for _,pair in kinds:
        normalized=listening_audio(pair).T
        for channel in range(2):
            f,t,z=stft(normalized[first:last,channel],fs=sr,window='hann',nperseg=400,noverlap=240,nfft=512)
            spectra.append((f,t+begin,np.abs(z)**2))
    maximum=max(p.max() for _,_,p in spectra)
    for col,(title,_) in enumerate(kinds):
        for row in range(2):
            f,t,p=spectra[2*col+row]
            mesh=axes[row,col].pcolormesh(t,f,10*np.log10(np.maximum(p/maximum,1e-10)),
                                         shading='auto',cmap='magma',vmin=-70,vmax=0)
            axes[row,col].set(title=f'{title} {row+1}',xlabel='Time (s)',ylabel='Hz',ylim=(0,8000),xlim=(begin,end))
            for interval in result.get('intervals',[]):
                if interval['speaker']=='overlap' and interval['start']<end and interval['end']>begin:
                    axes[row,col].axvline(interval['start'],color='cyan',ls='--',lw=.7)
                    axes[row,col].axvline(interval['end'],color='cyan',ls='--',lw=.7)
    fig.colorbar(mesh,ax=list(axes.ravel()),label='Power dB, shared reference (RMS-normalized for display)')
    return fig


def plot_correlation_matching(result):
    real=result['kind']=='real'
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),constrained_layout=True)
    for ax,title,matrix in zip(axes,['Before permutation matching','After matching (sign corrected in audio)'],
                              [result['before'],result['after']]):
        mesh=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis')
        ax.set(title=title,xticks=[0,1],xticklabels=['Output 1','Output 2'],
               yticks=[0,1],yticklabels=['Headset A (approx.)','Headset B (approx.)'] if real else ['Source A','Source B'])
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f'{matrix[i,j]:.4f}',ha='center',va='center',color='white',
                        bbox=dict(facecolor='black',alpha=.25,edgecolor='none'))
    fig.colorbar(mesh,ax=list(axes),label='Absolute zero-lag Pearson correlation')
    return fig


def plot_controlled_summary(metrics):
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),constrained_layout=True)
    labels=metrics.case.str.replace('_','\n')
    bars=axes[0].bar(labels,metrics.min_abs_correlation,color=['tab:blue','tab:orange','tab:blue','tab:orange'])
    axes[0].bar_label(bars,fmt='%.4f',padding=3,fontsize=9)
    axes[0].set(ylabel='Minimum matched |correlation|',ylim=(0,1.15),title='Controlled recovery: clean vs noise 20 dB')
    bars=axes[1].bar(labels,metrics.condition,color='tab:purple')
    axes[1].bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    axes[1].set(ylabel='Condition number of A',title='Near-singular mixing amplifies noise',ylim=(0,115))
    return fig
