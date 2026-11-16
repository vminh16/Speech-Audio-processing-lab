"""Fetch 23.28s from four real AMI microphones, plus two solo crops for ICA lab."""
import hashlib
import json
from pathlib import Path
import sys
import wave
from xml.etree import ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_ami import download_range, wav_format, BASE
from src.speaker_experiments import conversation_intervals

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/speaker_recognition/ami'
OUT=ROOT/'data/separation/ami23s'


def write_pcm(path, pcm, sr=16000):
    path.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(path),'wb') as w:
        w.setparams((1,2,sr,0,'NONE','not compressed'))
        w.writeframes(pcm)


def main():
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    preview=next(r for r in manifest if r['split']=='preview')
    meeting=preview['meeting']
    records=[]
    for channel,role in [('Array1-01','microphone'),('Array1-05','microphone'),
                         ('Headset-0','approximate_reference'),('Headset-1','approximate_reference')]:
        url=BASE+f'AMICorpusMirror/amicorpus/{meeting}/audio/{meeting}.{channel}.wav'
        path=OUT/f'{channel}.wav'
        if not path.exists():
            header=download_range(0,4095,url=url)
            offset,sr,alignment=wav_format(header)
            pcm=download_range(offset+round(preview['start']*sr)*alignment,
                               offset+round(preview['end']*sr)*alignment-1,url=url)
            write_pcm(path,pcm,sr)
        with wave.open(str(path)) as w:
            assert (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getnframes())==(16000,1,2,372480)
        records.append(dict(path=path.name,role=role,source_url=url,meeting=meeting,
                            start=preview['start'],end=preview['end'],license='CC-BY-4.0',
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(path.name,flush=True)
    labels={r['ami_agent']:r['speaker'] for r in manifest if 'ami_agent' in r}
    annotations=[]
    for path in (SOURCE/'metadata').glob(f'{meeting}.*.segments.xml'):
        agent=path.name.split('.')[1]
        for e in ET.parse(path).getroot():
            start=max(0.,float(e.attrib['transcriber_start'])-preview['start'])
            end=min(preview['end']-preview['start'],float(e.attrib['transcriber_end'])-preview['start'])
            if end>start:annotations.append(dict(start=start,end=end,speaker=labels.get(agent,agent)))
    intervals=conversation_intervals(annotations,preview['end']-preview['start'])
    (OUT/'intervals.json').write_text(json.dumps(intervals,indent=2)+'\n')
    with wave.open(str(SOURCE/preview['path'])) as w:
        sr=w.getframerate(); pcm=w.readframes(w.getnframes())
    for name,speaker in [('source_A','spk01'),('source_B','spk02')]:
        turn=next(r for r in intervals if r['speaker']==speaker and r['end']-r['start']>=3.)
        start=round(turn['start']*sr);length=3*sr
        path=OUT/f'{name}.wav'
        write_pcm(path,pcm[2*start:2*(start+length)],sr)
        records.append(dict(path=path.name,role='controlled_source',speaker=speaker,
            source_url=preview['source_url'],meeting=meeting,
            preview_start=start/sr,preview_end=(start+length)/sr,
            start=preview['start']+start/sr,end=preview['start']+(start+length)/sr,
            license='CC-BY-4.0',sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (OUT/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    print('Ready: two real array mics, two approximate headset references, two solo 3s sources.',flush=True)

if __name__=='__main__':main()
