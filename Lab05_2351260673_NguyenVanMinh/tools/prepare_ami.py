"""Small AMI natural-conversation subset, manual labels, no speech synthesis.

Fetch only required PCM byte ranges from one WAV, not the full 89-minute meeting.
Usage: python3 tools/prepare_ami.py [--annotations /path/to/ami_manual.zip]
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET
import wave
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/speaker_recognition/ami'
MEETING = 'EN2009d'
BASE = 'https://groups.inf.ed.ac.uk/ami/'
ANNOTATIONS = BASE+'AMICorpusAnnotations/ami_public_manual_1.6.2.zip'
AUDIO = BASE+'AMICorpusMirror/amicorpus/HeadsetAudio/'+MEETING+'.Mix-Headset.wav'
NS = '{http://nite.sourceforge.net/}'
PREVIEW = (4941.96, 4965.24)


def download_range(start, end, url=AUDIO):
    request = Request(url, headers={'Range': f'bytes={start}-{end}'})
    with urlopen(request, timeout=90) as response:
        if response.status != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
            raise RuntimeError('Server did not honor Range; refusing to download full meeting.')
        data = response.read(end-start+2)
    if len(data) != end-start+1:
        raise ValueError('Incomplete audio range.')
    return data


def wav_format(header):
    if header[:4] != b'RIFF' or header[8:12] != b'WAVE':
        raise ValueError('Not a RIFF WAV file.')
    offset, fmt = 12, None
    while offset+8 <= len(header):
        kind, length = struct.unpack_from('<4sI', header, offset)
        if kind == b'fmt ':
            fmt = struct.unpack_from('<HHIIHH', header, offset+8)
        if kind == b'data':
            if fmt is None or (fmt[0],fmt[1],fmt[2],fmt[5]) != (1,1,16000,16):
                raise ValueError(f'Unexpected source format: {fmt}')
            return offset+8, fmt[2], fmt[4]
        offset += 8+length+(length%2)
    raise ValueError('WAV header exceeds probe size.')


def select_clips(segments, words, speaker):
    candidates = []
    others = [item for spk, items in segments.items() if spk != speaker for item in items]
    for segment in segments[speaker]:
        sw = [w for w in words[speaker] if w['start'] >= segment['start'] and w['end'] <= segment['end']]
        # At most one clip from each annotated turn; cuts at word boundaries.
        for i, first in enumerate(sw):
            included = [w for w in sw[i:] if w['end'] <= first['start']+3.9]
            if len(included) < 5:
                continue
            start, end = first['start'], included[-1]['end']
            if end-start < 2.0 or sum(w['end']-w['start'] for w in included) < .6*(end-start):
                continue
            if any(item['start'] < end+.1 and item['end'] > start-.1 for item in others):
                continue
            if start < PREVIEW[1] and end > PREVIEW[0]:
                continue
            candidates.append(dict(start=start, end=end, segment_id=segment['id'],
                                   transcript=' '.join(w['text'] for w in included)))
            break
    if len(candidates) < 8:
        raise ValueError(f'{speaker}: only {len(candidates)} suitable turns.')
    # Eight distinct turns spread over the recording; splits stay chronological.
    chosen = [candidates[round(i*(len(candidates)-1)/7)] for i in range(8)]
    return chosen, len(candidates)


def prepare(annotation_path=None):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = OUT/'metadata'
    meta.mkdir(exist_ok=True)
    if annotation_path:
        archive = Path(annotation_path).read_bytes()
    else:
        with urlopen(ANNOTATIONS, timeout=120) as response:
            archive = response.read()
    segments, words = {}, {}
    with ZipFile(io.BytesIO(archive)) as z:
        for speaker in 'ABCD':
            name = f'segments/{MEETING}.{speaker}.segments.xml'
            raw = z.read(name)
            (meta/Path(name).name).write_bytes(raw)
            segments[speaker] = [dict(id=e.attrib[NS+'id'], start=float(e.attrib['transcriber_start']),
                                       end=float(e.attrib['transcriber_end'])) for e in ET.fromstring(raw)]
            name = f'words/{MEETING}.{speaker}.words.xml'
            raw = z.read(name)
            (meta/Path(name).name).write_bytes(raw)
            words[speaker] = [dict(start=float(e.attrib['starttime']), end=float(e.attrib['endtime']),
                                  text=e.text or '') for e in ET.fromstring(raw)
                              if e.tag=='w' and e.attrib.get('punc')!='true'
                              and 'starttime' in e.attrib and 'endtime' in e.attrib
                              and float(e.attrib['endtime']) > float(e.attrib['starttime'])]
        root = ET.fromstring(z.read('corpusResources/meetings.xml'))
        meeting = next(e for e in root if e.attrib.get('observation')==MEETING)
        (meta/'meeting.xml').write_bytes(ET.tostring(meeting, encoding='utf-8'))
    identities = {e.attrib['nxt_agent']:e.attrib['global_name'] for e in meeting}
    plans = {}
    for speaker in 'ABC':
        plans[speaker], count = select_clips(segments, words, speaker)
        print(f'{speaker} ({identities[speaker]}): selected 8 of {count} suitable turns.', flush=True)
    header = download_range(0,4095)
    data_offset, rate, alignment = wav_format(header)
    (meta/'source_header.bin').write_bytes(header)
    records = []
    def extract(path, start, end):
        first, last = round(start*rate), round(end*rate)
        pcm = download_range(data_offset+first*alignment, data_offset+last*alignment-1)
        path.parent.mkdir(parents=True,exist_ok=True)
        with wave.open(str(path),'wb') as w:
            w.setparams((1,2,rate,0,'NONE','not compressed'))
            w.writeframes(pcm)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    for number, (speaker, clips) in enumerate(plans.items(),1):
        for index, clip in enumerate(clips):
            split = 'enroll' if index<4 else ('dev' if index<6 else 'test')
            split_number = index+1 if split=='enroll' else (index-3 if split=='dev' else index-5)
            path = OUT/split/f'spk{number:02}_{split}_{split_number:02}.wav'
            digest = extract(path,clip['start'],clip['end'])
            records.append(dict(path=str(path.relative_to(OUT)), speaker=f'spk{number:02}',
                                ami_agent=speaker, global_speaker=identities[speaker], split=split,
                                meeting=MEETING, source_url=AUDIO, license='CC-BY-4.0',
                                sha256=digest, **clip))
            print(path.relative_to(OUT),flush=True)
    path = OUT/'conversation_23s.wav'
    digest = extract(path,*PREVIEW)
    records.append(dict(path=path.name,split='preview',meeting=MEETING,source_url=AUDIO,
                        start=PREVIEW[0],end=PREVIEW[1],license='CC-BY-4.0',sha256=digest))
    (OUT/'manifest.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
    (meta/'source.json').write_text(json.dumps(dict(audio_url=AUDIO,annotations_url=ANNOTATIONS,
        annotations_sha256=hashlib.sha256(archive).hexdigest(),
        license_url=BASE+'corpus/license.shtml',meeting_type=meeting.attrib['type'],
        selection='8 distinct non-overlapping turns per speaker, 2–4s, no other annotated speaker ±100ms',
        splits='first 4 turns enrollment, next 2 dev, last 2 test; chronological, same session',
        processing='PCM byte-range crops only; no synthesis, concatenation, denoising, or normalization'),indent=2)+'\n')
    with urlopen(BASE+'corpus/license.shtml',timeout=60) as r:
        (meta/'license.html').write_bytes(r.read())
    print('Prepared 24 speaker clips and a 23.28s conversation preview.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',type=Path)
    prepare(parser.parse_args().annotations)
