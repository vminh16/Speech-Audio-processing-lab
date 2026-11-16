"""Prepare only Part 1 enhancement data: FSDD reference/dev + ESC-50 noise."""
import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
from urllib.request import urlopen
import wave
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FSDD = 'https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/v1.0.10/'
ESC = 'https://raw.githubusercontent.com/karolpiczak/ESC-50/master/'


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(url, timeout=60) as response:
            content = response.read()
        path.write_bytes(content)
    return path


def convert(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', str(source),
                    '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(target)], check=True)


def main():
    records = []
    speakers = ['jackson', 'nicolas', 'theo']
    # Six dev clips and ONE reference/test clip are used by enhancement.
    plans = [(speaker_id, speaker, 'dev', number, index)
             for speaker_id, speaker in enumerate(speakers, 1)
             for number, index in enumerate([9, 10], 1)]
    plans.append((1, 'jackson', 'test', 1, 0))
    jobs = [(FSDD + f'recordings/{digit}_{speaker}_{index}.wav',
             DATA / 'raw/fsdd' / f'{digit}_{speaker}_{index}.wav')
            for _, speaker, _, _, index in plans for digit in range(10)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda job: fetch(*job), jobs))
    fetch(FSDD + 'README.md', DATA / 'raw/fsdd/README.upstream.md')
    fetch(ESC + 'README.md', DATA / 'raw/esc50/README.upstream.md')
    fetch(ESC + 'LICENSE', DATA / 'raw/esc50/LICENSE')
    metadata = fetch(ESC + 'meta/esc50.csv', DATA / 'raw/esc50/esc50.csv')
    rows = list(csv.DictReader(io.StringIO(metadata.read_text())))
    for speaker_id, speaker, split, number, index in plans:
        files = [DATA / 'raw/fsdd' / f'{d}_{speaker}_{index}.wav' for d in range(10)]
        frames = []
        for file in files:
            with wave.open(str(file)) as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (8000, 1, 2):
                    raise ValueError(f'Unexpected WAV format: {file}')
                frames.append(wav.readframes(wav.getnframes()))
        samples = b''.join(frames)[:4 * 8000 * 2]
        samples += b'\0' * max(0, 2 * 8000 * 2 - len(samples))
        target = DATA / 'clean' / f'spk{speaker_id:02}_{split}_{number:02}.wav'
        with TemporaryDirectory() as folder:
            temporary = Path(folder)/'concatenated.wav'
            with wave.open(str(temporary), 'wb') as wav:
                wav.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
                wav.writeframes(samples)
            convert(temporary, target)
        records.append({'path': str(target.relative_to(DATA)), 'speaker': speaker,
                        'split': split, 'purpose': 'enhancement',
                        'source_files': [p.name for p in files]})
    for category in ['rain', 'vacuum_cleaner']:
        row = next(r for r in rows if r['category'] == category)
        raw = fetch(ESC + 'audio/' + row['filename'], DATA / 'raw/esc50' / row['filename'])
        target = DATA / 'noise' / f'{category}.wav'
        convert(raw, target)
        records.append({'path': str(target.relative_to(DATA)), 'split': 'noise', 'source': row})
    (DATA / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    print(f'Prepared {len(records)} audio files. Manifest: data/manifest.json')

if __name__ == '__main__':
    main()
