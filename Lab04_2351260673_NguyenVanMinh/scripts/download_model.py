"""One-time setup; synthesis itself never uses the network."""
import hashlib
import json
import urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'data/models/piper'
m=json.loads((root/'manifest.json').read_text())
p=root/'vi_VN-vais1000-medium.onnx'
if not p.exists():
    url=f"https://huggingface.co/{m['repository']}/resolve/{m['revision']}/{m['path']}/{p.name}"
    urllib.request.urlretrieve(url,p)
if hashlib.sha256(p.read_bytes()).hexdigest()!=m['sha256']:
    raise RuntimeError('Model checksum mismatch')
print('Model verified:',p)
