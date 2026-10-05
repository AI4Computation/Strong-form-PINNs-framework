"""Bookkeeping for the separate plain-PINN cost cohort; no solver imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT.parent / 'controlled_pinn'
PROJECT = ROOT.parents[1]
METHODS = ['vanilla_matched', 'anchored', 'fourier_half']

def utc():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_hashes(mapping):
    for path, expected in mapping.items():
        assert sha(path) == expected, f'Frozen file changed: {path}'

def consent():
    path = HERE / 'idle_confirmation.json'
    assert path.exists(), 'Fresh author confirmation of an idle computer is required.'
    data = read(path)
    assert data['confirmed_idle'] is True and data['author_message'].strip()
    assert data['protocol_sha256'] == sha(HERE / 'protocol.json')
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(data['received_utc'])).total_seconds()
    assert 0 <= age <= 6 * 3600, 'Idle window confirmation is older than six hours.'
    return data
