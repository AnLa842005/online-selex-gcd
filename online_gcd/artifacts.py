"""Canonical, checksummed JSON and file provenance for generated artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess

from online_gcd.state import _unique_object, _reject_constant


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode('utf-8')).hexdigest()


def seal(value):
    return dict(value, checksum=digest(value))


def verify(value):
    if not isinstance(value, dict) or value.get('checksum') != digest({k: v for k, v in value.items() if k != 'checksum'}):
        raise ValueError('artifact checksum mismatch')
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique_object,
                      parse_constant=_reject_constant)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def code_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=Path(__file__).resolve().parents[1], text=True).strip()
