"""Read the existing, trusted CUB split without importing dataset/image code."""
import hashlib
from pathlib import Path
import pickle

from config import osr_split_dir

CUB_SPLIT_SHA256 = '48bdcaa05603d932037738d3df776ab500945e9b7aeff85f4668305bf71bd1e7'


def load_cub_ssb_classes():
    # Restrict pickle loading to the pinned repository asset, never user metadata.
    content = (Path(osr_split_dir) / 'cub_osr_splits.pkl').read_bytes()
    if hashlib.sha256(content).hexdigest() != CUB_SPLIT_SHA256:
        raise ValueError('CUB SSB split checksum differs from the reproduced baseline')
    split = pickle.loads(content)
    unknown = split['unknown_classes']
    # Exactly data/get_datasets.py's CUB ordering, with no new class partition.
    return tuple(split['known_classes']), tuple(unknown['Hard'] + unknown['Medium'] + unknown['Easy'])
