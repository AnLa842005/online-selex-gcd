"""Fetch official assets only, outside Git. Requires curl and 15 GiB free."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile


def fetch(url, path):
    if path.exists():
        return
    part = path.with_suffix(path.suffix + '.part')
    subprocess.run(['curl', '-fL', '--retry', '3', '-C', '-', '-o', str(part), url], check=True)
    part.rename(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pretrained-root', type=Path, required=True)
    args = parser.parse_args()
    for root in (args.data_root, args.pretrained_root):
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < 15 * 2**30:
            raise RuntimeError('At least 15 GiB free is required before downloads')
    archive = args.data_root / 'CUB_200_2011.tgz'
    if not (args.data_root / 'CUB_200_2011').exists():
        fetch('https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz', archive)
        digest = hashlib.md5()
        with archive.open('rb') as stream:
            for chunk in iter(lambda: stream.read(2**20), b''):
                digest.update(chunk)
        if digest.hexdigest() != '97eceeb196236b17998738112f37df78':
            raise RuntimeError('CUB checksum mismatch; extraction refused')
        with tarfile.open(archive) as tar:
            tar.extractall(args.data_root, filter='data')
    fetch('https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth',
          args.pretrained_root / 'dino_vitbase16_pretrain.pth')
    digest = hashlib.sha256()
    with (args.pretrained_root / 'dino_vitbase16_pretrain.pth').open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    if digest.hexdigest() != 'bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e':
        raise RuntimeError('DINO backbone checksum mismatch')


if __name__ == '__main__':
    main()
