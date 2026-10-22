"""Validate metadata, every image, and the unchanged official SSB loader."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
from PIL import Image
from data.get_datasets import get_class_splits, get_datasets
from config import cub_root, osr_split_dir


def verify(check_images=True):
    root = Path(cub_root) / 'CUB_200_2011'
    for name in ('classes.txt', 'images.txt', 'image_class_labels.txt', 'train_test_split.txt',
                 'attributes/image_attribute_labels.txt'):
        assert (root / name).is_file(), root / name
    assert len((root / 'classes.txt').read_text().splitlines()) == 200
    records = (root / 'images.txt').read_text().splitlines()
    assert len(records) == 11788
    assert {int(r.split()[0]) for r in records} == set(range(1, 11789))
    if check_images:
        for record in records:
            path = root / 'images' / record.split(maxsplit=1)[1]
            with Image.open(path) as im:
                im.verify()
    args = get_class_splits(argparse.Namespace(dataset_name='cub', use_ssb_splits=True,
                                              prop_train_labels=0.5))
    old, new = set(args.train_classes), set(args.unlabeled_classes)
    assert len(old) == len(new) == 100
    assert not old & new and old | new == set(range(200))
    train, test, unlabelled, datasets = get_datasets('cub', None, None, args)
    labelled = datasets['train_labelled']
    assert not set(labelled.uq_idxs) & set(unlabelled.uq_idxs)
    assert len(train) == 5994 and len(test) == 5794
    assert set(train.labelled_dataset.data.target) | set(unlabelled.data.target) == set(range(1, 201))
    split = Path(osr_split_dir) / 'cub_osr_splits.pkl'
    report = dict(root=str(root), classes=200, images=len(records),
                  images_verified=check_images, labelled=len(labelled),
                  unlabelled=len(unlabelled), test=len(test), ssb='PASS',
                  split_sha256=hashlib.sha256(split.read_bytes()).hexdigest())
    print(json.dumps(report, indent=2))
    return args, train, test, datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-only', action='store_true')
    verify(not parser.parse_args().metadata_only)
