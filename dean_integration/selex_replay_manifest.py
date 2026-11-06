"""Reconstruct ONLY the pinned original CUB train_labelled eligibility split."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import ROOT, UPSTREAM_SHA
from .selex_hierarchy_provider import canonical, digest
from .selex_labeled_replay import MANIFEST_SCHEMA, VerifiedEligibilityManifest

FILES = ("images.txt", "image_class_labels.txt", "train_test_split.txt")


def reconstruct(dataset_root, prop_samples=.8, base_classes=160):
    if type(base_classes) is not int or base_classes < 1 or not 0 < prop_samples <= 1:
        raise ValueError("Invalid labelled split parameters")
    root = Path(dataset_root)
    raw = {name: (root/name).read_bytes() for name in FILES}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    parsed = {name: [line.split(maxsplit=1) for line in data.decode("utf-8").splitlines() if line.strip()]
              for name, data in raw.items()}
    for name, rows in parsed.items():
        if any(len(r) != 2 for r in rows) or len({int(r[0]) for r in rows}) != len(rows):
            raise ValueError("Malformed/duplicate CUB metadata: "+name)
    labels = {int(i): int(y)-1 for i, y in parsed["image_class_labels.txt"]}
    splits = {int(i): int(y) for i, y in parsed["train_test_split.txt"]}
    images = parsed["images.txt"]  # pandas merge preserves images-file left order.
    if set(labels) != set(splits) or set(labels) != {int(i) for i, _ in images} or any(v not in (0, 1) for v in splits.values()):
        raise ValueError("CUB metadata joins/split mismatch")
    training = [(int(i), path) for i, path in images if splits[int(i)] == 1]
    records = []
    for cls in range(base_classes):
        ids = np.array([uq for uq, (image_id, _) in enumerate(training) if labels[image_id] == cls], dtype=np.int64)
        # Upstream subsample_instances resets NumPy seed 0 INSIDE each class call.
        selected = np.random.RandomState(0).choice(ids, replace=False, size=int(prop_samples*len(ids)))
        for uq in sorted(selected.tolist()):
            image_id, path = training[uq]
            records.append(dict(sample_id=uq, canonical_path=path, true_label=labels[image_id]))
    records.sort(key=lambda r: (r["true_label"], r["sample_id"]))
    split = dict(base_classes=base_classes, prop_samples=prop_samples, internal_numpy_seed=0,
                 sampling="per-class-floor-original-train-uq-index")
    dataset = digest(hashes)
    core = dict(schema=MANIFEST_SCHEMA, dataset_fingerprint=dataset, split_fingerprint=digest(dict(dataset=dataset, **split)),
                source_split="train_labelled", label_origin="ground_truth", records=records)
    lock = json.loads((ROOT/"configs/dean/upstream_lock.json").read_text())
    source = dict(upstream_sha=UPSTREAM_SHA, source_tree_hash=digest(lock["files"]), metadata_hashes=hashes, **split)
    return dict(schema="dean-cub-replay-source-v1", source=source, eligibility=core)


def verified_manifest(dataset_root, path=None, expected_hash=None, prop_samples=.8, base_classes=160):
    expected = reconstruct(dataset_root, prop_samples, base_classes)
    if path is not None:
        if expected_hash is None:
            raise ValueError("An existing replay manifest requires an explicit trusted canonical SHA-256")
        actual = json.loads(Path(path).read_text(encoding="utf-8"))
        if digest(actual) != expected_hash or actual != expected:
            raise ValueError("Replay manifest hash/source/split/dataset mismatch")
    core = expected["eligibility"]
    verified = VerifiedEligibilityManifest(core, digest(core), core["dataset_fingerprint"], core["split_fingerprint"])
    return expected, verified


def write_manifest(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical(value)+"\n")
