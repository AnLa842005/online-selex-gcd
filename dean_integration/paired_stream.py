"""Explicit paired online protocol: independent stream RNG and immutable ID batches."""
import json
from pathlib import Path
import random

import numpy as np
import torch

from .preflight import sha256


def validate_online_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("online_seed must be an integer in [0, 2**32)")


def reset_online_rng(seed):
    validate_online_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_ids(dataset):
    values = np.asarray(dataset.uq_idxs)
    if values.ndim != 1 or len(values) != len(dataset) or values.dtype.kind not in "iu":
        raise ValueError("Paired stream dataset requires aligned integer uq_idxs")
    ids = [int(v) for v in values]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Paired stream dataset IDs must be nonempty and unique")
    return ids


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def validate_manifest(record, context, datasets=None, batch_size=None):
    """Validate context, exact coverage, order and batch boundaries before use."""
    def fail(reason):
        raise ValueError("Invalid/incompatible stream manifest: " + reason)

    if not isinstance(record, dict) or set(record) != {"version", "context", "batch_size", "sessions"}:
        fail("schema")
    if type(record["version"]) is not int or record["version"] != 1:
        fail("version")
    if canonical_bytes(record["context"]) != canonical_bytes(context):
        fail("online seed, checkpoint, protocol or dataset metadata mismatch")
    validate_online_seed(context["online_seed"])
    size = record["batch_size"]
    if type(size) is not int or size < 1 or (batch_size is not None and size != batch_size):
        fail("batch size")
    sessions = record["sessions"]
    if not isinstance(sessions, list) or not sessions or (datasets is not None and len(sessions) != len(datasets)):
        fail("session count")
    for i, session in enumerate(sessions):
        if not isinstance(session, dict) or set(session) != {"session_id", "dataset_sample_ids", "batches"}:
            fail("session schema")
        if type(session["session_id"]) is not int or session["session_id"] != i:
            fail("session ID")
        ids, batches = session["dataset_sample_ids"], session["batches"]
        if not isinstance(ids, list) or not ids or any(type(v) is not int for v in ids) or len(set(ids)) != len(ids):
            fail("dataset sample IDs")
        if datasets is not None and ids != sample_ids(datasets[i]):
            fail("dataset sample IDs/order mismatch")
        if not isinstance(batches, list) or not batches:
            fail("empty/malformed batches")
        flat = []
        for j, batch in enumerate(batches):
            if not isinstance(batch, list) or any(type(v) is not int for v in batch):
                fail("batch sample IDs")
            if not 1 <= len(batch) <= size or (j < len(batches) - 1 and len(batch) != size):
                fail("batch boundaries")
            flat.extend(batch)
        if len(flat) != len(ids) or len(set(flat)) != len(flat) or set(flat) != set(ids):
            fail("missing, duplicate or unknown samples")
    return record


def make_manifest(datasets, batch_size, context):
    validate_online_seed(context["online_seed"])
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Paired batch size must be a positive integer")
    sessions = []
    for index, dataset in enumerate(datasets):
        ids = sample_ids(dataset)
        # A private generator never reads or advances any global RNG.
        generator = torch.Generator().manual_seed(context["online_seed"] + index)
        order = [ids[j] for j in torch.randperm(len(ids), generator=generator).tolist()]
        sessions.append(dict(session_id=index, dataset_sample_ids=ids,
                             batches=[order[j:j + batch_size] for j in range(0, len(ids), batch_size)]))
    return validate_manifest(dict(version=1, context=context, batch_size=batch_size, sessions=sessions),
                             context, datasets, batch_size)


def read_manifest(path, context, datasets=None, batch_size=None):
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read stream manifest: {exc}") from exc
    return validate_manifest(record, context, datasets, batch_size)


class PairedOnlineStream:
    def __init__(self, context, output, manifest_path=None):
        self.context = context
        self.output = Path(output)
        self.manifest_path = manifest_path
        self.record = None

    def prepare(self, datasets, batch_size):
        if self.record is not None:
            raise ValueError("Paired online boundary may only be entered once")
        record = (read_manifest(self.manifest_path, self.context, datasets, batch_size)
                  if self.manifest_path else make_manifest(datasets, batch_size, self.context))
        # Exclusive creation protects input manifests and existing run artifacts.
        with self.output.open("xb") as handle:
            handle.write(canonical_bytes(record))
        self.record = record
        reset_online_rng(self.context["online_seed"])

    def loader(self, dataset, session_id, batch_size, num_workers):
        if self.record is None:
            raise ValueError("Paired stream not prepared at online boundary")
        if type(session_id) is not int or not 0 <= session_id < len(self.record["sessions"]):
            raise ValueError("Invalid paired stream session ID")
        session = self.record["sessions"][session_id]
        ids = sample_ids(dataset)
        if ids != session["dataset_sample_ids"] or batch_size != self.record["batch_size"]:
            raise ValueError("Paired loader dataset/batch size differs from manifest")
        positions = {value: index for index, value in enumerate(ids)}
        batches = [[positions[value] for value in batch] for batch in session["batches"]]
        # Separate generator controls worker seeds; iterator creation uses no global RNG.
        worker_rng = torch.Generator().manual_seed(self.context["online_seed"] + session_id + 2**32)
        return torch.utils.data.DataLoader(dataset, batch_sampler=batches, num_workers=num_workers,
                                           generator=worker_rng)


def manifest_summary(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(path=str(Path(path).resolve()), sha256=sha256(path),
                online_seed=record["context"]["online_seed"],
                batches=sum(len(s["batches"]) for s in record["sessions"]),
                samples=sum(len(s["dataset_sample_ids"]) for s in record["sessions"]))
