"""Verified GT references only; no images, features, pseudo labels or VFA replay.

The independently trusted eligibility digest is a preflight input. This module
checks membership against that manifest; it cannot establish historical split
provenance by trusting a caller's label tensor or a self-asserted boolean.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath

from .selex_hierarchy_provider import canonical, digest

REPLAY_SCHEMA = "verified-labelled-replay-v1"
MANIFEST_SCHEMA = "base-train-labelled-eligibility-v1"
INELIGIBLE = "replay.class_ineligible.insufficient_distinct_samples"
CLASS_SELECTED = "replay.class_selected"
REFERENCE_SELECTED = "replay.reference_selected"
SELECTION_HASH = "replay.selection_hash"


def fingerprint(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class VerifiedReplayEntry:
    sample_id: int
    canonical_path: str
    true_label: int
    dataset_fingerprint: str
    split_fingerprint: str
    eligibility_manifest: str
    admission_batch: int
    admission_order: int
    label_origin: str = "ground_truth"
    source_split: str = "train_labelled"

    def __post_init__(self):
        for name in ("sample_id", "true_label", "admission_batch", "admission_order"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("Replay IDs/labels/admission metadata require nonnegative integers")
        path = self.canonical_path
        if (not isinstance(path, str) or not path or "\\" in path or ":" in path
                or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
                or str(PurePosixPath(path)) != path):
            raise ValueError("Replay requires a canonical dataset-relative image path")
        if not all(fingerprint(x) for x in (self.dataset_fingerprint, self.split_fingerprint, self.eligibility_manifest)):
            raise ValueError("Replay requires dataset/split/manifest SHA-256 fingerprints")
        if self.label_origin != "ground_truth" or self.source_split != "train_labelled":
            raise ValueError("Only verified train_labelled ground truth may enter replay")


class VerifiedEligibilityManifest:
    """Trusted digest + dataset/split identity, independently provided by preflight.

    Records are exact sample-ID/path/GT triples from the original labelled split.
    No dataset reads or label discovery take place here.
    """
    def __init__(self, payload, expected_hash, dataset_fingerprint, split_fingerprint):
        required = {"schema", "dataset_fingerprint", "split_fingerprint", "source_split", "label_origin", "records"}
        if not isinstance(payload, dict) or set(payload) != required or not fingerprint(expected_hash):
            raise ValueError("Invalid verified eligibility manifest schema/hash")
        if (payload["schema"] != MANIFEST_SCHEMA or payload["source_split"] != "train_labelled"
                or payload["label_origin"] != "ground_truth"):
            raise ValueError("Manifest must identify original train_labelled ground truth")
        if (payload["dataset_fingerprint"] != dataset_fingerprint or payload["split_fingerprint"] != split_fingerprint
                or not fingerprint(dataset_fingerprint) or not fingerprint(split_fingerprint)):
            raise ValueError("Eligibility dataset/split fingerprint mismatch")
        if not isinstance(payload["records"], list):
            raise ValueError("Eligibility records must be a canonical list")
        records = {}
        paths = set()
        for row in payload["records"]:
            if not isinstance(row, dict) or set(row) != {"sample_id", "canonical_path", "true_label"}:
                raise ValueError("No pseudo labels, gt_y, VFA or feature fields are accepted")
            entry = VerifiedReplayEntry(**row, dataset_fingerprint=dataset_fingerprint,
                                        split_fingerprint=split_fingerprint, eligibility_manifest=expected_hash,
                                        admission_batch=0, admission_order=0)
            if entry.sample_id in records or entry.canonical_path in paths:
                raise ValueError("Duplicate real sample in eligibility manifest")
            records[entry.sample_id] = canonical(row)
            paths.add(entry.canonical_path)
        ordered = sorted(payload["records"], key=lambda r: (r["true_label"], r["sample_id"]))
        if payload["records"] != ordered or digest(payload) != expected_hash:
            raise ValueError("Eligibility manifest canonical order/hash mismatch")
        self._records = records
        self.identity = expected_hash
        self.dataset_fingerprint, self.split_fingerprint = dataset_fingerprint, split_fingerprint

    @classmethod
    def from_file(cls, path, expected_hash, dataset_fingerprint, split_fingerprint):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), expected_hash,
                   dataset_fingerprint, split_fingerprint)

    def entry(self, sample_id, admission_batch=0, admission_order=0):
        if type(sample_id) is not int or sample_id not in self._records:
            raise ValueError("Sample is outside verified train_labelled eligibility")
        return VerifiedReplayEntry(**json.loads(self._records[sample_id]), dataset_fingerprint=self.dataset_fingerprint,
                                   split_fingerprint=self.split_fingerprint, eligibility_manifest=self.identity,
                                   admission_batch=admission_batch, admission_order=admission_order)

    def verify(self, entry):
        if not isinstance(entry, VerifiedReplayEntry):
            raise ValueError("Replay accepts verified reference entries, never tensors/diagnostic labels")
        expected = self.entry(entry.sample_id, entry.admission_batch, entry.admission_order)
        if entry != expected:
            raise ValueError("Replay reference does not match independently verified manifest")


@dataclass(frozen=True)
class ReplaySelection:
    batch_id: int
    class_ids: tuple
    sample_ids: tuple
    pre_cursor: int
    post_cursor: int
    selection_hash: str


class VerifiedLabeledReplay:
    capacity = 320
    per_class = 2
    selection_size = 32

    def __init__(self, eligibility):
        if not isinstance(eligibility, VerifiedEligibilityManifest):
            raise ValueError("A verified eligibility manifest is required")
        self.eligibility = eligibility
        self._entries = {}
        self._class_cursor = 0
        self._selection = None
        self._selection_payload = None
        self._admission_order = 0
        self._last_batch = -1
        self.counters = Counter()
        self.last_counts = {}

    @property
    def entries(self):
        return tuple(sorted(self._entries.values(), key=lambda e: (e.true_label, e.sample_id)))

    @property
    def class_order(self):
        return tuple(sorted({e.true_label for e in self.entries}))

    @property
    def class_cursor(self):
        return self._class_cursor

    @property
    def selection(self):
        return self._selection

    def admit(self, entries, current_sample_ids=()):
        if self._selection is not None:
            raise ValueError("Finish the current frozen batch before admitting replay")
        current = set(current_sample_ids)
        incoming = {}
        for entry in entries:
            self.eligibility.verify(entry)
            if entry.sample_id in current:
                raise ValueError("Current unlabelled samples cannot be admitted as replay")
            if entry.sample_id in incoming and incoming[entry.sample_id] != entry:
                raise ValueError("Conflicting duplicate replay reference")
            incoming[entry.sample_id] = entry
        # A completed selection is a journal event, not a live buffer snapshot.
        # Keep its batch cursor separately when the inventory next changes.
        self._selection_payload = None
        self.last_counts = {}
        # Validate everything before changing state; canonical order fixes ages.
        for entry in sorted(incoming.values(), key=lambda e: (e.true_label, e.sample_id)):
            if entry.sample_id in self._entries:
                continue
            same = [e for e in self._entries.values() if e.true_label == entry.true_label]
            if len(same) >= self.per_class:
                continue
            stored = self.eligibility.entry(entry.sample_id, entry.admission_batch, self._admission_order)
            self._admission_order += 1
            self._entries[entry.sample_id] = stored
            if len(self._entries) > self.capacity:
                counts = Counter(e.true_label for e in self._entries.values())
                victim = min(self._entries.values(), key=lambda e: (-counts[e.true_label], e.admission_order,
                                                                  e.true_label, e.sample_id))
                del self._entries[victim.sample_id]

    def select(self, batch_id, current_sample_ids=()):
        if type(batch_id) is not int or batch_id < 0:
            raise ValueError("Replay selection requires a nonnegative incoming batch ID")
        if self._selection is not None:
            if self._selection.batch_id != batch_id:
                raise ValueError("Finish the frozen batch before selecting another")
            if set(self._selection.sample_ids) & set(current_sample_ids):
                raise ValueError("Selected labelled replay overlaps unlabelled current batch")
            return self._selection
        if batch_id <= self._last_batch:
            raise ValueError("Replay batch selection cannot be repeated after completion")
        classes = self.class_order
        by_class = {c: tuple(e.sample_id for e in self.entries if e.true_label == c) for c in classes}
        pre = post = self._class_cursor
        ordered = [c for c in classes if c >= pre] + [c for c in classes if c < pre]
        chosen = []
        for c in ordered:
            post = classes[(classes.index(c)+1) % len(classes)]
            if len(by_class[c]) == 2:
                chosen.append(c)
            if len(chosen) == 16:
                break
        ids = tuple(k for c in chosen for k in by_class[c])
        if set(ids) & set(current_sample_ids):
            raise ValueError("Selected labelled replay overlaps unlabelled current batch")
        p = dict(schema=REPLAY_SCHEMA, eligibility_manifest=self.eligibility.identity, batch_id=batch_id,
                 pre_cursor=pre, post_cursor=post,
                 selected=[dict(class_id=c, sample_ids=list(by_class[c])) for c in chosen])
        hashed = digest(p)
        self._selection = ReplaySelection(batch_id, tuple(chosen), ids, pre, post, hashed)
        self._selection_payload = p
        self._class_cursor = post
        self._last_batch = batch_id
        self.last_counts = {INELIGIBLE: sum(len(v) < 2 for v in by_class.values()),
                            CLASS_SELECTED: len(chosen), REFERENCE_SELECTED: len(ids), SELECTION_HASH: hashed}
        self.counters.update({k: v for k, v in self.last_counts.items() if k != SELECTION_HASH})
        return self._selection

    def finish_batch(self, batch_id):
        if self._selection is None or self._selection.batch_id != batch_id:
            raise ValueError("No matching frozen replay batch")
        self._selection = None

    def state_dict(self):
        p = dict(schema=REPLAY_SCHEMA, provider="verified_train_labelled", capacity=self.capacity,
                 per_class=self.per_class, selection_size=self.selection_size,
                 dataset_fingerprint=self.eligibility.dataset_fingerprint,
                 split_fingerprint=self.eligibility.split_fingerprint, eligibility_manifest=self.eligibility.identity,
                 entries=[asdict(e) for e in self.entries], class_order=list(self.class_order),
                 class_cursor=self._class_cursor, admission_order=self._admission_order,
                 last_batch=self._last_batch,
                 selected=asdict(self._selection) if self._selection else None,
                 selection_payload=self._selection_payload, counters=dict(self.counters), last_counts=dict(self.last_counts))
        # JSON copy ensures serialization uses only primitive metadata, never tensors/objects.
        return json.loads(canonical(dict(payload=p, state_hash=digest(p))))

    def load_state_dict(self, state):
        expected = set(self.state_dict()["payload"])
        if not isinstance(state, dict) or set(state) != {"payload", "state_hash"}:
            raise ValueError("Invalid replay state envelope")
        p = state["payload"]
        if not isinstance(p, dict) or set(p) != expected or digest(p) != state["state_hash"]:
            raise ValueError("Corrupt replay state/hash/fields")
        for k in ("schema", "provider", "capacity", "per_class", "selection_size", "dataset_fingerprint",
                  "split_fingerprint", "eligibility_manifest"):
            if p[k] != self.state_dict()["payload"][k]:
                raise ValueError("Replay schema/provider/manifest/config mismatch: " + k)
        restored = VerifiedLabeledReplay(self.eligibility)
        entries = [VerifiedReplayEntry(**e) for e in p["entries"]]
        for entry in entries:
            self.eligibility.verify(entry)
        if len({e.sample_id for e in entries}) != len(entries) or len(entries) > self.capacity:
            raise ValueError("Invalid replay capacity/duplicate state")
        if any(n > 2 for n in Counter(e.true_label for e in entries).values()):
            raise ValueError("Invalid replay class capacity")
        restored._entries = {e.sample_id: e for e in entries}
        if p["class_order"] != list(restored.class_order):
            raise ValueError("Corrupt canonical class order")
        for k in ("class_cursor", "admission_order"):
            if type(p[k]) is not int or p[k] < 0:
                raise ValueError("Invalid replay cursor/age")
        restored._class_cursor, restored._admission_order = p["class_cursor"], p["admission_order"]
        if type(p["last_batch"]) is not int or p["last_batch"] < -1:
            raise ValueError("Invalid replay last-batch cursor")
        if entries and (len({e.admission_order for e in entries}) != len(entries)
                        or max(e.admission_order for e in entries) >= restored._admission_order):
            raise ValueError("Invalid replay admission ages")
        sp = p["selection_payload"]
        if sp is not None:
            if (sp.get("schema") != REPLAY_SCHEMA or sp.get("eligibility_manifest") != self.eligibility.identity
                    or sp.get("post_cursor") != restored.class_cursor):
                raise ValueError("Corrupt replay selection identity/cursor")
            # Re-run deterministic metadata-only selection to validate pairs/hash/counters.
            restored._class_cursor = sp["pre_cursor"]
            selected = restored.select(sp["batch_id"])
            if p["last_batch"] != sp["batch_id"]:
                raise ValueError("Corrupt replay last-batch cursor")
            if canonical(restored._selection_payload) != canonical(sp):
                raise ValueError("Corrupt replay selection payload")
            if p["selected"] is not None and canonical(asdict(selected)) != canonical(p["selected"]):
                raise ValueError("Corrupt selected sample IDs/hash")
            if p["last_counts"] != restored.last_counts:
                raise ValueError("Corrupt replay selection counters/hash")
            if p["selected"] is None:
                restored._selection = None
        elif p["selected"] is not None or p["last_counts"]:
            raise ValueError("Missing replay selection payload")
        if any(k not in (INELIGIBLE, CLASS_SELECTED, REFERENCE_SELECTED) or type(v) is not int or v < 0
               for k, v in p["counters"].items()):
            raise ValueError("Invalid replay counters")
        restored.counters = Counter(p["counters"])
        restored.last_counts = dict(p["last_counts"])
        restored._last_batch = p["last_batch"]
        self.__dict__.update(restored.__dict__)
