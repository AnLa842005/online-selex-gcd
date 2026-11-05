"""Metadata-only verified replay: complete pairs, canonical cursor and resume."""
from copy import deepcopy
from dataclasses import replace
import json
import random
import unittest

import torch

from dean_integration.selex_hierarchy_provider import digest
from dean_integration.selex_labeled_replay import (
    CLASS_SELECTED, INELIGIBLE, MANIFEST_SCHEMA, REFERENCE_SELECTED, SELECTION_HASH,
    VerifiedEligibilityManifest, VerifiedLabeledReplay)


def manifest(classes=20, singleton=False, per_class=2):
    records = [dict(sample_id=10*c+j+1, canonical_path=f"images/{c}/{j}.jpg", true_label=c)
               for c in range(classes) for j in range(per_class if not singleton or c else 1)]
    p = dict(schema=MANIFEST_SCHEMA, dataset_fingerprint="a"*64, split_fingerprint="b"*64,
             source_split="train_labelled", label_origin="ground_truth", records=records)
    return p, VerifiedEligibilityManifest(p, digest(p), "a"*64, "b"*64)


def filled(classes=20, singleton=False, reverse=False):
    p, m = manifest(classes, singleton)
    entries = [m.entry(row["sample_id"]) for row in p["records"]]
    if reverse:
        entries.reverse()
    buffer = VerifiedLabeledReplay(m)
    buffer.admit(entries)
    return buffer


class VerifiedReplayTests(unittest.TestCase):
    def test_only_verified_train_labelled_ground_truth(self):
        _, m = manifest()
        b = VerifiedLabeledReplay(m)
        e = m.entry(1)
        for origin in ("pseudo_label", "current_gt_y", "vfa_gaussian"):
            with self.assertRaises(ValueError):
                replace(e, label_origin=origin)
        for value in (torch.randn(768), {"gt_y": 1}, {"pseudo_label": 0}, {"vfa_feature": [1.]}, "image"):
            with self.assertRaises(ValueError):
                b.admit([value])
        for changed in (replace(e, true_label=7), replace(e, canonical_path="other.jpg"),
                        replace(e, dataset_fingerprint="f"*64), replace(e, eligibility_manifest="c"*64)):
            with self.assertRaises(ValueError):
                b.admit([changed])
        with self.assertRaises(ValueError):
            b.admit([e], current_sample_ids=[1])
        self.assertEqual(b.entries, ())
        b.admit([e])
        self.assertEqual(b.entries[0].true_label, 0)

    def test_manifest_schema_hash_dataset_split_and_duplicates(self):
        p, _ = manifest()
        for k, v in (("label_origin", "pseudo_label"), ("source_split", "unlabelled"), ("schema", "unknown")):
            bad = deepcopy(p)
            bad[k] = v
            with self.assertRaises(ValueError):
                VerifiedEligibilityManifest(bad, digest(bad), "a"*64, "b"*64)
        for bad in (dict(p, records=p["records"] + [p["records"][0]]),
                    dict(p, records=list(reversed(p["records"]))),
                    dict(p, records=[dict(p["records"][0], gt_y=0)])):
            with self.assertRaises(ValueError):
                VerifiedEligibilityManifest(bad, digest(bad), "a"*64, "b"*64)
        for h, d, s in (("0"*64, "a"*64, "b"*64), (digest(p), "f"*64, "b"*64), (digest(p), "a"*64, "f"*64)):
            with self.assertRaises(ValueError):
                VerifiedEligibilityManifest(p, h, d, s)

    def test_selection_complete_pairs_maximum_16_classes_and_counters(self):
        b = filled(30)
        selected = b.select(0)
        self.assertEqual(len(selected.sample_ids), 32)
        self.assertEqual(len(selected.class_ids), 16)
        for c in selected.class_ids:
            self.assertEqual(sum(b.eligibility.entry(k).true_label == c for k in selected.sample_ids), 2)
        self.assertEqual(b.last_counts[CLASS_SELECTED], 16)
        self.assertEqual(b.last_counts[REFERENCE_SELECTED], 32)
        self.assertEqual(b.last_counts[SELECTION_HASH], selected.selection_hash)

    def test_singleton_and_second_view_cannot_create_anchor(self):
        _, m = manifest(1, singleton=True)
        b = VerifiedLabeledReplay(m)
        # Two copies/views of one reference are deduplicated, not distinct support.
        b.admit([m.entry(1), m.entry(1)])
        self.assertEqual(len(b.entries), 1)
        s = b.select(0, current_sample_ids=[100])
        self.assertEqual(s.class_ids, ())
        self.assertEqual(s.sample_ids, ())
        self.assertEqual(b.last_counts[INELIGIBLE], 1)
        self.assertEqual(b.last_counts[CLASS_SELECTED], 0)
        self.assertEqual(b.last_counts[REFERENCE_SELECTED], 0)

    def test_underfilled_selection_keeps_complete_classes_only(self):
        b = filled(4, singleton=True)
        s = b.select(0)
        self.assertEqual(s.class_ids, (1, 2, 3))
        self.assertEqual(len(s.sample_ids), 6)
        self.assertEqual(b.last_counts[INELIGIBLE], 1)

    def test_loader_order_independent_admission_selection_hash_and_rng(self):
        a, b = filled(reverse=False), filled(reverse=True)
        rng = random.getstate()
        torch_rng = torch.get_rng_state().clone()
        self.assertEqual(a.state_dict(), b.state_dict())
        self.assertEqual(a.select(0), b.select(0))
        self.assertEqual(random.getstate(), rng)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        self.assertEqual(a.state_dict(), b.state_dict())

    def test_round_robin_cursor_is_class_key_and_wraps(self):
        b = filled(20)
        self.assertEqual(b.select(0).class_ids, tuple(range(16)))
        self.assertEqual(b.class_cursor, 16)
        b.finish_batch(0)
        self.assertEqual(b.select(1).class_ids, (16, 17, 18, 19, *range(12)))
        self.assertEqual(b.class_cursor, 12)

    def test_midbatch_resume_restores_exact_cursor_selection_and_hash(self):
        b = filled(24)
        s = b.select(0)
        state = b.state_dict()
        restored = VerifiedLabeledReplay(b.eligibility)
        restored.load_state_dict(json.loads(json.dumps(state)))
        self.assertEqual(restored.state_dict(), state)
        self.assertEqual(restored.select(0), s)
        self.assertEqual(restored.state_dict(), state)  # No duplicate counter/cursor advance.
        for buffer in (b, restored):
            buffer.finish_batch(0)
        self.assertEqual(b.select(1), restored.select(1))
        self.assertEqual(b.state_dict(), restored.state_dict())

    def test_boundary_resume_and_admission_after_completed_selection(self):
        b = filled(24)
        b.select(0)
        b.finish_batch(0)
        restored = VerifiedLabeledReplay(b.eligibility)
        restored.load_state_dict(b.state_dict())
        self.assertEqual(b.state_dict(), restored.state_dict())
        b.admit([])
        restored.load_state_dict(b.state_dict())
        self.assertEqual(b.select(1), restored.select(1))
        with self.assertRaises(ValueError):
            b.admit([])

    def test_corrupt_state_hash_schema_provider_manifest_selection(self):
        b = filled()
        b.select(0)
        state = b.state_dict()
        for field, value in (("schema", "bad"), ("provider", "dean_prototypes"),
                             ("eligibility_manifest", "c"*64), ("class_cursor", 1234),
                             ("capacity", 321), ("class_order", [999])):
            bad = deepcopy(state)
            bad["payload"][field] = value
            bad["state_hash"] = digest(bad["payload"])
            with self.assertRaises(ValueError):
                VerifiedLabeledReplay(b.eligibility).load_state_dict(bad)
        bad = deepcopy(state)
        bad["payload"]["selected"]["selection_hash"] = "0"*64
        bad["state_hash"] = digest(bad["payload"])
        with self.assertRaises(ValueError):
            b.load_state_dict(bad)
        bad = deepcopy(state)
        bad["state_hash"] = "bad"
        with self.assertRaises(ValueError):
            b.load_state_dict(bad)

    def test_capacity_eviction_and_per_class_capacity(self):
        p, m = manifest(170, per_class=3)
        b = VerifiedLabeledReplay(m)
        b.admit([m.entry(r["sample_id"]) for r in reversed(p["records"])])
        self.assertLessEqual(len(b.entries), 320)
        for c in b.class_order:
            self.assertLessEqual(sum(e.true_label == c for e in b.entries), 2)
        restored = VerifiedLabeledReplay(m)
        restored.load_state_dict(b.state_dict())
        self.assertEqual(b.state_dict(), restored.state_dict())
        first = filled(160)
        self.assertEqual(len(first.entries), 320)
        # Original oldest references are eviction candidates, not a raw tensor bank.
        self.assertTrue(any(e.sample_id not in {x.sample_id for x in b.entries}
                            for e in [m.entry(1), m.entry(2)]))

    def test_serialized_state_has_only_bounded_primitive_references(self):
        b = filled()
        b.select(0)
        encoded = json.dumps(b.state_dict())
        for forbidden in ("pixels", "PIL", "feature_tensor", "pseudo_label", "gt_y", "vfa_feature"):
            self.assertNotIn(forbidden, encoded)
        state = b.state_dict()
        self.assertLessEqual(len(state["payload"]["entries"]), 320)
        state["payload"]["entries"][0]["true_label"] = 999
        self.assertEqual(b.entries[0].true_label, 0)

    def test_empty_selection_preserves_cursor_and_can_resume(self):
        _, m = manifest(0)
        b = VerifiedLabeledReplay(m)
        self.assertEqual(b.select(0).sample_ids, ())
        self.assertEqual(b.class_cursor, 0)
        other = VerifiedLabeledReplay(m)
        other.load_state_dict(b.state_dict())
        self.assertEqual(other.state_dict(), b.state_dict())


if __name__ == "__main__":
    unittest.main()
