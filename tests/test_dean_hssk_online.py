"""Leaf-only HSSK invariants; no production hierarchy/training is enabled."""
import ast
from copy import deepcopy
from dataclasses import replace
import inspect
import json
import random
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from dean_integration.config import ROOT
from dean_integration.selex_config import SOURCE_SHA
from dean_integration.selex_hierarchy_provider import FullTrainingBlocked, ensure_provider_ready, digest
from dean_integration.selex_config import ReferenceConfig
from dean_integration.selex_hssk_online import (
    CandidateFit, HSSKLeafConfig, HSSKLeafFitter, HSSKPrototypeMemory,
    LeafFitStatus, VerifiedReplayFeatures, candidate_counts, checked_rows)
from dean_integration.selex_labeled_replay import MANIFEST_SCHEMA, VerifiedEligibilityManifest, VerifiedLabeledReplay


def basis(n, offset=0):
    x = torch.zeros(n, 768)
    if n:
        x[torch.arange(n), (torch.arange(n)+offset) % 768] = 1
    return x


def replay_features(a=1, batch_id=0):
    records = [dict(sample_id=10000+10*c+j, canonical_path=f"images/{c}/{j}.jpg", true_label=11+7*c)
               for c in range(a) for j in range(2)]
    p = dict(schema=MANIFEST_SCHEMA, dataset_fingerprint="a"*64, split_fingerprint="b"*64,
             source_split="train_labelled", label_origin="ground_truth", records=records)
    m = VerifiedEligibilityManifest(p, digest(p), "a"*64, "b"*64)
    buffer = VerifiedLabeledReplay(m)
    buffer.admit([m.entry(r["sample_id"]) for r in records])
    selected = buffer.select(batch_id)
    values = torch.repeat_interleave(basis(a), 2, dim=0)
    return VerifiedReplayFeatures.from_selection(buffer, values, selected.sample_ids), buffer, values


def publish(memory, centers, batch, gt=None, counts=None):
    keys = sorted(centers)
    plan, mapping = memory.updated_copy(centers, counts or {k: 2 for k in keys}, {k: .1 for k in keys},
                                       gt or {k: None for k in keys}, batch, digest(dict(batch=batch)))
    memory.load_state_dict(plan.state_dict())
    return mapping


class LeafFitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.fitter = HSSKLeafFitter()

    def fit(self, x, ids=None, **kw):
        memory = kw.pop("memory", HSSKPrototypeMemory())
        return self.fitter.fit(x, list(range(len(x))) if ids is None else ids,
                               memory=memory, seed=kw.pop("seed", 17), batch_id=kw.pop("batch_id", 0), **kw)

    def test_row_permutation_does_not_change_result_memory_or_hash(self):
        x = torch.repeat_interleave(basis(3), 4, dim=0)
        ids = [91, 8, 19, 77, 30, 40, 2, 1, 52, 71, 90, 15]
        r, buffer, values = replay_features(2)
        a, b = HSSKPrototypeMemory(), HSSKPrototypeMemory()
        first = self.fit(x, ids, replay=r, memory=a)
        perm = torch.tensor([9, 2, 0, 11, 4, 8, 3, 10, 5, 1, 7, 6])
        replay_ids = list(buffer.selection.sample_ids)
        other_replay = VerifiedReplayFeatures.from_selection(buffer, values.flip(0), replay_ids[::-1])
        second = self.fit(x[perm], torch.tensor(ids)[perm], replay=other_replay, memory=b)
        self.assertEqual(first.fit_hash, second.fit_hash)
        self.assertEqual(first.state_dict(), second.state_dict())
        self.assertEqual(a.state_dict(), b.state_dict())

    def test_api_cannot_accept_dean_classifier_vfa_masks_or_gt_diagnostics(self):
        x = torch.repeat_interleave(basis(2), 2, dim=0)
        signature = inspect.signature(self.fitter.fit)
        self.assertFalse(any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()))
        for name in ("dean_pseudo_labels", "classifier_ids", "classifier_weights", "known_mask",
                     "ap_labels", "vfa_features", "gt_y", "dean_ema"):
            with self.assertRaises(TypeError):
                self.fit(x, **{name: torch.randn(4)})
        expected = self.fit(x).fit_hash
        # Perturb all global RNGs and unrelated DEAN-shaped data. The fitter has
        # no access to any of them or to the legacy EMA/provider constructors.
        random.seed(999)
        np.random.seed(998)
        torch.manual_seed(997)
        _ = torch.randn(160, 768)
        with patch("dean_integration.selex_hierarchy.EMAHierarchy", side_effect=AssertionError("DEAN EMA")), \
             patch("dean_integration.selex_hierarchy_provider.DeanPrototypeHierarchyProvider", side_effect=AssertionError("fallback")):
            self.assertEqual(self.fit(x).fit_hash, expected)

    def test_duplicate_nonfinite_wrong_width_and_misaligned_rows_are_hard_errors(self):
        x = basis(2)
        for values, ids in ((x, [1, 1]), (x, [1]), (x[:, :767], [1, 2]), (x, [-1, 2]),
                            (x.long(), [1, 2]), (x, [1., 2.])):
            with self.assertRaises(ValueError):
                self.fit(values, ids)
        for bad in (float("nan"), float("inf")):
            values = x.clone()
            values[0, 0] = bad
            with self.assertRaises(ValueError):
                self.fit(values)

    def test_detach_and_fp16_bf16_fp32_under_autocast(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            x = torch.repeat_interleave(basis(2), 2, dim=0).to(dtype).requires_grad_()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                result = self.fit(x)
            self.assertEqual(result.status, LeafFitStatus.SUCCESS)
            self.assertIsNone(x.grad)
            for center in result.centers.values():
                self.assertEqual(center.dtype, torch.float32)
                self.assertFalse(center.requires_grad)
                self.assertIsNone(center.grad_fn)
                self.assertTrue(torch.isfinite(center).all())

    def test_feature_axis_normalization_and_copy_out(self):
        x = basis(2)*torch.tensor([[7.], [13.]])
        rows, _ = checked_rows(x.requires_grad_(), [9, 1])
        self.assertFalse(rows.requires_grad)
        result = self.fit(torch.repeat_interleave(x, 2, dim=0))
        captured = result.fit_hash
        result.centers[next(iter(result.centers))].zero_()
        self.assertEqual(captured, result.fit_hash)
        self.assertTrue(all(abs(float(c.norm())-1) < 1e-6 for c in result.centers.values()))

    def test_verified_gt_assignments_fixed_in_every_phase(self):
        replay, _, _ = replay_features(2)
        x = torch.repeat_interleave(basis(3), 3, dim=0)
        original = self.fitter._assign
        phases = []
        def observed(rows, centers, fixed, balanced):
            labels, distances = original(rows, centers, fixed, balanced)
            self.assertTrue(torch.equal(labels[fixed >= 0], fixed[fixed >= 0]))
            phases.append(balanced)
            return labels, distances
        with patch.object(self.fitter, "_assign", side_effect=observed):
            result = self.fit(x, replay=replay)
        self.assertIn(True, phases)
        self.assertIn(False, phases)
        _, replay_ids, labels, _ = replay.values()
        state = result.state_dict()
        for key, gt in zip(replay_ids.tolist(), labels.tolist()):
            group = result.sample_to_leaf[key]
            self.assertEqual(state["leaves"][str(group)]["anchor_gt"], gt)
            self.assertLess(group, 0)

    def test_two_gt_classes_with_coincident_centers_are_not_merged(self):
        _, buffer, values = replay_features(2)
        values[:] = basis(1)[0]
        replay = VerifiedReplayFeatures.from_selection(buffer, values, buffer.selection.sample_ids)
        result = self.fit(torch.empty(0, 768), replay=replay)
        self.assertEqual(result.selected_k, 2)
        self.assertEqual(len(set(result.sample_to_leaf.values())), 2)
        self.assertEqual({v["anchor_gt"] for v in result.state_dict()["leaves"].values()}, {11, 18})

    def test_replay_cannot_be_bare_labels_or_singletons_or_overlap_current(self):
        with self.assertRaises(ValueError):
            self.fit(basis(2), replay={"gt_y": [0, 0]})
        replay, buffer, values = replay_features(1)
        ids = list(buffer.selection.sample_ids)
        with self.assertRaises(ValueError):
            VerifiedReplayFeatures.from_selection(buffer, values[:1], ids[:1])
        with self.assertRaises(ValueError):
            VerifiedReplayFeatures.from_selection(buffer, values, [ids[0], ids[0]])
        with self.assertRaises(ValueError):
            self.fit(basis(2), ids, replay=replay)
        with self.assertRaises(ValueError):
            self.fit(basis(2), replay=replay, batch_id=1)

    def test_k_bounds_exhaustive_n_and_a_and_zero_free_only_with_anchors(self):
        for n in range(0, 129):
            for a in range(min(16, n//2)+1):
                expected = tuple(a+f for f in range(0 if a else 1, min(16, n//2-a)+1))
                self.assertEqual(candidate_counts(n, a), expected)
                for k in candidate_counts(n, a):
                    self.assertGreaterEqual(k, max(a, 1))
                    self.assertLessEqual(k, min(a+16, n//2))
                    self.assertTrue(k-a > 0 or a > 0)
        for n, a in ((-1, 0), (1, 1), (64, 17), (True, 0)):
            with self.assertRaises(ValueError):
                candidate_counts(n, a)

    def test_support_properties_for_small_current_and_replay_pools(self):
        for b in (0, 1, 2, 3, 4, 7, 12):
            for a in (0, 1, 3):
                with self.subTest(b=b, a=a):
                    replay = replay_features(a)[0] if a else None
                    x = basis(b, 4)
                    result = self.fit(x, replay=replay)
                    if b+2*a < 2:
                        self.assertEqual(result.status, LeafFitStatus.INSUFFICIENT_SUPPORT)
                        continue
                    self.assertEqual(result.status, LeafFitStatus.SUCCESS)
                    p = result.state_dict()
                    self.assertIn(result.selected_k, candidate_counts(b+2*a, a))
                    self.assertEqual(sum(v["support_count"] for v in p["leaves"].values()), b+2*a)
                    self.assertEqual(len(result.sample_to_leaf), b+2*a)
                    for group in p["leaves"].values():
                        self.assertGreaterEqual(group["support_count"], 2)
                        if not group["anchored"]:
                            self.assertGreaterEqual(group["current_count"], 2)

    def test_refinement_singleton_rejects_k_after_balanced_phase(self):
        x = torch.cat((basis(1).repeat(3, 1), basis(1, 1)))
        result = self.fit(x)
        self.assertEqual(result.selected_k, 1)
        self.assertEqual(result.state_dict()["rejected_k"]["2"], "post_refinement_min_real_support")

    def test_empty_reseed_uses_current_high_residual_and_preserves_gt(self):
        rows = basis(6)
        ids = torch.tensor([10, 20, 30, 40, 50, 60])
        current = torch.tensor([False, False, True, True, True, True])
        labels = torch.zeros(6, dtype=torch.long)
        distances = torch.tensor([[0., 2.], [0., 2.], [1., 2.], [2., 2.], [3., 2.], [4., 2.]])
        centers = basis(2)
        seeds = self.fitter._repair_empty(rows, ids, current, labels, centers, 1, distances)
        self.assertEqual(seeds, [60])
        self.assertTrue(torch.equal(centers[1], rows[5]))
        self.assertEqual(labels[:2].tolist(), [0, 0])
        self.assertEqual(int((labels == 1).sum()), 2)

    def test_empty_center_without_current_donor_is_rejected(self):
        rows = basis(3)
        labels = torch.zeros(3, dtype=torch.long)
        self.assertIsNone(self.fitter._repair_empty(rows, torch.arange(3), torch.tensor([False, False, True]),
                                                   labels, basis(2), 1, torch.ones(3, 2)))

    def test_score_formula_uses_only_real_rows(self):
        x = torch.cat((basis(1).repeat(4, 1), basis(1, 1).repeat(4, 1)))
        m = HSSKPrototypeMemory()
        publish(m, {0: basis(1, 2)[0]}, 0, counts={0: 100000})
        result = self.fit(x, memory=m, batch_id=1)
        p = result.state_dict()
        self.assertEqual(set(result.sample_to_leaf), set(range(8)))
        self.assertEqual(sum(v["support_count"] for v in p["leaves"].values()), 8)
        real_error = sum(float((x[i]-result.centers[result.sample_to_leaf[i]]).square().sum()) for i in range(8))/8
        self.assertAlmostEqual(p["score"], real_error+.02*p["k_free"], places=6)
        self.assertLessEqual(len(p["memory_seed_ids"]), 128)

    def test_score_tie_selects_smaller_k(self):
        x = torch.repeat_interleave(basis(2), 2, dim=0)
        def equal_score(rows, ids, current, fixed, initial, a):
            labels = torch.arange(len(rows)) // (len(rows)//len(initial))
            centers = torch.stack([rows[labels == j].mean(0) for j in range(len(initial))])
            return CandidateFit(labels, centers, .5, {}), None
        with patch.object(self.fitter, "_fit_candidate", side_effect=equal_score):
            self.assertEqual(self.fit(x).selected_k, 1)

    def test_same_seed_state_result_hash_and_global_rng_isolation(self):
        x = torch.repeat_interleave(basis(3), 3, dim=0)
        python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        a, b = self.fit(x), self.fit(x)
        self.assertEqual(a.fit_hash, b.fit_hash)
        self.assertEqual(a.state_dict(), b.state_dict())
        self.assertEqual(random.getstate(), python)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy[1]))
        self.assertEqual(np.random.get_state()[2:], numpy[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu))

    def test_failed_rejected_fit_and_hard_errors_do_not_mutate_memory_allocator(self):
        m = HSSKPrototypeMemory()
        publish(m, {0: basis(1)[0]}, 0)
        before = m.state_dict()
        for x, ids in ((torch.empty(0, 768), []), (basis(1), [8])):
            r = self.fit(x, ids, memory=m, batch_id=1)
            self.assertEqual(r.status, LeafFitStatus.INSUFFICIENT_SUPPORT)
            self.assertEqual(r.state_dict()["memory_hash_before"], r.state_dict()["memory_hash_after"])
            self.assertEqual(m.state_dict(), before)
        with patch.object(self.fitter, "_fit_candidate", return_value=(None, "fixture_rejection")):
            self.assertEqual(self.fit(basis(4), memory=m, batch_id=1).status, LeafFitStatus.INSUFFICIENT_SUPPORT)
        self.assertEqual(m.state_dict(), before)
        with self.assertRaises(ValueError):
            self.fit(basis(2), [8, 8], memory=m, batch_id=1)
        self.assertEqual(m.state_dict(), before)

    def test_success_updates_once_and_duplicate_batch_is_error(self):
        m = HSSKPrototypeMemory()
        self.fit(basis(4), memory=m)
        after = m.state_dict()
        with self.assertRaises(ValueError):
            self.fit(basis(4), memory=m)
        self.assertEqual(after, m.state_dict())

    def test_batch_64_with_16_verified_anchor_classes_has_real_support_only(self):
        replay = replay_features(16)[0]
        current = torch.repeat_interleave(basis(16, 32), 4, dim=0)
        result = self.fit(current, replay=replay)
        self.assertEqual(result.status, LeafFitStatus.SUCCESS)
        p = result.state_dict()
        self.assertEqual(p["a"], 16)
        self.assertEqual(p["n"], 96)
        self.assertLessEqual(result.selected_k, 32)
        self.assertEqual(sum(v["support_count"] for v in p["leaves"].values()), 96)
        self.assertEqual(sum(v["anchored"] for v in p["leaves"].values()), 16)

    def test_refined_anchor_centers_are_final_real_means_not_stale_gt_means(self):
        rows = torch.cat((basis(1).repeat(2, 1), F.normalize(basis(1)+.1*basis(1, 1), dim=1).repeat(2, 1)))
        fixed = torch.tensor([0, 0, -1, -1])
        candidate, reason = self.fitter._fit_candidate(rows, torch.arange(4), fixed < 0, fixed, basis(1), 1)
        self.assertIsNone(reason)
        torch.testing.assert_close(candidate.centers[0], rows.mean(0), rtol=0, atol=0)
        self.assertLess(float(candidate.centers[0].norm()), 1.)
        self.assertFalse(torch.equal(candidate.centers[0], basis(1)[0]))

    def test_production_provider_stays_fail_closed(self):
        self.fit(torch.repeat_interleave(basis(2), 2, dim=0))
        with self.assertRaisesRegex(FullTrainingBlocked, "Phase 4"):
            ensure_provider_ready(ReferenceConfig(hierarchy_source="hssk_online"))

    def test_locked_numerical_config_validation(self):
        for kw in (dict(max_free=40), dict(min_support=1), dict(k_penalty=.01), dict(momentum=.8),
                   dict(max_iterations=11), dict(tolerance=.01), dict(memory_capacity=513),
                   dict(memory_seed_limit=129), dict(epsilon=float("nan"))):
            with self.assertRaises(ValueError):
                HSSKLeafConfig(**kw)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_momentum_matching_counts_detach_and_negative_monotonic_ids(self):
        memory = HSSKPrototypeMemory()
        a = publish(memory, {0: basis(1)[0]}, 0)
        self.assertEqual(a[0], -1)
        new = .8*basis(1)[0]+.2*basis(1, 1)[0]
        expected = F.normalize(.9*basis(1)[0]+.1*new, dim=0)
        b = publish(memory, {0: new.requires_grad_()}, 1)
        self.assertEqual(b, a)
        torch.testing.assert_close(memory.tracks[-1].center, expected, rtol=0, atol=0)
        self.assertFalse(memory.tracks[-1].center.requires_grad)
        self.assertEqual(memory.tracks[-1].support_count, 4)
        self.assertEqual(memory.tracks[-1].last_active_batch, 1)
        self.assertEqual(memory.tracks[-1].updates, 2)
        mapping = publish(memory, {0: basis(1)[0], 1: basis(1, 2)[0]}, 2)
        self.assertTrue(all(k < 0 for k in mapping.values()))
        self.assertLess(memory.next_id, -2)

    def test_capacity_512_seed_limit_128_and_no_active_eviction(self):
        memory = HSSKPrototypeMemory()
        centers = dict(enumerate(basis(512)))
        publish(memory, centers, 0, gt={k: k for k in centers})
        self.assertEqual(len(memory.tracks), 512)
        before = memory.state_dict()
        seeds, ids = memory.seed_candidates(basis(64))
        self.assertEqual(seeds.shape, (128, 768))
        self.assertEqual(len(set(ids)), 128)
        self.assertEqual(before, memory.state_dict())
        active = publish(memory, {0: basis(1)[0], 1: basis(1, 1)[0]}, 1, gt={0: 600, 1: 601})
        self.assertEqual(len(memory.tracks), 512)
        self.assertEqual(set(active.values()), {-513, -514})
        self.assertNotIn(-512, memory.tracks)
        self.assertNotIn(-511, memory.tracks)
        self.assertTrue(set(active.values()) <= set(memory.tracks))

    def test_inactive_eviction_and_ties_are_deterministic(self):
        cfg = HSSKLeafConfig(memory_capacity=3, memory_seed_limit=2)
        a, b = HSSKPrototypeMemory(cfg), HSSKPrototypeMemory(cfg)
        for m in (a, b):
            publish(m, dict(enumerate(basis(3))), 0, gt={0: 0, 1: 1, 2: 2})
            publish(m, {0: basis(1, 4)[0]}, 1, gt={0: 4})
        self.assertEqual(a.state_dict(), b.state_dict())
        self.assertNotIn(-3, a.tracks)
        self.assertIn(-4, a.tracks)

    def test_failed_memory_update_never_mutates_original_allocator(self):
        cfg = HSSKLeafConfig(memory_capacity=2)
        memory = HSSKPrototypeMemory(cfg)
        publish(memory, {0: basis(1)[0]}, 0)
        before = memory.state_dict()
        with self.assertRaises(ValueError):
            publish(memory, dict(enumerate(basis(3))), 1)
        self.assertEqual(before, memory.state_dict())
        with self.assertRaises(ValueError):
            publish(memory, {0: basis(1)[0]}, 0)
        self.assertEqual(before, memory.state_dict())

    def test_seed_relevance_ties_and_copy_out(self):
        memory = HSSKPrototypeMemory(HSSKLeafConfig(memory_seed_limit=1))
        publish(memory, {0: basis(1)[0], 1: basis(1, 1)[0]}, 0)
        seeds, ids = memory.seed_candidates(basis(1, 1))
        self.assertEqual(ids, (-2,))
        seeds.zero_()
        self.assertEqual(float(memory.tracks[-2].center[1]), 1.)

    def test_state_resume_reproduces_fit_hash_memory_and_allocator(self):
        memory = HSSKPrototypeMemory()
        publish(memory, {0: basis(1)[0]}, 0)
        other = HSSKPrototypeMemory()
        other.load_state_dict(json.loads(json.dumps(memory.state_dict())))
        x = torch.repeat_interleave(basis(3), 3, dim=0)
        a = HSSKLeafFitter().fit(x, range(9), memory=memory, seed=7, batch_id=1)
        b = HSSKLeafFitter().fit(x, range(9), memory=other, seed=7, batch_id=1)
        self.assertEqual(a.fit_hash, b.fit_hash)
        self.assertEqual(memory.state_dict(), other.state_dict())

    def test_memory_provenance_records_verified_anchor_manifest_without_samples(self):
        replay, buffer, _ = replay_features(1)
        memory = HSSKPrototypeMemory()
        result = HSSKLeafFitter().fit(basis(2), [8, 90], replay=replay, memory=memory, seed=1, batch_id=0)
        self.assertEqual(result.status, LeafFitStatus.SUCCESS)
        anchored = [t for t in memory.tracks.values() if t.anchor_gt is not None]
        self.assertEqual(len(anchored), 1)
        origin = json.loads(anchored[0].provenance)
        self.assertEqual(origin["eligibility_manifest"], buffer.eligibility.identity)
        self.assertEqual(origin["label_origin"], "ground_truth")
        self.assertNotIn("sample_ids", origin)

    def test_state_rejects_corrupt_hash_provider_schema_allocator_and_extra_history(self):
        memory = HSSKPrototypeMemory()
        publish(memory, {0: basis(1)[0]}, 0)
        state = memory.state_dict()
        for key, value in (("provider", "dean_prototypes"), ("schema", "bad"), ("next_id", 0)):
            bad = deepcopy(state)
            bad["payload"][key] = value
            bad["state_hash"] = digest(bad["payload"])
            with self.assertRaises(ValueError):
                memory.load_state_dict(bad)
        bad = deepcopy(state)
        bad["payload"]["tracks"][0]["sample_feature_history"] = []
        bad["state_hash"] = digest(bad["payload"])
        with self.assertRaises(ValueError):
            memory.load_state_dict(bad)
        bad = deepcopy(state)
        bad["state_hash"] = "0"*64
        with self.assertRaises(ValueError):
            memory.load_state_dict(bad)
        self.assertEqual(state, memory.state_dict())

    def test_serialized_state_is_bounded_summaries_without_raw_sample_bank(self):
        memory = HSSKPrototypeMemory()
        publish(memory, {0: basis(1)[0]}, 0)
        state = memory.state_dict()
        encoded = json.dumps(state)
        for key in ("pixels", "PIL", "sample_ids", "gt_y", "image_paths", "feature_history", "vfa"):
            self.assertNotIn(key, encoded)
        self.assertEqual(len(state["payload"]["tracks"][0]["center_values"]), 768)


class CorrectedReferenceTests(unittest.TestCase):
    def test_nondegenerate_fixed_k_two_phase_reference_fixture(self):
        # Extract the authoritative source without importing CUDA/data/sklearn
        # infrastructure. Only device routing is corrected; this supported
        # fixture never enters buggy empty/capacity-repair branches.
        path = "methods/clustering/faster_mix_k_means_pytorch.py"
        source = subprocess.check_output(["git", "show", SOURCE_SHA+":"+path], cwd=ROOT, text=True)
        tree = ast.parse(source)
        pairwise = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pairwise_distance")
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "K_Means")
        method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "fit_mix_once")
        class CPUDevice(ast.NodeTransformer):
            def visit_Constant(self, node):
                if node.value == "cuda":
                    node.value = "cpu"
                return node
        method = CPUDevice().visit(method)
        module = ast.fix_missing_locations(ast.Module(body=[pairwise, method], type_ignores=[]))
        ns = dict(torch=torch, np=np)
        exec(compile(module, "<SelEx-c3d3b42-supported-BSSK-fixture>", "exec"), ns)
        torch.set_num_threads(1)
        labels = torch.tensor([11, 11, 29, 29])
        labelled = torch.repeat_interleave(basis(2), 2, dim=0)
        unlabelled = torch.cat((labelled, basis(1, 2).repeat(4, 1)))
        # Nonzero in-group geometry, with four real rows in each well-separated leaf.
        labelled[0, 10], labelled[1, 10] = .02, -.02
        # Equal mean norms keep this nondegenerate fixture outside the literal
        # source's stale best-centers bookkeeping defect on early convergence.
        unlabelled[0, 11], unlabelled[1, 11] = .02, -.02
        labelled, unlabelled = F.normalize(labelled, dim=1), F.normalize(unlabelled, dim=1)
        initial = torch.stack((labelled[:2].mean(0), labelled[2:].mean(0), unlabelled[4:].mean(0)))
        obj = SimpleNamespace(k=3, tolerance=1e-4, max_iterations=10, protos=initial.numpy(),
                              cluster_size=4, pairwise_batch_size=None)
        expected_labels, _, expected_centers, _ = ns["fit_mix_once"](obj, unlabelled, labelled, labels, None)
        rows = torch.cat((labelled, unlabelled))
        fixed = torch.tensor([0, 0, 1, 1]+[-1]*8)
        current = fixed < 0
        actual, reason = HSSKLeafFitter()._fit_candidate(rows, torch.arange(12), current, fixed, initial, 2)
        self.assertIsNone(reason)
        self.assertTrue(torch.equal(expected_labels, actual.labels))
        torch.testing.assert_close(expected_centers, actual.centers, atol=1e-6, rtol=1e-6)
        self.assertTrue(actual.diagnostics["balanced"]["converged"])
        self.assertTrue(actual.diagnostics["refinement"]["converged"])


if __name__ == "__main__":
    unittest.main()
