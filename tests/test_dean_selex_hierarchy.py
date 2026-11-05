"""Causal, bounded EMA prototype and ID mapping contracts."""
from dataclasses import replace
import unittest

import torch

from dean_integration.selex_config import ReferenceConfig
from dean_integration.selex_hierarchy import EMAHierarchy


class HierarchyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = replace(ReferenceConfig(), prototype_capacity=8)

    def test_dean_leaves_authoritative_and_all_level_ids_globally_unique(self):
        y = torch.tensor([0, 9, 18, 40, 160, 169, 211, 512])
        bank = EMAHierarchy(self.config)
        snapshot = bank.update(torch.randn(8, 768), y, torch.arange(8)*10)
        self.assertEqual(snapshot.mappings[0], {int(k): int(k) for k in y})
        used = set()
        for mapping in snapshot.mappings:
            ids = set(mapping.values())
            self.assertFalse(used & ids)
            used |= ids
            self.assertEqual(set(mapping), set(y.tolist()))
            for group_id in ids:
                members = [leaf for leaf, value in mapping.items() if value == group_id]
                self.assertTrue(all(k < 160 for k in members) or all(k >= 160 for k in members))
        parent_ids = used-set(y.tolist())
        second = bank.update(torch.randn(8, 768), y, torch.arange(8)*10)
        newer = set().union(*(set(m.values()) for m in second.mappings[1:]))
        self.assertFalse(parent_ids & newer)  # IDs also never recycle across rebuilds.

    def test_sparse_permuted_ids_and_prototypes_are_order_invariant(self):
        y = torch.tensor([10, 10, 88, 88, 165, 165])
        ids = torch.tensor([200, 4, 91, 505, 33, 8000])
        z = torch.randn(6, 768)
        a, b = EMAHierarchy(self.config), EMAHierarchy(self.config)
        s1 = a.update(z, y, ids)
        perm = torch.tensor([5, 2, 0, 4, 1, 3])
        s2 = b.update(z[perm], y[perm], ids[perm])
        self.assertEqual(s1.mappings, s2.mappings)
        for key in a.prototypes:
            torch.testing.assert_close(a.prototypes[key], b.prototypes[key])
        assigned = s1.assignments(y[perm])
        self.assertEqual(assigned[0].tolist(), y[perm].tolist())

    def test_ema_normalization_snapshot_is_detached_and_fixed(self):
        y = torch.tensor([0, 0, 1, 1])
        bank = EMAHierarchy(self.config)
        z = torch.randn(4, 768, requires_grad=True)
        snapshot = bank.update(z, y, torch.arange(4))
        before = snapshot.centers[0][0].clone()
        bank.update(torch.randn(4, 768), y, torch.arange(4))
        self.assertTrue(torch.equal(before, snapshot.centers[0][0]))
        self.assertFalse(snapshot.centers[0][0].requires_grad)
        for center in bank.prototypes.values():
            torch.testing.assert_close(center.norm(), torch.tensor(1.))
        self.assertEqual(bank.counts[0], 4)
        self.assertEqual(bank.counters["hierarchy_rebuilds"], 2)

    def test_bounded_deterministic_eviction_protects_active_classes(self):
        config = replace(self.config, prototype_capacity=3)
        bank = EMAHierarchy(config)
        bank.update(torch.randn(3, 768), torch.tensor([0, 1, 2]), torch.arange(3))
        bank.update(torch.randn(2, 768), torch.tensor([1, 3]), torch.arange(2))
        self.assertEqual(set(bank.prototypes), {1, 2, 3})
        self.assertEqual(bank.counters["prototype_evictions"], 1)
        self.assertEqual(set(bank.counts), set(bank.prototypes))
        with self.assertRaisesRegex(ValueError, "capacity"):
            bank.update(torch.randn(4, 768), torch.arange(4), torch.arange(4))

    def test_empty_all_known_all_novel_singleton_and_ties(self):
        for labels in ([], [8], [0, 1, 2, 3], [160, 161, 162, 163]):
            y = torch.tensor(labels, dtype=torch.long)
            bank = EMAHierarchy(self.config)
            s = bank.update(torch.ones(len(y), 768), y, torch.arange(len(y)))
            self.assertEqual(set(s.leaf_ids), set(labels))
            self.assertEqual(bank.counters["hierarchy_rebuilds"], 1)
            if labels:
                self.assertEqual(len(set(s.mappings[-1].values())), 1)
            else:
                self.assertEqual(s.mappings, ())

    def test_invalid_structure_and_nonfinite_inputs_raise(self):
        bank = EMAHierarchy(self.config)
        for z, y, ids in ((torch.randn(2, 767), torch.tensor([0, 1]), torch.tensor([2, 3])),
                          (torch.randn(2, 768), torch.tensor([0, 1]), torch.tensor([2, 2])),
                          (torch.randn(2, 768), torch.tensor([-1, 1]), torch.tensor([2, 3])),
                          (torch.full((2, 768), float("nan")), torch.tensor([0, 1]), torch.tensor([2, 3]))):
            with self.assertRaises(ValueError):
                bank.update(z, y, ids)


if __name__ == "__main__":
    unittest.main()
