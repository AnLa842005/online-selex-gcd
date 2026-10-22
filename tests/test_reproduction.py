import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from project_utils.reproduction import limited_batches, save_training_state, restore_training_state, load_dino_backbone
from methods.contrastive_training.contrastive_training import SupConLoss, LabelSmoothingLoss


class ReproductionTests(unittest.TestCase):
    def test_upstream_algorithm_parity(self):
        import ast
        import subprocess
        root = Path(__file__).resolve().parents[1]
        commit = 'c3d3b42a0aa7d7c9ea748a8305a6890a34100f73'
        def original(path):
            return subprocess.check_output(['git', 'show', commit+':'+path], cwd=root, text=True)
        path = 'methods/contrastive_training/contrastive_training.py'
        before, after = original(path), (root/path).read_text()
        old, new = ast.parse(before), ast.parse(after)
        for name in ('SupConLoss', 'LabelSmoothingLoss', 'info_nce_logits',
                     'extract_labeled_protos', 'test_kmeans'):
            x = next(n for n in old.body if getattr(n, 'name', None) == name)
            y = next(n for n in new.body if getattr(n, 'name', None) == name)
            normalized = ast.unparse(x).replace('images.cuda()', 'images.to(device)').replace(
                'images[0].cuda()', 'images[0].to(device)')
            self.assertEqual(ast.dump(ast.parse(normalized)), ast.dump(ast.parse(ast.unparse(y))), name)
        start, end = '            # Extract features with base model', '            # Train acc'
        self.assertEqual(before[before.index(start):before.index(end)],
                         after[after.index(start):after.index(end)])
        path = 'methods/clustering/faster_mix_k_means_pytorch.py'
        normalized = original(path).replace('from sklearn.utils._joblib import', 'from joblib import').replace(
            "        device = torch.device('cuda')", '        device = u_feats.device')
        normalized = normalized.replace('np.arange(self.k)', 'np.arange(self.k, dtype=np.int64)')
        self.assertEqual(ast.dump(ast.parse(normalized)), ast.dump(ast.parse((root/path).read_text())))
        path = 'project_utils/cluster_and_log_utils.py'
        self.assertEqual(original(path), (root/path).read_text())

    def test_paths(self):
        import config
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(os.environ, {'SELEX_DATA_ROOT': root,
                    'SELEX_PRETRAINED_PATH': str(Path(root)/'weights.pth'),
                    'SELEX_OUTPUT_ROOT': str(Path(root)/'output')}):
                importlib.reload(config)
                self.assertEqual(config.cub_root, root)
                self.assertEqual(config.exp_root, str(Path(root)/'output'))
                self.assertEqual(config.dino_pretrain_path, str(Path(root)/'weights.pth'))
                self.assertTrue((Path(config.osr_split_dir)/'cub_osr_splits.pkl').is_file())
        importlib.reload(config)

    def test_limit(self):
        self.assertEqual(list(limited_batches(range(4))), [0,1,2,3])
        self.assertEqual(list(limited_batches(range(4), 2)), [0,1])
        with self.assertRaises(ValueError): limited_batches([], -1)

    def test_checkpoint_resume(self):
        model, head = torch.nn.Linear(2,2), torch.nn.Linear(2,2)
        optimizer = torch.optim.SGD(list(model.parameters())+list(head.parameters()), lr=.1, momentum=.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
        head(model(torch.ones(2,2))).sum().backward()
        optimizer.step()
        scheduler.step()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'training.pt'
            expected = model.weight.detach().clone()
            save_training_state(path, model, head, optimizer, scheduler, 0)
            expected_rng = torch.rand(3)
            model.weight.data.zero_()
            state = restore_training_state(path, model, head, optimizer, scheduler)
            self.assertTrue(torch.equal(model.weight, expected))
            self.assertTrue(torch.equal(torch.rand(3), expected_rng))
            self.assertEqual(state['epoch'], 0)
            self.assertTrue(optimizer.state)
            torch.save({'teacher': {'backbone.weight': expected, 'head.weight': expected}}, path)
            self.assertEqual(set(load_dino_backbone(path)), {'weight'})

    def test_upstream_losses_finite_backward(self):
        torch.manual_seed(1)
        features = torch.nn.functional.normalize(torch.randn(4,2,16), dim=-1).requires_grad_()
        sse = SupConLoss()(features, labels=torch.tensor([0,0,1,1]))
        logits = torch.randn(8,7, requires_grad=True)
        similarity = torch.softmax(torch.randn(8,7), dim=1)
        use = LabelSmoothingLoss()(logits, torch.zeros(8,dtype=torch.long), similarity, 1.0)
        total = .65*use + .35*sse/2
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_hssk_cpu_integer_portability(self):
        from methods.clustering.faster_mix_k_means_pytorch import K_Means
        labelled = torch.tensor([[-10., 0.], [-10., .1], [10., 0.], [10., .1]])
        unlabelled = torch.tensor([[-10., -.1], [10., -.1], [0., 10.], [.1, 10.],
                                   [-.1, 10.], [0., -10.], [.1, -10.], [-.1, -10.]])
        clustering = K_Means(k=4, max_iterations=3, n_init=1, random_state=0,
                             pairwise_batch_size=8, cluster_size=3)
        clustering.fit_mix(unlabelled, labelled, torch.tensor([0, 0, 1, 1]))
        self.assertEqual(clustering.labels_.dtype, torch.int64)
        self.assertEqual(len(clustering.labels_), 12)
        self.assertTrue(torch.isfinite(clustering.cluster_centers_).all())


if __name__ == '__main__': unittest.main()
