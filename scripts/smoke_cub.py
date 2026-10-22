"""Small real-CUB exercise of upstream training, all 200 SSB classes retained.

This is validation only. No smoke accuracy is a research result.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from copy import deepcopy
import json
import time
import uuid
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from data.data_utils import MergedDataset
from data.augmentations import get_transform
from config import exp_root, dino_pretrain_path
from project_utils.reproduction import load_dino_backbone
from methods.contrastive_training import contrastive_training as upstream
from scripts.verify_cub import verify


def take_per_class(dataset, count):
    result = deepcopy(dataset)
    positions = []
    for target in sorted(set(dataset.data.target)):
        positions.extend(np.flatnonzero(dataset.data.target.to_numpy() == target)[:count])
    result.data = result.data.iloc[positions].copy()
    result.uq_idxs = dataset.uq_idxs[positions].copy()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, choices=(2, 4), default=4)
    parser.add_argument('--max-train-batches', type=int, default=1)
    opts = parser.parse_args()
    assert opts.max_train_batches > 0
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(1)
    np.random.seed(1)
    device = torch.device(opts.device)
    upstream.device = device
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    args, _, test, datasets = verify(False)
    args.__dict__.update(dict(batch_size=opts.batch_size, num_workers=0, image_size=224,
        interpolation=3, crop_pct=0.875, n_views=2, temperature=1.0,
        num_labeled_classes=100, num_unlabeled_classes=100, epochs=1,
        lr=0.1, momentum=0.9, weight_decay=1e-4, sup_con_weight=0.35,
        strategy='zero_one', cluster_momentum=1, unsupervised_smoothing=1.0,
        train_report_interval=200, prototype_extraction_interval=1,
        distance='euclidean', contrast_unlabel_only=False, eval_funcs=['v1','v2'],
        unbalanced=False, gpu_clustering=device.type == 'cuda',
        max_train_batches=opts.max_train_batches, resume=None))
    transform, test_transform = get_transform('imagenet', 224, args)
    views = upstream.ContrastiveLearningViewGenerator(transform, 2)
    labelled = take_per_class(datasets['train_labelled'], 1)
    # One old unlabelled image and two novel images per class: two total per class.
    unlabelled = take_per_class(datasets['train_unlabelled'], 2)
    keep = []
    for target in sorted(set(unlabelled.data.target)):
        indices = np.flatnonzero(unlabelled.data.target.to_numpy() == target)
        keep.extend(indices[:1 if target - 1 in args.train_classes else 2])
    unlabelled.data = unlabelled.data.iloc[keep].copy()
    unlabelled.uq_idxs = unlabelled.uq_idxs[keep]
    # Dense IDs only for the debug subset; preserve the original IDs in the report.
    original_ids = np.concatenate([labelled.uq_idxs, unlabelled.uq_idxs]).tolist()
    labelled.uq_idxs = np.arange(len(labelled))
    unlabelled.uq_idxs = np.arange(len(labelled), len(labelled) + len(unlabelled))
    labelled.transform = unlabelled.transform = views
    merged = MergedDataset(labelled, unlabelled)
    # Explicit mixed batches ensure SSE has labelled examples with tiny batch sizes.
    novel_positions = [i for i, target in enumerate(unlabelled.data.target)
                       if target - 1 in args.unlabeled_classes]
    indices = []
    half = opts.batch_size // 2
    for b in range(opts.max_train_batches):
        indices.extend([(b * half + j) % len(labelled) for j in range(half)])
        indices.extend([len(labelled) + novel_positions[(b * half + j) % len(novel_positions)] for j in range(half)])
    train_loader = DataLoader(Subset(merged, indices), batch_size=opts.batch_size)
    merge_loader = DataLoader(merged, batch_size=opts.batch_size)
    test = take_per_class(test, 1)
    test.transform = test_transform
    test_loader = DataLoader(test, batch_size=opts.batch_size)
    model = upstream.vits.vit_base()
    model.load_state_dict(load_dino_backbone(dino_pretrain_path), strict=True)
    print('Backbone strict load PASS', flush=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = 'blocks.' in name and int(name.split('.')[1]) >= 10
    model.to(device)
    head = upstream.vits.DINOHead(768, 65536, nlayers=3).to(device)
    output = Path(exp_root) / ('smoke-' + uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    args.model_dir = str(output)
    args.model_path = str(output / 'model.pt')
    args.writer = SummaryWriter(str(output / 'tensorboard'))
    # Upstream plotting uses a relative Plots directory; isolate it per smoke run.
    import os
    previous = Path.cwd()
    (output / 'Plots').mkdir()
    os.chdir(output)
    try:
        upstream.train(head, model, train_loader, test_loader, test_loader, merge_loader, args)
    finally:
        os.chdir(previous)
        args.writer.close()
    state = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state, strict=True)
    assert all(torch.isfinite(v).all() for v in state.values())
    head_state = torch.load(output / 'model_proj_head.pt', map_location='cpu')
    head.load_state_dict(head_state, strict=True)
    assert all(torch.isfinite(v).all() for v in head_state.values())
    report = dict(status='PASS', device=str(device), batch_size=opts.batch_size,
        batches=opts.max_train_batches, classes=200, labelled=len(labelled),
        unlabelled=len(unlabelled), evaluation_samples=len(test),
        original_train_ids=original_ids, checkpoint_save_load='PASS',
        forward='PASS', backward='PASS', optimizer_step='PASS', HSSK='PASS',
        evaluation='PASS', losses=args.smoke_losses,
        runtime_seconds=time.monotonic()-started,
        peak_vram_mb=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None)
    (output / 'smoke_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='original_train_ids'}, indent=2))
    print('Output:', output)


if __name__ == '__main__':
    main()
