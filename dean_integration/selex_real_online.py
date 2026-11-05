"""SelEx adapted to DEAN online discovery; explicit raw-image/replay boundaries."""
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from .provenance import write_json
from .selex_config import SOURCE_SHA
from .selex_hierarchy import EMAHierarchy, integer_ids
from .selex_projector import SelExProjector
from .selex_reference import expertise_losses
from .selex_rng import SelExRNG


class CurrentBatchImages:
    """Only current real images; no targets or historical images are exposed."""
    def __init__(self, sample_ids, load_image, train_transform, eval_transform):
        self.sample_ids = integer_ids(sample_ids, "image IDs").cpu().clone()
        if len(set(self.sample_ids.tolist())) != len(self.sample_ids):
            raise ValueError("Duplicate current image IDs")
        self.allowed = set(self.sample_ids.tolist())
        self.load_image, self.train_transform, self.eval_transform = load_image, train_transform, eval_transform

    @classmethod
    def from_cub(cls, dataset, make_transform):
        ids = integer_ids(dataset.uq_idxs, "CUB sample IDs").cpu()
        if len(ids) != len(dataset.data):
            raise ValueError("CUB data/sample IDs differ in length")
        positions = {key: i for i, key in enumerate(ids.tolist())}

        def load_image(key):
            # Access only filepath. Never invoke __getitem__ or read sample.target/gt_y.
            row = dataset.data.iloc[positions[key]]
            path = Path(dataset.root) / dataset.base_folder / row.filepath
            return dataset.loader(str(path))

        return cls(ids, load_image, make_transform(is_train=True), make_transform(is_train=False))

    def images(self, sample_ids, *, two_views):
        ids = integer_ids(sample_ids, "requested image IDs").cpu().tolist()
        if len(set(ids)) != len(ids) or not set(ids) <= self.allowed:
            raise ValueError("Requested images are duplicate, historical or outside the current batch")
        if not ids:
            raise ValueError("Cannot fetch an empty raw-image request")
        output = []
        for key in ids:
            image = self.load_image(key)
            transform = self.train_transform if two_views else self.eval_transform
            def copy_image():
                return image.clone() if isinstance(image, torch.Tensor) else image.copy()
            output.append(torch.stack([transform(copy_image()) for _ in range(2)]) if two_views
                          else transform(copy_image()))
        return torch.stack(output)


@dataclass(frozen=True)
class LabeledReplay:
    """Explicit supervised caller input; production wrapper supplies none by default.

    Availability and true-label provenance belong to the replay provider. This
    object is never inferred from current DEAN pseudo labels or gt_y diagnostics.
    """
    images: CurrentBatchImages
    true_labels: torch.Tensor

    def __post_init__(self):
        labels = integer_ids(self.true_labels, "true replay labels")
        if len(labels) != len(self.images.sample_ids) or (labels < 0).any():
            raise ValueError("True replay labels must align with explicit replay images")
        object.__setattr__(self, "true_labels", labels.detach().cpu().clone())


def assert_optimizer_order(optimizer, projector, classifier):
    expected = {id(p) for p in classifier.parameters()}
    groups = optimizer.param_groups
    if not groups or {id(p) for p in groups[-1]["params"]} != expected:
        raise ValueError("DEAN dynamically expanded classifier must be the final optimizer group")
    positions = [i for i, g in enumerate(groups) if g.get("name") == "selex_projector"]
    if len(positions) != 1 or positions[0] != len(groups)-2:
        raise ValueError("SelEx projector must immediately precede the final classifier group")
    if {id(p) for p in groups[positions[0]]["params"]} != {id(p) for p in projector.parameters() if p.requires_grad}:
        raise ValueError("Projector optimizer group contains incorrect parameters")
    all_ids = [id(p) for g in groups for p in g["params"]]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Optimizer parameter groups overlap")


class RealSelExOnline:
    def __init__(self, config, seed, output=None, base_classes=160):
        if config.gamma <= 0:
            raise ValueError("Gamma-zero must bypass real SelEx construction entirely")
        self.config, self.rng = config, SelExRNG(seed)
        self.hierarchy = EMAHierarchy(config, base_classes)
        self.projector = None
        self.counters = Counter()
        self.batch, self.step = -1, 0
        self.output = Path(output) if output else None
        self.telemetry = None
        self.optimizer_groups = []

    def start_session(self, model):
        if self.projector is not None:
            raise ValueError("Real v1 expects the official single incremental session")
        self.device = next(model.parameters()).device
        with self.rng.scope(self.device):
            self.projector = SelExProjector(self.config).to(self.device)
        if self.output:
            self.telemetry = self.output / "selex_reference_metrics.jsonl"
            with self.telemetry.open("x", encoding="utf-8"):
                pass
        self._event("session", config=asdict(self.config), source_sha=SOURCE_SHA, rng_seed=self.rng.seed,
                    projector_trainable=sum(p.numel() for p in self.projector.parameters() if p.requires_grad),
                    projector_frozen=sum(p.numel() for p in self.projector.parameters() if not p.requires_grad))

    def prepare_batch(self, model, dataset, make_transform, num_classes):
        with self.rng.scope(self.device):
            images = CurrentBatchImages.from_cub(dataset, make_transform)
        labels = integer_ids(dataset.ys, "DEAN pseudo labels")
        self.prepare(model, images, labels, num_classes)

    @torch.no_grad()
    def prepare(self, model, images, labels, num_classes):
        if self.projector is None:
            raise ValueError("Real SelEx session not initialized")
        labels = integer_ids(labels, "DEAN pseudo labels").cpu()
        if len(labels) != len(images.sample_ids) or type(num_classes) is not int or num_classes < 1:
            raise ValueError("DEAN labels/class inventory/current IDs must align")
        valid = (labels >= 0) & (labels < num_classes)
        self.counters["invalid_pseudo_samples"] += int((~valid).sum())
        self.images = images  # Replaces the previous current-batch provider; no raw history.
        self.current_labels = dict(zip(images.sample_ids.tolist(), labels.tolist()))
        self.num_classes = num_classes
        states = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            if valid.any():
                with self.rng.scope(self.device):
                    inputs = images.images(images.sample_ids[valid], two_views=False).to(self.device)
                    features = model(inputs)
            else:
                features = torch.empty(0, 768, device=self.device)
            self.snapshot = self.hierarchy.update(features, labels[valid], images.sample_ids[valid])
        finally:
            for module, training in states:
                module.training = training
        self.batch += 1
        self.step = 0
        self._event("batch", batch=self.batch, samples=len(labels), valid_samples=int(valid.sum()),
                    prototype_count=len(self.hierarchy.prototypes), hierarchy_levels=len(self.snapshot.mappings))

    def attach_optimizer(self, optimizer, classifier):
        if not optimizer.param_groups or {id(p) for p in optimizer.param_groups[-1]["params"]} != {id(p) for p in classifier.parameters()}:
            raise ValueError("Classifier must be final before adding projector optimizer group")
        if not any(g.get("name") == "selex_projector" for g in optimizer.param_groups):
            last = optimizer.param_groups.pop()
            optimizer.add_param_group(dict(params=[p for p in self.projector.parameters() if p.requires_grad],
                                           lr=self.config.projector_lr, name="selex_projector"))
            optimizer.param_groups.append(last)
        assert_optimizer_order(optimizer, self.projector, classifier)
        self.optimizer_groups = [dict(index=i, name=g.get("name", "dean_classifier" if i == len(optimizer.param_groups)-1 else f"dean_group_{i}"),
                                      lr=g["lr"], trainable=sum(p.numel() for p in g["params"] if p.requires_grad),
                                      weight_decay=g.get("weight_decay"), betas=g.get("betas"),
                                      tensors=len(g["params"])) for i, g in enumerate(optimizer.param_groups)]
        self._event("optimizer", groups=self.optimizer_groups,
                    trainable_parameters=sum(g["trainable"] for g in self.optimizer_groups))

    def augment(self, official_loss, ce, weighted_ec, model, pseudo_labels, sample_ids, replay=None):
        if self.batch < 0:
            raise ValueError("Auxiliary update before current batch hierarchy preparation")
        ids = integer_ids(sample_ids, "optimization sample IDs").cpu()
        labels = integer_ids(pseudo_labels, "optimization pseudo labels").cpu()
        if len(ids) != len(labels) or len(set(ids.tolist())) != len(ids):
            raise ValueError("Optimization sample IDs/labels must align uniquely")
        if any(key not in self.current_labels or self.current_labels[key] != label for key, label in zip(ids.tolist(), labels.tolist())):
            raise ValueError("Optimization labels/IDs differ from discovery snapshot")
        if not torch.isfinite(official_loss).all():
            raise ValueError("Nonfinite official DEAN loss; not silently repaired by SelEx")
        valid = (labels >= 0) & (labels < self.num_classes)
        self.counters["invalid_pseudo_update_samples"] += int((~valid).sum())
        labels = labels[valid].to(self.device)
        ids = ids[valid]
        self.projector.train()
        with self.rng.scope(self.device):
            if len(ids):
                views = self.images.images(ids, two_views=True).to(self.device)
                # View-major model input, matching the reference.
                inputs = torch.cat(views.unbind(1), dim=0)
                features = model(inputs)
                if features.shape != (2*len(ids), 768) or not torch.isfinite(features).all():
                    raise ValueError("Live DEAN backbone features must be finite [2B,768]")
                geometry = features.reshape(2, len(ids), 768).transpose(0, 1)
                flat_projected = self.projector(features)
                projected = flat_projected.reshape(2, len(ids), self.config.output_dim).transpose(0, 1)
            else:
                geometry = torch.empty(0, 2, 768, device=self.device)
                projected = self.projector(torch.empty(0, 768, device=self.device)).reshape(0, 2, self.config.output_dim)
            replay_q, replay_y = None, None
            if replay is not None:
                if not isinstance(replay, LabeledReplay):
                    raise ValueError("GT supervision accepts only explicit LabeledReplay")
                if set(replay.images.sample_ids.tolist()) & set(self.images.sample_ids.tolist()):
                    raise ValueError("Labelled replay IDs must be disjoint from current pseudo-labelled stream")
                if len(replay.images.sample_ids):
                    replay_views = replay.images.images(replay.images.sample_ids, two_views=True).to(self.device)
                    replay_features = model(torch.cat(replay_views.unbind(1), dim=0))
                    replay_flat = self.projector(replay_features)
                    replay_q = replay_flat.reshape(2, len(replay.images.sample_ids), self.config.output_dim).transpose(0, 1)
                    replay_y = replay.true_labels.to(self.device).long()
                else:
                    self.counters["sse.gt.empty_replay"] += 1
            losses = expertise_losses(projected, geometry, labels, self.snapshot, self.config,
                                      replay_projected=replay_q, replay_labels=replay_y, counters=self.counters)
            total = official_loss + self.config.gamma * losses["selex_total"]
        if not torch.isfinite(total).all():
            raise ValueError("Nonfinite real SelEx total")
        self.step += 1
        values = {name: float(value.detach()) for name, value in losses.items()}
        self._event("update", batch=self.batch, step=self.step, ce=float(ce.detach()), ec=float(weighted_ec.detach()),
                    **values, final_total=float(total.detach()), counters=dict(self.counters),
                    replay_samples=0 if replay is None else len(replay.images.sample_ids))
        return total

    def _event(self, event, **values):
        if self.telemetry:
            with self.telemetry.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(event=event, **values), allow_nan=False) + "\n")
            write_json(self.output / "selex_reference_summary.json",
                       dict(implementation="reference_v1", source_sha=SOURCE_SHA, config=asdict(self.config),
                            rng_seed=self.rng.seed, batch=self.batch, step=self.step,
                            counters=dict(self.counters), hierarchy_counters=dict(self.hierarchy.counters),
                            prototype_counts=dict(self.hierarchy.counts),
                            optimizer_groups=self.optimizer_groups,
                            projector_trainable=sum(p.numel() for p in self.projector.parameters() if p.requires_grad)))
