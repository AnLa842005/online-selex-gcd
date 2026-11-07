"""Frozen live-view bridge; verified replay and hierarchy never see stream GT."""
from copy import deepcopy
import torch
from .selex_hierarchy import integer_ids
from .selex_hierarchy_provider import HierarchySupportPool, HierarchyBuildStatus
from .selex_hssk_online import VerifiedReplayFeatures
from .selex_reference import expertise_losses


@torch.no_grad()
def prepare(engine, model, images):
    if not engine.lifecycle_mode:
        return _prepare(engine, model, images)
    replay = engine.replay_manager.state_dict() if engine.replay_manager else None
    provider = engine.provider.state_dict()
    domains = {name: deepcopy(vars(getattr(engine,name))) for name in ("eval_rng",)}
    try:
        return _prepare(engine, model, images)
    except Exception as exc:
        engine.provider.load_state_dict(provider)
        engine.provider.counters["hssk.error"] += 1
        if replay is not None: engine.replay_manager.load_state_dict(replay)
        for name,state in domains.items(): getattr(engine,name).__dict__.update(state)
        engine._event("error", reason=str(exc), error_type=type(exc).__name__)
        raise


@torch.no_grad()
def _prepare(engine, model, images):
    if engine.batch >= 0 and not engine.lifecycle_mode:
        raise ValueError("Bounded smoke permits one incoming batch only")
    if engine.lifecycle_mode and engine.phase not in ("READY", "BOUNDARY"):
        raise ValueError("Previous batch freeze window is not complete")
    if hasattr(engine, "smoke_budget"):
        engine.smoke_budget.hierarchy_started()
    batch = engine.batch+1
    ids = images.sample_ids
    selected_images, verified_features = None, None
    if engine.replay_manager is not None:
        selection = engine.replay_manager.select(batch, ids.tolist())
        if selection.sample_ids:
            selected_images = engine.replay_image_factory(selection.sample_ids)
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with engine.eval_rng.scope(engine.device), torch.autocast(device_type=engine.device.type, enabled=False):
            features = model(images.images(ids, two_views=False).to(engine.device)).float() if len(ids) else torch.empty(0, 768)
            if selected_images is not None:
                replay_rows = model(selected_images.images(selected_images.sample_ids, two_views=False).to(engine.device)).float()
                verified_features = VerifiedReplayFeatures.from_selection(engine.replay_manager, replay_rows, selected_images.sample_ids)
    finally:
        for module, training in states:
            module.training = training
    context = dict(engine.hssk_context, batch=batch, seed=engine.hssk_seed)
    result = engine.provider.build(HierarchySupportPool(ids, features), labeled_replay=verified_features, context=context)
    engine.build_result = result
    engine.snapshot = result.snapshot
    engine.images, engine.selected_replay_images = images, selected_images
    engine.batch, engine.step = batch, 0
    engine.phase = "FROZEN"
    engine.frozen_hash = engine.snapshot.snapshot_hash if engine.snapshot else None
    if hasattr(engine, "smoke_budget"):
        engine.smoke_budget.hierarchy_completed()
    if selected_images is None:
        engine.counters["sse.gt.unavailable.no_verified_selection"] += 1
    engine._event("hierarchy" if engine.lifecycle_mode else "batch", batch=batch, provider_status=result.status.value, reason=result.reason,
                  snapshot_hash=engine.snapshot.snapshot_hash if engine.snapshot else None,
                  diagnostics=engine.snapshot.diagnostics if engine.snapshot else {},
                  replay=engine.replay_manager.last_counts if engine.replay_manager else {},
                  memory_hash=engine.provider.memory.state_hash)


def augment(engine, official_loss, ce, ec, model, sample_ids, external_replay):
    if external_replay is not None:
        raise ValueError("HSSK GT SSE accepts only its owned verified replay selection")
    if engine.batch < 0 or engine.step >= 15:
        raise ValueError("Bounded smoke update outside prepared 15-epoch batch")
    if engine.lifecycle_mode and (engine.phase != "FROZEN" or engine.frozen_hash != (engine.snapshot.snapshot_hash if engine.snapshot else None)):
        raise ValueError("Hierarchy changed inside freeze window")
    ids = integer_ids(sample_ids, "optimization sample IDs").cpu()
    if len(set(ids.tolist())) != len(ids) or not set(ids.tolist()) <= engine.images.allowed:
        raise ValueError("Optimization IDs must belong to the frozen current batch")
    if not torch.isfinite(official_loss).all():
        raise ValueError("Nonfinite official DEAN loss")
    if hasattr(engine, "smoke_budget") and engine.smoke_budget.tick is None:
        engine.smoke_budget.update_started("auxiliary_to_optimizer_step")
    if engine.build_result.status == HierarchyBuildStatus.SKIP:
        zero = official_loss*0
        losses = {name: zero for name in ("use", "sse", "sse_gt", "sse_hierarchy", "selex_total")}
        engine.counters["selex.batch_skipped_updates"] += 1
    else:
        engine.projector.train()
        with engine.rng.scope(engine.device):
            views = engine.images.images(ids, two_views=True).to(engine.device)
            live = model(torch.cat(views.unbind(1), dim=0))
            geometry = live.reshape(2, len(ids), 768).transpose(0, 1)
            q = engine.projector(live).reshape(2, len(ids), engine.config.output_dim).transpose(0, 1)
        replay_q, replay_labels = None, None
        replay = engine.selected_replay_images
        if replay is not None:
            labels = []
            for key in replay.sample_ids.tolist():
                entry = next(e for e in engine.replay_manager.entries if e.sample_id == key)
                engine.replay_manager.eligibility.verify(entry)
                labels.append(entry.true_label)
            with engine.replay_rng.scope(engine.device):
                views = replay.images(replay.sample_ids, two_views=True).to(engine.device)
                live_replay = model(torch.cat(views.unbind(1), dim=0))
                replay_q = engine.projector(live_replay).reshape(2, len(labels), engine.config.output_dim).transpose(0, 1)
            replay_labels = torch.tensor(labels, dtype=torch.long, device=engine.device)
        losses = expertise_losses(q, geometry, ids.to(engine.device), engine.snapshot, engine.config,
                                  replay_projected=replay_q, replay_labels=replay_labels, counters=engine.counters)
    total = official_loss+engine.config.gamma*losses["selex_total"]
    if not torch.isfinite(total).all():
        raise ValueError("Nonfinite HSSK online total")
    engine.step += 1
    engine.smoke_losses = dict(ce=float(ce.detach()), ec=float(ec.detach()),
                              **{k: float(v.detach()) for k, v in losses.items()}, total=float(total.detach()))
    event = dict(batch=engine.batch, step=engine.step, ce=float(ce.detach()), ec=float(ec.detach()),
                  **{k: float(v.detach()) for k, v in losses.items()}, final_total=float(total.detach()),
                  counters=dict(engine.counters), snapshot_hash=engine.snapshot.snapshot_hash if engine.snapshot else None)
    if engine.lifecycle_mode:
        engine.pending_update = event
    else:
        engine._event("update", **event)
    return total
