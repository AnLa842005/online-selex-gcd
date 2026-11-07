"""Versioned strict batch-boundary state. No raw images or historical feature bank."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
import platform
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .preflight import sha256
from .selex_hierarchy_provider import HierarchySnapshot, HierarchyBuildResult, HierarchyBuildStatus, HierarchyProviderKind, digest
from .selex_real_online import assert_optimizer_order

SCHEMA = "dean-hssk-boundary-v1"
IDENTITY_FIELDS = {"git_sha", "upstream_sha", "config_sha256", "selex_config_sha256",
                   "dataset_sha256", "dino_sha256", "base_sha256", "replay_manifest_sha256",
                   "stream_sha256", "online_seed", "seed"}
RNG_DOMAINS = ("rng", "eval_rng", "replay_rng", "projector_rng")


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone(v) for v in value)
    return deepcopy(value)


def state_hash(value):
    h = hashlib.sha256()
    def visit(v):
        h.update(type(v).__name__.encode()+b":")
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().contiguous()
            h.update(str((v.dtype, tuple(v.shape))).encode())
            h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(str((v.dtype, v.shape)).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            for key in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                visit(key); visit(v[key])
        elif isinstance(v, (list, tuple)):
            for item in v: visit(item)
        elif v is None or isinstance(v, (str, bool, int, float, np.generic)):
            h.update(repr(v).encode())
        else:
            raise ValueError("Unsupported checkpoint value: "+type(v).__name__)
        h.update(b";")
    visit(value)
    return h.hexdigest()


def official_rng():
    return clone(dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                      cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []))


def restore_rng(value):
    if set(value) != {"python", "numpy", "cpu", "cuda"}: raise ValueError("Invalid RNG schema")
    random.setstate(value["python"]); np.random.set_state(value["numpy"]); torch.set_rng_state(value["cpu"])
    if value["cuda"]:
        if not torch.cuda.is_initialized() or len(value["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG topology mismatch")
        torch.cuda.set_rng_state_all(value["cuda"])


def runtime_identity():
    return dict(cpu_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                thread_environment={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")},
                python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda,
                devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_initialized() else [],
                cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark)


def module_state(module):
    return dict(weights=clone(module.state_dict()),
                training={n: m.training for n,m in module.named_modules()},
                requires_grad={n:p.requires_grad for n,p in module.named_parameters()})


def restore_module(module, state):
    if set(state) != {"weights", "training", "requires_grad"}: raise ValueError("Missing module state")
    expected = module.state_dict()
    if set(expected) != set(state["weights"]): raise ValueError("Model state keys mismatch")
    for name,tensor in state["weights"].items():
        if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
            raise ValueError("Model state shape/dtype mismatch: "+name)
        if tensor.is_floating_point() and not torch.isfinite(tensor).all(): raise ValueError("Nonfinite model state")
    if set(state["training"]) != dict(module.named_modules()).keys() or set(state["requires_grad"]) != dict(module.named_parameters()).keys():
        raise ValueError("Module flags mismatch")
    module.load_state_dict(state["weights"], strict=True)
    for n,m in module.named_modules(): m.training=state["training"][n]
    for n,p in module.named_parameters(): p.requires_grad_(state["requires_grad"][n])


def engine_state(engine):
    if not getattr(engine, "lifecycle_mode", False) or engine.phase != "BOUNDARY" or engine.step != 15:
        raise ValueError("Only completed batch-boundary checkpoints supported; mid-batch resume is forbidden")
    return clone(dict(config=asdict(engine.config), context=engine.hssk_context, hssk_seed=engine.hssk_seed,
        provider=engine.provider.state_dict(), replay=engine.replay_manager.state_dict() if engine.replay_manager else None,
        snapshot=engine.snapshot.state_dict() if engine.snapshot else None,
        result=dict(status=engine.build_result.status.value, reason=engine.build_result.reason),
        batch=engine.batch, step=engine.step, phase=engine.phase, counters=dict(engine.counters),
        completed_updates=engine.completed_updates, domains={n:vars(getattr(engine,n)) for n in RNG_DOMAINS},
        projector=module_state(engine.projector), optimizer_groups=engine.optimizer_groups,
        event_sequence=engine.event_sequence))


def restore_engine(engine, state):
    if set(state) != {"config","context","hssk_seed","provider","replay","snapshot","result","batch","step","phase",
                      "counters","completed_updates","domains","projector","optimizer_groups","event_sequence"}:
        raise ValueError("Missing/unknown engine state")
    if state["config"] != asdict(engine.config) or state["context"] != engine.hssk_context or state["hssk_seed"] != engine.hssk_seed:
        raise ValueError("Engine config/lineage/seed mismatch")
    if any(type(state[k]) is not int or state[k] < 0 for k in ("batch","step","completed_updates","event_sequence")):
        raise ValueError("Invalid engine checkpoint counters")
    if any(type(v) is not int or v < 0 for v in state["counters"].values()):
        raise ValueError("Invalid engine counters")
    if state["phase"] != "BOUNDARY" or state["step"] != 15 or state["completed_updates"] != (state["batch"]+1)*15:
        raise ValueError("Unsupported mid-batch/freeze position")
    if set(state["domains"]) != set(RNG_DOMAINS): raise ValueError("Missing private RNG domains")
    provider=deepcopy(engine.provider); provider.load_state_dict(state["provider"])
    replay=deepcopy(engine.replay_manager)
    if (replay is None) != (state["replay"] is None): raise ValueError("Replay availability mismatch")
    if replay: replay.load_state_dict(state["replay"])
    snapshot=HierarchySnapshot.from_state_dict(state["snapshot"]) if state["snapshot"] else None
    status=HierarchyBuildStatus(state["result"]["status"])
    if (snapshot is None) != (status == HierarchyBuildStatus.SKIP): raise ValueError("Snapshot/status mismatch")
    if provider.implementation.last_batch != state["batch"] or provider.implementation.last_status != status.value:
        raise ValueError("Provider/freeze position mismatch")
    if snapshot is not None:
        if snapshot.config_version != digest(asdict(engine.config)) or snapshot.diagnostics["status"] != status.value:
            raise ValueError("Frozen snapshot/config/status mismatch")
        if status == HierarchyBuildStatus.BUILD and snapshot.snapshot_hash != provider.implementation.previous.snapshot_hash:
            raise ValueError("Published BUILD snapshot mismatch")
        if status == HierarchyBuildStatus.REUSE and snapshot.diagnostics.get("previous_snapshot_hash") != provider.implementation.previous.snapshot_hash:
            raise ValueError("REUSE source snapshot mismatch")
    if replay is not None and (replay._last_batch != state["batch"] or replay.selection is not None):
        raise ValueError("Replay cursor/freeze position mismatch")
    for name in RNG_DOMAINS:
        if set(state["domains"][name]) != set(vars(getattr(engine,name))) or state["domains"][name]["seed"] != getattr(engine,name).seed:
            raise ValueError("Private RNG schema/seed mismatch")
    restore_module(engine.projector,state["projector"])
    engine.provider.load_state_dict(state["provider"])
    if engine.replay_manager: engine.replay_manager.load_state_dict(state["replay"])
    engine.hierarchy=engine.provider.memory
    engine.snapshot=snapshot
    engine.build_result=HierarchyBuildResult(HierarchyProviderKind.HSSK_ONLINE,status,snapshot,state["result"]["reason"])
    for name in ("batch","step","phase","completed_updates","event_sequence","optimizer_groups"):
        setattr(engine,name,clone(state[name]))
    engine.counters=Counter(state["counters"])
    for name in RNG_DOMAINS:
        if set(state["domains"][name]) != set(vars(getattr(engine,name))): raise ValueError("Private RNG schema mismatch")
        getattr(engine,name).__dict__.update(clone(state["domains"][name]))
    engine.images=engine.selected_replay_images=None


def parameter_catalog(modules):
    return {owner+"."+name:p for owner,module in modules.items() for name,p in module.named_parameters()}


def optimizer_state(optimizer, modules):
    names={id(p):n for n,p in parameter_catalog(modules).items()}
    return dict(state=clone(optimizer.state_dict()),
                groups=[[names[id(p)] for p in group["params"]] for group in optimizer.param_groups])


def restore_optimizer(state, modules):
    if set(state) != {"state","groups"}: raise ValueError("Missing optimizer state/groups")
    catalog=parameter_catalog(modules)
    flat=[n for group in state["groups"] for n in group]
    if len(flat)!=len(set(flat)) or any(n not in catalog for n in flat): raise ValueError("Optimizer identity mismatch")
    groups=[dict(params=[catalog[n] for n in group]) for group in state["groups"]]
    optimizer=torch.optim.AdamW(groups)
    optimizer.load_state_dict(state["state"])
    assert_optimizer_order(optimizer,modules["projector"],modules["classifier"])
    for param,values in optimizer.state.items():
        for key,value in values.items():
            if isinstance(value,torch.Tensor) and (not torch.isfinite(value).all() or (key != "step" and value.shape != param.shape)):
                raise ValueError("Invalid optimizer moments")
    return optimizer


def validate_identity(identity):
    if set(identity)!=IDENTITY_FIELDS: raise ValueError("Missing checkpoint lineage fields")
    for key,value in identity.items():
        if key in ("seed","online_seed"):
            if type(value) is not int: raise ValueError("Invalid lineage seed")
        elif not isinstance(value,str) or len(value) not in (40,64) or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Invalid lineage hash: "+key)


def save_checkpoint(path, *, identity, engine, modules, optimizer, discovery, position, journal):
    validate_identity(identity)
    if set(position)!={"session","next_batch"} or position["session"]!=0 or position["next_batch"]!=engine.batch+1:
        raise ValueError("Invalid checkpoint stream position")
    payload=dict(schema=SCHEMA,identity=identity,runtime=runtime_identity(),engine=engine_state(engine),
                 modules={k:module_state(v) for k,v in modules.items()}, optimizer=optimizer_state(optimizer,modules),
                 discovery=clone(discovery),position=position,official_rng=official_rng(),journal=journal)
    envelope=dict(payload=payload,state_hash=state_hash(payload))
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=".state-",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as handle:
            torch.save(envelope,handle); handle.flush(); os.fsync(handle.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)
    return dict(path=str(path.resolve()),sha256=sha256(path),state_hash=envelope["state_hash"])


def read_checkpoint(path, identity):
    validate_identity(identity)
    value=torch.load(path,map_location="cpu",weights_only=False)
    if not isinstance(value,dict) or set(value)!={"payload","state_hash"} or state_hash(value["payload"])!=value["state_hash"]:
        raise ValueError("Corrupt checkpoint envelope/hash")
    p=value["payload"]
    if set(p)!={"schema","identity","runtime","engine","modules","optimizer","discovery","position","official_rng","journal"}:
        raise ValueError("Missing required checkpoint state")
    if p["schema"]!=SCHEMA or p["identity"]!=identity: raise ValueError("Checkpoint schema/lineage mismatch")
    if p["runtime"]!=runtime_identity(): raise ValueError("Resume requires the same deterministic runtime/backend")
    if p["position"]!={"session":0,"next_batch":p["engine"]["batch"]+1}: raise ValueError("Stream position mismatch")
    if p["engine"]["phase"]!="BOUNDARY" or p["engine"]["step"]!=15: raise ValueError("Mid-batch resume unsupported")
    return p
