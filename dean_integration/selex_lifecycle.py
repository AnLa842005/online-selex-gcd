"""Explicit full/bounded lifecycle over the pinned DEAN incoming loop.

Only batch-boundary checkpoints are supported. The one-batch SmokeBudget is
unchanged; the separate lifecycle smoke completes two full 15-update windows.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import torch
from .preflight import sha256
from .selex_online_state import (engine_state, restore_engine, save_checkpoint, read_checkpoint,
                                restore_module, restore_optimizer, restore_rng, state_hash)

DISCOVERY_FIELDS = ("nb_classes_prv", "nb_classes_now", "nb_classes_old", "nb_classes_k_online",
                    "online_seen_classes", "score_summary", "list_old_inc", "energy_store")


class LifecycleSmokeComplete(Exception):
    pass


class OnlineLifecycle:
    def __init__(self, engine, paired, identity, output, *, smoke=False, resume=None):
        if not engine.lifecycle_mode or not engine.is_hssk:
            raise ValueError("Lifecycle requires active HSSK")
        self.engine,self.paired,self.identity,self.output=engine,paired,dict(identity),Path(output)
        self.smoke,self.resume=smoke,Path(resume) if resume else None
        self.started,self.completed,self.start_batch=0,0,0
        self.active=None
        self.checkpoint=None

    def modules(self, ns):
        return dict(student=ns["model_now"],classifier=ns["header_now"],
                    teacher=ns["model"],teacher_classifier=ns["header"],projector=self.engine.projector)

    def session_ready(self, ns):
        if ns["sess_idx"] != 0 or ns["args"].inc_epochs != 15:
            raise ValueError("Production v1 requires one session and 15 epochs")
        self.identity["stream_sha256"]=sha256(self.paired.output)
        if not self.resume: return
        p=read_checkpoint(self.resume,self.identity)
        if set(p["modules"]) != set(self.modules(ns)) or set(p["discovery"]) != set(DISCOVERY_FIELDS):
            raise ValueError("Missing DEAN model/discovery state")
        # Allocate classifier shape before strict restore; setup RNG is discarded
        # by restoring the saved official RNG only after all objects are loaded.
        count=p["modules"]["classifier"]["weights"]["head.weight"].shape[0]
        ns["header_now"].update(nb_classes_now=count)
        ns["header_now"].to(self.engine.device)
        for name,module in self.modules(ns).items(): restore_module(module,p["modules"][name])
        restore_engine(self.engine,p["engine"])
        ns["opt"]=restore_optimizer(p["optimizer"],self.modules(ns))
        ns.update(deepcopy(p["discovery"]))
        self.start_batch=p["position"]["next_batch"]
        self.completed=self.start_batch
        self.checkpoint=dict(path=str(self.resume.resolve()),sha256=sha256(self.resume),state_hash=state_hash(p))
        journal=p["journal"]
        if set(journal)!={"path","offset","sha256"}: raise ValueError("Missing journal checkpoint")
        adjacent=self.resume.parent/"selex_reference_metrics.jsonl"
        journal_path=adjacent if adjacent.is_file() else Path(journal["path"])
        with journal_path.open("rb") as handle: prefix=handle.read(journal["offset"])
        if len(prefix)!=journal["offset"] or hashlib.sha256(prefix).hexdigest()!=journal["sha256"]:
            raise ValueError("Resume journal prefix hash mismatch")
        records=[json.loads(line) for line in prefix.splitlines()]
        if [r["event_id"] for r in records]!=list(range(self.engine.event_sequence)):
            raise ValueError("Resume journal sequence mismatch")
        self.engine.telemetry.write_bytes(prefix)
        self.engine._event("resume", checkpoint_sha256=sha256(self.resume), next_batch=self.start_batch)
        restore_rng(p["official_rng"])

    def stream(self, dataset, session, batch_size, workers):
        return enumerate(self.paired.loader(dataset,session,batch_size,workers,start_batch=self.start_batch),start=self.start_batch)

    def batch_started(self, session, batch):
        if self.active is not None or batch != self.completed or session != 0:
            raise ValueError("Invalid lifecycle batch boundary")
        if self.smoke and self.started >= 2:
            raise ValueError("Lifecycle smoke attempted a third incoming batch")
        self.active=(session,batch)
        self.started+=1
        self.updates=0
        self.update_tick=None
        if self.engine.device.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        self.tick=time.perf_counter()
        self.engine._event("incoming_batch",session=session,source_batch=batch,incoming_batches_started=self.started)

    def update_started(self):
        if self.active is None or self.updates >= 15 or self.update_tick is not None or self.engine.phase != "FROZEN":
            raise ValueError("Lifecycle forward outside frozen update window")
        if self.engine.device.type == "cuda": torch.cuda.synchronize()
        self.update_tick=time.perf_counter()

    def step_completed(self):
        e=self.engine
        if self.active is None or self.update_tick is None or self.updates>=15 or not hasattr(e,"pending_update") or e.step != self.updates+1:
            raise ValueError("Lifecycle optimizer update outside frozen batch")
        if e.frozen_hash != (e.snapshot.snapshot_hash if e.snapshot else None):
            raise ValueError("Frozen hierarchy changed")
        from .selex_smoke_gate import SmokeBudget
        projector=SmokeBudget._gradients(p for p in e.projector.parameters() if p.requires_grad)
        lora=SmokeBudget._gradients(p for n,p in e.smoke_model.named_parameters() if p.requires_grad and "lora" in n.lower())
        if any(g["gradient_tensors"] and not g["finite"] for g in (projector,lora)):
            raise ValueError("Nonfinite lifecycle gradients")
        if e.device.type == "cuda": torch.cuda.synchronize()
        peaks=dict(cuda_peak_allocated=torch.cuda.max_memory_allocated() if e.device.type == "cuda" else None,
                   cuda_peak_reserved=torch.cuda.max_memory_reserved() if e.device.type == "cuda" else None)
        wall=time.perf_counter()-self.update_tick
        self.update_tick=None
        self.updates+=1; e.completed_updates+=1
        e.counters["hssk.frozen_reuse_updates"] += 1
        event=dict(e.pending_update, hierarchy_usage="FROZEN_REUSE", counters=dict(e.counters),
                   gradients=dict(projector=projector,lora=lora),wall_seconds=wall,**peaks); del e.pending_update
        e._event("update",**event)

    def batch_completed(self, ns):
        e=self.engine
        if self.active is None or self.updates!=15 or e.step!=15:
            raise ValueError("Incoming batch ended without exactly 15 committed optimizer updates")
        ns["opt"].zero_grad(set_to_none=True)
        e.phase="BOUNDARY"
        if e.replay_manager: e.replay_manager.finish_batch(e.batch)
        e.images=e.selected_replay_images=None
        self.completed+=1
        e._event("batch_complete",session=self.active[0],source_batch=self.active[1],
                 updates=self.updates,wall_seconds=time.perf_counter()-self.tick,
                 snapshot_hash=e.snapshot.snapshot_hash if e.snapshot else None)
        raw=e.telemetry.read_bytes()
        journal=dict(path=str(e.telemetry.resolve()),offset=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        self.checkpoint=save_checkpoint(self.output/"online_state.pth",identity=self.identity,engine=e,
            modules=self.modules(ns),optimizer=ns["opt"],discovery={k:ns[k] for k in DISCOVERY_FIELDS},
            position=dict(session=0,next_batch=self.completed),journal=journal)
        self.active=None
        if self.smoke and self.started==2:
            self.complete()
            raise LifecycleSmokeComplete("Two incoming batches and 30 updates completed")

    def failed(self, error):
        from .provenance import write_json
        record=dict(exit_code=1,error_type=type(error).__name__,error=str(error),
                    oom=isinstance(error,torch.cuda.OutOfMemoryError),
                    incoming_batches_started=self.started,completed_batches=self.completed,
                    cuda_peak_allocated=None,cuda_peak_reserved=None)
        if getattr(self.engine,"device",torch.device("cpu")).type == "cuda" and torch.cuda.is_initialized():
            try:
                record.update(cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                              cuda_peak_reserved=torch.cuda.max_memory_reserved())
            except Exception:
                pass
        write_json(self.output/"lifecycle_failure.json",record)
        self.engine._event("failed",**record)

    def complete(self):
        if self.active is not None or self.checkpoint is None or (self.started == 0 and not self.resume):
            raise ValueError("Lifecycle exited without complete batches")
        if self.smoke and self.started!=2:
            raise ValueError("Lifecycle smoke did not finish two batches")
        self.engine._event("complete",exit_code=0,checkpoint=self.checkpoint,
                           incoming_batches_started=self.started,completed_batches=self.completed)
