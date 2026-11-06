"""Explicit one-batch, 1..2-update smoke termination and read-only diagnostics."""
import json
import time
import torch


class BoundedSmokeComplete(Exception):
    pass


class SmokeBatchLimitExceeded(RuntimeError):
    pass


class SmokeBudget:
    def __init__(self, updates):
        if type(updates) is not int or not 1 <= updates <= 2:
            raise ValueError("Bounded online smoke requires 1..2 optimizer updates")
        self.limit, self.steps = updates, 0
        self.max_incoming_batches = 1
        self.incoming_batches_started = 0
        self.source_batches = []
        self.batch_limit_failure_reason = None
        self.engine, self.output, self.metadata = None, None, {}
        self.updates, self.hierarchy = [], None
        self.tick, self.hierarchy_tick = None, None
        self.peak_allocated, self.peak_reserved = 0, 0

    def batch_started(self, session=None, batch=None):
        if self.incoming_batches_started >= self.max_incoming_batches:
            self.batch_limit_failure_reason = "Attempted second incoming batch before smoke completion"
            self._record("batch_limit_exceeded", attempted_session=session, attempted_batch=batch,
                         exit_code=1, **self.batch_summary())
            raise SmokeBatchLimitExceeded(self.batch_limit_failure_reason)
        self.incoming_batches_started += 1
        self.source_batches.append(dict(session=session, batch=batch))
        self._record("batch_started", **self.batch_summary())

    def batch_summary(self):
        return dict(incoming_batches_started=self.incoming_batches_started,
                    source_batches=list(self.source_batches),
                    requested_optimizer_updates=self.limit, completed_optimizer_updates=self.steps,
                    batch_limit_failure_reason=self.batch_limit_failure_reason)

    def require_active_batch(self):
        if self.incoming_batches_started != 1 or self.batch_limit_failure_reason:
            raise SmokeBatchLimitExceeded("Smoke update requires exactly one admitted incoming batch")

    def bind_engine(self, engine, metadata):
        self.engine, self.metadata = engine, dict(metadata)
        engine.smoke_budget = self
        if engine.output:
            self.output = engine.output/"smoke_diagnostics.jsonl"
            with self.output.open("x", encoding="utf-8"):
                pass

    def _cuda(self, reset=False):
        # No CUDA initialization/query on CPU fixtures or failed pre-CUDA setup.
        device = getattr(self.engine, "device", None)
        if self.engine is None or (device is not None and device.type != "cuda") or not torch.cuda.is_initialized():
            return dict(cuda_peak_allocated=None, cuda_peak_reserved=None)
        torch.cuda.synchronize()
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return dict(cuda_peak_allocated=allocated, cuda_peak_reserved=reserved)

    def _record(self, event, **values):
        record = dict(self.metadata, event=event, **values)
        if self.output:
            with self.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False)+"\n")
        return record

    def hierarchy_started(self):
        self._cuda(reset=True)
        self.hierarchy_tick = time.perf_counter()

    def hierarchy_completed(self):
        cuda = self._cuda()
        engine = self.engine
        self.hierarchy = self._record("hierarchy", wall_seconds=time.perf_counter()-self.hierarchy_tick,
            **cuda, status=engine.build_result.status.value, reason=engine.build_result.reason,
            snapshot_hash=engine.snapshot.snapshot_hash if engine.snapshot else None,
            memory_hash=engine.provider.memory.state_hash,
            diagnostics=engine.snapshot.diagnostics if engine.snapshot else {},
            replay_ids=list(engine.replay_manager.selection.sample_ids) if engine.replay_manager else [],
            replay_selection_hash=engine.replay_manager.selection.selection_hash if engine.replay_manager else None)

    def update_started(self, scope="full_update"):
        self.require_active_batch()
        if self.steps >= self.limit:
            raise ValueError("Smoke budget exhausted before another optimizer update")
        self._cuda(reset=True)
        self.tick, self.timing_scope = time.perf_counter(), scope

    @staticmethod
    def _gradients(parameters):
        parameters = list(parameters)
        grads = [p.grad for p in parameters if p.grad is not None]
        return dict(parameter_count=sum(p.numel() for p in parameters),
                    gradient_tensors=len(grads),
                    finite=bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads),
                    nonzero=bool(grads) and any(bool(torch.count_nonzero(g)) for g in grads))

    def failed(self, error):
        try:
            cuda = self._cuda()
        except Exception:
            cuda = dict(cuda_peak_allocated=self.peak_allocated, cuda_peak_reserved=self.peak_reserved)
        self._record("failed", exit_code=1, error_type=type(error).__name__, error=str(error),
                     oom=isinstance(error, torch.cuda.OutOfMemoryError), **self.batch_summary(), **cuda)

    def summary(self):
        return dict(hierarchy=self.hierarchy, updates=self.updates, cuda_peak_allocated=self.peak_allocated,
                    cuda_peak_reserved=self.peak_reserved, **self.batch_summary())

    def step_completed(self):
        self.require_active_batch()
        if self.engine is not None:
            cuda = self._cuda()
            elapsed = time.perf_counter()-self.tick if self.tick is not None else None
            projector = self._gradients(p for p in self.engine.projector.parameters() if p.requires_grad)
            lora = self._gradients(p for n, p in self.engine.smoke_model.named_parameters() if p.requires_grad and "lora" in n.lower())
            self.updates.append(self._record("optimizer_update", update=self.steps+1, wall_seconds=elapsed,
                timing_scope=getattr(self, "timing_scope", "not_started"), gradients=dict(projector=projector, lora=lora),
                losses=self.engine.smoke_losses, **cuda))
            if not all(g["finite"] and g["nonzero"] for g in (projector, lora)):
                raise ValueError("Smoke requires finite nonzero gradients for projector and LoRA")
            self.tick = None
        self.steps += 1
        if self.steps >= self.limit:
            self._record("complete", exit_code=0, updates=self.steps, **self.batch_summary(), **self._cuda())
            raise BoundedSmokeComplete("Bounded smoke update budget completed")
