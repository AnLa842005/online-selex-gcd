"""Runtime incoming-batch guard regressions against the pinned upstream AST."""
import ast
from pathlib import Path
import unittest
from unittest.mock import Mock

from dean_integration.runtime import patch_bounded_smoke, patch_experiment
from dean_integration.selex_smoke_gate import SmokeBudget, SmokeBatchLimitExceeded, BoundedSmokeComplete
from dean_integration.selex_hierarchy_provider import ensure_provider_ready, FullTrainingBlocked
from tests.test_dean_hssk_phase3 import cfg


class SmokeBatchGuardTests(unittest.TestCase):
    def tree(self):
        path = Path(__file__).resolve().parents[2]/"OCGCD-dean-audit/Ours_train_online.py"
        return ast.parse(path.read_text(encoding="utf-8"))

    def loop(self, tree):
        return next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                    and ast.unparse(n.target) == "(batch_idx, (on_x, on_y, on_z))")

    def test_two_updates_complete_inside_first_batch(self):
        budget = SmokeBudget(2)
        budget.batch_started(0, 0)
        budget.step_completed()
        with self.assertRaises(BoundedSmokeComplete): budget.step_completed()
        self.assertEqual(budget.summary()["incoming_batches_started"], 1)
        self.assertEqual(budget.summary()["completed_optimizer_updates"], 2)

    def exercise_second_batch(self, steps, status):
        tree = patch_bounded_smoke(self.tree())
        loop = self.loop(tree)
        # Execute the actual inserted AST guard before mocked discovery/replay/work.
        guard = loop.body[0]
        loop.body = [guard] + ast.parse("work.hssk(status)\nwork.replay()\nwork.forward()\nwork.optimizer()\nfor update in range(steps):\n    _dean_smoke.step_completed()").body
        program = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
        work = Mock()
        budget = SmokeBudget(2)
        with self.assertRaises(SmokeBatchLimitExceeded):
            exec(compile(program, "<actual-smoke-loop>", "exec"), dict(
                _dean_smoke=budget, sess_idx=0, pbar=enumerate([(None,None,None)]*2),
                work=work, steps=steps, status=status))
        work.hssk.assert_called_once_with(status)
        work.replay.assert_called_once_with()
        work.forward.assert_called_once_with()
        work.optimizer.assert_called_once_with()
        self.assertEqual(budget.incoming_batches_started, 1)
        self.assertEqual(budget.steps, steps)
        self.assertTrue(budget.batch_limit_failure_reason)
        with self.assertRaises(SmokeBatchLimitExceeded): budget.step_completed()

    def test_one_update_then_second_batch_fails_before_work(self):
        self.exercise_second_batch(1, "BUILD")

    def test_zero_updates_then_second_batch_fails_before_work(self):
        self.exercise_second_batch(0, "BUILD")

    def test_build_reuse_skip_cannot_bypass_guard(self):
        for status in ("BUILD", "REUSE", "SKIP"):
            for steps in (0, 1):
                with self.subTest(status=status, steps=steps): self.exercise_second_batch(steps, status)

    def test_hook_is_first_statement_in_pinned_incoming_loop(self):
        tree = patch_bounded_smoke(self.tree())
        loop = self.loop(tree)
        self.assertEqual(loop.lineno, 446)
        self.assertEqual(ast.unparse(loop.body[0]), "_dean_smoke.batch_started(session=sess_idx, batch=batch_idx)")
        self.assertIsInstance(loop.body[1], ast.For)
        self.assertEqual(sum(isinstance(n, ast.Call) and ast.unparse(n.func) == "_dean_smoke.batch_started"
                             for n in ast.walk(tree)), 1)

    def test_normal_paths_have_no_smoke_batch_hook(self):
        for options in ({}, dict(paired_online=True), dict(paired_online=True, reference_selex=True)):
            tree = patch_experiment(self.tree(), **options)
            self.assertNotIn("_dean_smoke", ast.unparse(tree))

    def test_changed_stream_boundary_fails_closed(self):
        tree = self.tree()
        self.loop(tree).iter = ast.Name(id="other", ctx=ast.Load())
        with self.assertRaisesRegex(ValueError, "incoming batch loop"): patch_bounded_smoke(tree)

    def test_updates_without_batch_fail(self):
        budget = SmokeBudget(2)
        with self.assertRaises(SmokeBatchLimitExceeded): budget.step_completed()
        with self.assertRaises(SmokeBatchLimitExceeded): budget.update_started()

    def test_batch_failure_diagnostics(self):
        budget = SmokeBudget(2)
        budget._record = Mock()
        budget.batch_started(session=3, batch=4)
        budget.step_completed()
        with self.assertRaises(SmokeBatchLimitExceeded) as error: budget.batch_started(3, 5)
        budget.failed(error.exception)
        event = budget._record.call_args
        self.assertEqual(event.args[0], "failed")
        self.assertEqual(event.kwargs["exit_code"], 1)
        self.assertEqual(event.kwargs["incoming_batches_started"], 1)
        self.assertEqual(event.kwargs["source_batches"], [dict(session=3,batch=4)])
        self.assertEqual(event.kwargs["requested_optimizer_updates"], 2)
        self.assertEqual(event.kwargs["completed_optimizer_updates"], 1)
        self.assertTrue(event.kwargs["batch_limit_failure_reason"])

    def test_provenance_rejects_second_batch_or_latched_failure(self):
        import json
        import tempfile
        from types import SimpleNamespace
        from dean_integration.provenance import execute_run
        for count, reason in ((2, None), (1, "second batch attempted"), (0, None)):
            with tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                child = run/"seed1"
                budget = dict(incoming_batches_started=count, requested_optimizer_updates=2,
                              completed_optimizer_updates=2, batch_limit_failure_reason=reason)
                summary = dict(implementation="reference_v1", step=2, optimizer_groups=[{}],
                               last_event="bounded_smoke_complete", full_train_blocked=True,
                               hierarchy_state=dict(status="BUILD", memory_hash="m", snapshot_hash="s"),
                               smoke_diagnostics=budget)
                def runner(command, **kwargs):
                    (child/"selex_reference_summary.json").write_text(json.dumps(summary))
                    return SimpleNamespace(returncode=0)
                report = dict(real_selex_active=True, hierarchy_source="hssk_online",
                              bounded_smoke=dict(incoming_batches=1, optimizer_updates=2))
                with self.assertRaisesRegex(ValueError, "runtime batch budget"):
                    execute_run(run, report, [(1, child, ["fixture"])], runner)
                self.assertEqual(report["status"], "FAILED")

    def test_full_training_gate_still_closed(self):
        with self.assertRaises(FullTrainingBlocked): ensure_provider_ready(cfg())
        ensure_provider_ready(cfg(), bounded_smoke=True)


if __name__ == "__main__":
    unittest.main()
