"""Production-sized support and real CPU lifecycle/state kernels (no GPU)."""
import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dean_integration.preflight import probe_output
from dean_integration.selex_hierarchy_provider import OnlineHSSKHierarchyProvider, HierarchySupportPool
from dean_integration.selex_hssk_hierarchy import coarsen_children, weighted_fit
from dean_integration.selex_real_online import RealSelExOnline, CurrentBatchImages, hierarchy_aggregate, assert_optimizer_order
from dean_integration.selex_lifecycle import OnlineLifecycle, LifecycleSmokeComplete, DISCOVERY_FIELDS
from dean_integration.selex_online_state import (IDENTITY_FIELDS, official_rng, state_hash, read_checkpoint,
                                               save_checkpoint, engine_state, restore_rng)
from dean_integration.selex_reference import expertise_losses
from dean_integration.runtime import patch_experiment, patch_lifecycle
from tests.test_dean_hssk_online import basis, replay_features
from tests.test_dean_hssk_phase3 import cfg, context, Student


def Header(count=2):
    """Pinned native DEAN head class, without importing unrelated CUDA utilities."""
    path=Path(__file__).resolve().parents[2]/"OCGCD-dean-audit/net/Decoder.py"
    tree=ast.parse(path.read_text(encoding="utf-8"))
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="Head")
    scope=dict(torch=torch,nn=nn)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),"exec"),scope)
    return scope["Head"](768,count)


def identity():
    return {k:17 if k=="online_seed" else 1 if k=="seed" else "a"*(40 if k.endswith("git_sha") or k=="upstream_sha" else 64)
            for k in IDENTITY_FIELDS}


def fixture(output, resume=None, smoke=False):
    torch.manual_seed(7)
    output.mkdir()
    student=Student(); head=Header()
    _, replay, _=replay_features(16)
    replay.finish_batch(0)
    # Recreate a fresh admitted manager: helper selected once for feature fixtures.
    from dean_integration.selex_labeled_replay import VerifiedLabeledReplay
    fresh=VerifiedLabeledReplay(replay.eligibility)
    fresh.admit(replay.eligibility.entry(k) for k in sorted(replay.eligibility._records))
    e=RealSelExOnline(cfg(),17,output,lifecycle_mode=True,replay_manager=fresh,
                      context={k:v for k,v in context().items() if k not in ("batch","seed")})
    e.replay_image_factory=lambda ids:CurrentBatchImages(ids,lambda key:basis(1,(key-10000)//10)[0],
                                                       lambda x:x+torch.rand_like(x)*.001,lambda x:x)
    e.start_session(student)
    opt=torch.optim.AdamW(student.parameters(),lr=1e-4)
    opt.add_param_group(dict(params=head.parameters(),lr=1e-4));e.attach_optimizer(opt,head)
    stream=output/"stream_manifest.json";stream.write_text('{"fixture": "same-stream"}')
    ident=identity(); ident["stream_sha256"]=hashlib.sha256(stream.read_bytes()).hexdigest()
    controller=OnlineLifecycle(e,SimpleNamespace(output=stream),ident,output,smoke=smoke,resume=resume)
    ns=dict(model_now=student,header_now=head,model=deepcopy(student),header=deepcopy(head),opt=opt,
            args=SimpleNamespace(inc_epochs=15),sess_idx=0,nb_classes_prv=2,nb_classes_now=2,nb_classes_old=2,
            nb_classes_k_online=0,online_seen_classes=[],score_summary={},list_old_inc=np.array([]),energy_store=np.array([]))
    controller.session_ready(ns)
    return controller,ns


def expand_native(controller,ns):
    """Execute pinned DEAN classifier AdamW extension with CUDA calls mapped to CPU."""
    checkout=Path(__file__).resolve().parents[2]/"OCGCD-dean-audit"
    tree=ast.parse((checkout/"Ours_train_online.py").read_text(encoding="utf-8"))
    block=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="batch_idx > 0"
               and any(isinstance(x,ast.Assign) and ast.unparse(x).startswith("opt_dict =") for x in n.body))
    class CPU(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            return node.func.value if isinstance(node.func,ast.Attribute) and node.func.attr=="cuda" else node
        def visit_Constant(self,node):
            return ast.copy_location(ast.Constant("cpu"),node) if node.value=="cuda" else node
    block=CPU().visit(deepcopy(block))
    ns["header_now"].update(3)
    env=dict(ns,torch=torch,copy=__import__("copy"),batch_idx=1,nb_classes_k=1,args=SimpleNamespace(lr=1e-4))
    exec(compile(ast.fix_missing_locations(ast.Module(body=block.body,type_ignores=[])),"<pinned-expansion>","exec"),env)
    controller.engine.attach_optimizer(ns["opt"],ns["header_now"])
    ns["nb_classes_now"]=3;ns["nb_classes_k_online"]=1


def batch(controller,ns,index,free):
    e=controller.engine
    controller.batch_started(0,index)
    if index: expand_native(controller,ns)
    ids=torch.arange(64)+index*100
    images=CurrentBatchImages(ids,lambda key:basis(1,32+(key-index*100)%free)[0],
                              lambda x:x+torch.rand_like(x)*.001,lambda x:x)
    e.prepare(ns["model_now"],images,None,ns["nb_classes_now"])
    snapshot=e.snapshot.snapshot_hash
    fixed=images.images(ids,two_views=False)
    losses=[]
    for epoch in range(15):
        controller.update_started()
        logits=ns["header_now"](ns["model_now"](fixed))
        ce=F.cross_entropy(logits,torch.arange(64)%ns["nb_classes_now"])
        ec=ce*0  # Existing official CE path with EC disabled for this CPU fixture.
        loss=e.augment(ce+ec,ce,ec,ns["model_now"],None,ids)
        ns["opt"].zero_grad();loss.backward();ns["opt"].step();controller.step_completed()
        assert e.snapshot.snapshot_hash==snapshot
        losses.append(float(loss.detach()))
    controller.batch_completed(ns)
    return losses,snapshot


class Phase4Tests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_production_support_odd_free_is_leaf_only_finite_and_no_partial_publish(self):
        replay,_,_=replay_features(16)
        rows=basis(3,32)[torch.arange(64)%3]
        p=OnlineHSSKHierarchyProvider(cfg())
        with patch("dean_integration.selex_hssk_provider.coarsen_children",wraps=coarsen_children) as spy:
            result=p.build(HierarchySupportPool(range(64),rows),labeled_replay=replay,context=context())
        spy.assert_called_once()
        snap=result.snapshot
        self.assertEqual(snap.diagnostics["k_free"],3)
        self.assertEqual(snap.diagnostics["child_maps"],[])
        self.assertEqual(snap.diagnostics["higher_stop"],dict(status="LEAF_ONLY",reason="insufficient_free_child_support",
            child_count=3,requested_parent_count=2,minimum_children_per_parent=2,required_child_count=4))
        self.assertEqual(p.counters["hssk.higher.stop.insufficient_free_child_support"],1)
        q=torch.randn(64,2,16,requires_grad=True)
        losses=expertise_losses(q,rows[:,None,:].expand(-1,2,-1),torch.arange(64),snap,cfg())
        losses["selex_total"].backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertEqual(len(p.memory.tracks),19)

    def test_supported_four_and_six_free_produce_real_nested_deterministic_levels(self):
        for free in (4,6):
            replay,_,_=replay_features(16)
            rows=basis(free,32)[torch.arange(64)%free]
            snaps=[]
            for _ in range(2):
                p=OnlineHSSKHierarchyProvider(cfg())
                snaps.append(p.build(HierarchySupportPool(range(64),rows),labeled_replay=replay,context=context()).snapshot)
            a,b=snaps
            self.assertEqual(a.snapshot_hash,b.snapshot_hash)
            self.assertEqual(a.diagnostics["k_free"],free)
            self.assertGreater(len(a.mappings),1);self.assertTrue(a.diagnostics["child_maps"])
            used=set()
            for level in a.centers:
                self.assertFalse(used.intersection(level));used.update(level)
            for i,mapping in enumerate(a.diagnostics["child_maps"]):
                self.assertEqual(set(map(int,mapping)),set(a.centers[i]))
            self.assertEqual(hierarchy_aggregate(p.counters)["higher_build_fraction"],1)

    def test_weighted_coarsening_means(self):
        rows=basis(4); weights=torch.tensor([2.,4.,6.,8.])
        labels,centers=weighted_fit(rows,weights,torch.full((4,),-1),2,17)
        for k in range(2):
            mask=labels==k
            self.assertTrue(torch.equal(centers[k],(rows[mask]*weights[mask,None]).sum(0)/weights[mask].sum()))

    def test_two_batch_lifecycle_smoke_real_kernels_and_hard_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,ns=fixture(Path(tmp)/"run",smoke=True)
            batch(c,ns,0,3)
            first=c.engine.snapshot.snapshot_hash
            self.assertEqual(c.engine.snapshot.diagnostics["hierarchy_type"],"LEAF_ONLY")
            with self.assertRaises(LifecycleSmokeComplete): batch(c,ns,1,4)
            self.assertNotEqual(first,c.engine.snapshot.snapshot_hash)
            self.assertEqual(c.engine.snapshot.diagnostics["hierarchy_type"],"MULTI_LEVEL")
            self.assertEqual(c.engine.completed_updates,30)
            with self.assertRaises(ValueError): c.batch_started(0,2)
            records=[json.loads(x) for x in c.engine.telemetry.read_text().splitlines()]
            self.assertEqual(sum(r["event"]=="update" for r in records),30)
            self.assertEqual(sum(r["event"]=="hierarchy" for r in records),2)
            self.assertEqual(records[-1]["event"],"complete")
            self.assertEqual([r["event_id"] for r in records],list(range(len(records))))

    def test_complete_boundary_resume_matches_losses_models_optimizer_rng_and_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            c,ns=fixture(root/"continuous")
            batch(c,ns,0,3)
            checkpoint=root/"frozen.pth"
            checkpoint.write_bytes((c.output/"online_state.pth").read_bytes())
            expected_losses,expected_hash=batch(c,ns,1,4)
            expected=engine_state(c.engine)
            expected_rng=official_rng()
            expected_params=state_hash({k:m.state_dict() for k,m in c.modules(ns).items()})
            expected_opt=state_hash(ns["opt"].state_dict())
            restored,rns=fixture(root/"restored",resume=checkpoint)
            self.assertEqual(restored.engine.snapshot.diagnostics["hierarchy_type"],"LEAF_ONLY")
            actual_losses,actual_hash=batch(restored,rns,1,4)
            actual=engine_state(restored.engine)
            self.assertEqual(actual_losses,expected_losses);self.assertEqual(actual_hash,expected_hash)
            self.assertEqual(state_hash(official_rng()),state_hash(expected_rng))
            self.assertEqual(state_hash({k:m.state_dict() for k,m in restored.modules(rns).items()}),expected_params)
            self.assertEqual(state_hash(rns["opt"].state_dict()),expected_opt)
            for key in ("provider","replay","snapshot","counters","domains","completed_updates"):
                self.assertEqual(state_hash(actual[key]),state_hash(expected[key]),key)
            records=[json.loads(x) for x in restored.engine.telemetry.read_text().splitlines()]
            updates=[(r["batch"],r["step"]) for r in records if r["event"]=="update"]
            self.assertEqual(len(updates),len(set(updates)))
            self.assertEqual(len(updates),30)
            assert_optimizer_order(rns["opt"],restored.engine.projector,rns["header_now"])
            expanded,ens=fixture(root/"expanded_restore",resume=restored.output/"online_state.pth")
            self.assertEqual(ens["header_now"].head.weight.shape[0],3)
            self.assertEqual(expanded.engine.snapshot.snapshot_hash,expected_hash)
            self.assertEqual(expanded.engine.snapshot.diagnostics["hierarchy_type"],"MULTI_LEVEL")
            self.assertEqual(state_hash(ens["opt"].state_dict()),expected_opt)
            self.assertEqual(state_hash({k:m.state_dict() for k,m in expanded.modules(ens).items()}),expected_params)
            self.assertEqual(state_hash(official_rng()),state_hash(expected_rng))
            expanded.complete()  # Exhausted-stream resume can finish without fabricating another batch.


    def test_strict_lineage_schema_missing_state_and_mid_batch_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,ns=fixture(Path(tmp)/"run");batch(c,ns,0,3)
            path=c.output/"online_state.pth"
            for key in IDENTITY_FIELDS:
                bad=dict(c.identity);bad[key]=2 if key in ("seed","online_seed") else "b"*len(bad[key])
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,"lineage"):
                    read_checkpoint(path,bad)
            original=torch.load(path,weights_only=False)
            for mode in ("schema","missing","mid_batch","position"):
                value=deepcopy(original)
                if mode=="schema":value["payload"]["schema"]="bad"
                if mode=="missing":value["payload"].pop("optimizer")
                if mode=="mid_batch":value["payload"]["engine"]["phase"]="FROZEN"
                if mode=="position":value["payload"]["position"]["next_batch"]=20
                value["state_hash"]=state_hash(value["payload"])
                target=c.output/(mode+".pth");torch.save(value,target)
                with self.subTest(mode=mode),self.assertRaises(ValueError):read_checkpoint(target,c.identity)
            # Failure before atomic publication must leave the old state intact.
            before=path.read_bytes()
            p=original["payload"]
            with patch("dean_integration.selex_online_state.os.replace",side_effect=OSError("publication failure")):
                with self.assertRaises(OSError):
                    save_checkpoint(path,identity=c.identity,engine=c.engine,modules=c.modules(ns),optimizer=ns["opt"],
                                    discovery=p["discovery"],position=p["position"],journal=p["journal"])
            self.assertEqual(path.read_bytes(),before)
            self.assertFalse(list(c.output.glob(".state-*")))
            c.engine.phase="FROZEN"
            with self.assertRaisesRegex(ValueError,"mid-batch"):engine_state(c.engine)

    def test_fuse_probe_does_not_use_anonymous_file_and_cleans_only_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);keep=root/"user.txt";keep.write_text("preserve")
            with patch("tempfile.TemporaryFile",side_effect=OSError("O_TMPFILE unsupported")):
                probe_output(root)
            self.assertEqual(list(root.iterdir()),[keep])

    def test_lifecycle_ast_hooks_are_at_real_boundaries(self):
        path=Path(__file__).resolve().parents[2]/"OCGCD-dean-audit/Ours_train_online.py"
        tree=ast.parse(path.read_text(encoding="utf-8"))
        patch_experiment(tree,paired_online=True,reference_selex=True);patch_lifecycle(tree)
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=="(batch_idx, (on_x, on_y, on_z))")
        self.assertEqual(ast.unparse(loop.body[0]),"_dean_lifecycle.batch_started(sess_idx, batch_idx)")
        self.assertEqual(ast.unparse(loop.body[-1]),"_dean_lifecycle.batch_completed(globals())")
        text=ast.unparse(tree)
        self.assertIn("opt.step()\n                    _dean_lifecycle.step_completed()",text)
        self.assertNotIn("_dean_smoke",text)

    def test_hot_radii_copy_out_preserves_snapshot_hash(self):
        from tests.test_dean_hssk_phase3 import support
        snap=OnlineHSSKHierarchyProvider(cfg()).build(support(),context=context()).snapshot
        before=snap.state_dict()
        radii=snap.radii
        radii[0][next(iter(radii[0]))]=99
        self.assertEqual(snap.state_dict(),before)
        self.assertNotIn(99,snap.radii[0].values())

    def test_failed_higher_build_does_not_publish_memory_allocator_or_snapshot(self):
        from tests.test_dean_hssk_phase3 import support
        p=OnlineHSSKHierarchyProvider(cfg())
        previous=p.build(support(),context=context()).snapshot
        memory=p.memory.state_dict()
        with patch("dean_integration.selex_hssk_provider.coarsen_children",side_effect=ValueError("structural")):
            with self.assertRaisesRegex(ValueError,"structural"):
                p.build(support(100),context=context(1))
        self.assertEqual(p.memory.state_dict(),memory)
        self.assertEqual(p.implementation.previous.snapshot_hash,previous.snapshot_hash)
        self.assertEqual(p.implementation.last_batch,0)
        self.assertEqual(p.counters["hssk.error"],1)

    def test_lifecycle_rejects_early_boundary_and_extra_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,ns=fixture(Path(tmp)/"run")
            c.batch_started(0,0)
            with self.assertRaises(ValueError):c.batch_completed(ns)
            with self.assertRaises(ValueError):c.step_completed()
            with self.assertRaises(ValueError):c.batch_started(0,1)
            c.updates=15
            with self.assertRaises(ValueError):c.step_completed()

    def test_lifecycle_rejects_no_incoming_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,ns=fixture(Path(tmp)/"run",smoke=True)
            with self.assertRaises(ValueError):c.complete()

    def test_batch_manifest_cursor_retains_exact_order(self):
        from dean_integration.paired_stream import PairedOnlineStream, make_manifest
        class Data(torch.utils.data.Dataset):
            uq_idxs=np.arange(128)
            def __len__(self):return 128
            def __getitem__(self,i):return int(self.uq_idxs[i])
        data=Data();ctx=dict(online_seed=17)
        with tempfile.TemporaryDirectory() as tmp:
            stream=PairedOnlineStream(ctx,Path(tmp)/"manifest.json")
            stream.record=make_manifest([data],64,ctx)
            whole=list(stream.loader(data,0,64,0))
            suffix=list(stream.loader(data,0,64,0,start_batch=1))
            self.assertEqual(len(suffix),1);self.assertTrue(torch.equal(whole[1],suffix[0]))
            self.assertEqual(list(stream.loader(data,0,64,0,start_batch=2)),[])
            for bad in (-1,3,True):
                with self.assertRaises(ValueError):stream.loader(data,0,64,0,start_batch=bad)

    def test_phase4_cli_cannot_mix_smoke_and_production_or_escape_budget(self):
        import importlib.util
        import sys
        from dean_integration.config import ROOT
        spec=importlib.util.spec_from_file_location("phase4_cli",ROOT/"scripts/run_dean.py")
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        combinations=[("online-smoke",["--hssk-production"]),
                      ("lifecycle-smoke",["--smoke-updates","2"]),
                      ("worker",["--resume-online-state","missing.pth"]),
                      ("smoke-worker",["--hssk-production"])]
        for action,extra in combinations:
            argv=["run_dean","--upstream",".","--action",action,"--use_selex_online",
                  "--selex-implementation","reference_v1",*extra]
            with patch.object(sys,"argv",argv),patch.object(cli,"verify_checkout",return_value=ROOT), \
                 patch.object(cli,"check_assets",side_effect=AssertionError("Must fail before assets")):
                with self.subTest(action=action),self.assertRaises(SystemExit) as error:cli.main()
                self.assertEqual(error.exception.code,1)

    def test_named_probe_fsync_unsupported_and_readback_failure_cleanup(self):
        import errno
        with tempfile.TemporaryDirectory() as tmp:
            with patch("os.fsync",side_effect=OSError(errno.ENOTSUP,"FUSE")):probe_output(tmp)
            with patch.object(Path,"read_bytes",return_value=b"corrupt"):
                with self.assertRaisesRegex(OSError,"readback"):probe_output(tmp)
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_aggregate_reports_leaf_only_as_research_finding(self):
        result=hierarchy_aggregate({"hssk.build.leaf_only":9,"hssk.build.multi_level":1,
                                  "hssk.batch_decisions":12,"hssk.used_higher":2,"hssk.levels.1":9})
        self.assertEqual(result["higher_build_fraction"],.1)
        self.assertEqual(result["higher_batch_fraction"],2/12)
        self.assertIn("research finding",result["interpretation"])

    def test_corrupt_optimizer_or_missing_rng_is_rejected(self):
        from dean_integration.selex_online_state import optimizer_state,restore_optimizer,restore_engine
        with tempfile.TemporaryDirectory() as tmp:
            c,ns=fixture(Path(tmp)/"run");batch(c,ns,0,3)
            bad=optimizer_state(ns["opt"],c.modules(ns));bad["groups"][-1][0]="unknown.parameter"
            with self.assertRaisesRegex(ValueError,"identity"):restore_optimizer(bad,c.modules(ns))
            state=engine_state(c.engine);state["domains"].pop("replay_rng")
            with self.assertRaisesRegex(ValueError,"RNG"):restore_engine(c.engine,state)

    def test_resume_rejects_corrupt_journal_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,ns=fixture(root/"run");batch(c,ns,0,3)
            c.engine.telemetry.write_text("corrupt")
            with self.assertRaisesRegex(ValueError,"journal"):
                fixture(root/"restore",resume=c.output/"online_state.pth")

    def test_metrics_visible_after_each_committed_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,ns=fixture(Path(tmp)/"run")
            c.batch_started(0,0)
            c.engine.step=1
            c.engine.phase="FROZEN"
            c.engine.snapshot=None;c.engine.frozen_hash=None
            c.update_started()
            c.engine.pending_update=dict(batch=0,step=1,ce=1.,ec=0.,use=1.,sse=1.,sse_gt=1.,sse_hierarchy=1.,selex_total=1.,final_total=1.1)
            c.step_completed()
            records=[json.loads(x) for x in c.engine.telemetry.read_text().splitlines()]
            self.assertEqual(records[-1]["event"],"update")
            self.assertEqual(records[-1]["hierarchy_usage"],"FROZEN_REUSE")
            self.assertEqual(records[-1]["step"],1)

    def test_lifecycle_provenance_complete_failed_and_missing_state(self):
        from dean_integration.provenance import execute_run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,ns=fixture(root/"source");batch(c,ns,0,3);c.complete()
            summary=json.loads((c.output/"selex_reference_summary.json").read_text())
            for case in ("complete","failed","missing"):
                run=root/case;run.mkdir();child=run/"seed1"
                value=deepcopy(summary)
                if case=="missing":value["last_record"].pop("checkpoint")
                def runner(command,**kw):
                    (child/"selex_reference_summary.json").write_text(json.dumps(value))
                    return SimpleNamespace(returncode=1 if case=="failed" else 0)
                report=dict(real_selex_active=True,hierarchy_source="hssk_online",hssk_lifecycle=True)
                if case=="missing":
                    with self.assertRaises(ValueError):execute_run(run,report,[(1,child,["fixture"])],runner)
                else:
                    self.assertEqual(execute_run(run,report,[(1,child,["fixture"])],runner),1 if case=="failed" else 0)
                self.assertEqual(report["status"],"COMPLETE" if case=="complete" else "FAILED")

    def test_dataset_content_identity_changes_with_image_bytes(self):
        from dean_integration.preflight import dataset_content_hash
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"images").mkdir()
            (root/"images.txt").write_text("1 one.jpg\n2 two.jpg\n")
            (root/"images/one.jpg").write_bytes(b"one")
            (root/"images/two.jpg").write_bytes(b"two")
            before=dataset_content_hash(root)
            (root/"images.txt").write_text("2 two.jpg\n1 one.jpg\n")
            self.assertEqual(dataset_content_hash(root),before)
            (root/"images/two.jpg").write_bytes(b"changed")
            self.assertNotEqual(dataset_content_hash(root),before)
            (root/"images.txt").write_text("1 ../outside.jpg\n")
            with self.assertRaises(ValueError):dataset_content_hash(root)

    def test_failure_record_reports_oom_without_retry_or_budget_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,ns=fixture(Path(tmp)/"run",smoke=True)
            c.failed(torch.cuda.OutOfMemoryError("synthetic exception, no CUDA allocation"))
            record=json.loads((c.output/"lifecycle_failure.json").read_text())
            self.assertTrue(record["oom"]);self.assertEqual(record["exit_code"],1)
            self.assertEqual(c.started,0);self.assertTrue(c.smoke)
            self.assertEqual(c.engine.config.replay_selection_size,32)

    def test_each_present_side_must_strictly_reduce_per_contract(self):
        from dean_integration.selex_hssk_hierarchy import coarsening_stop_info
        centers=dict(enumerate(basis(5)))
        anchors={k:(11,) if k==0 else () for k in centers}
        self.assertIsNone(coarsen_children(centers=centers,support_counts={k:2 for k in centers},
                                         anchor_groups=anchors,seed=17,level=1))
        self.assertEqual(coarsening_stop_info(anchors)["reason"],"anchored_side_no_strict_reduction")


if __name__=="__main__":unittest.main()
