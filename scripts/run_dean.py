"""Plan/verify/smoke external official DEAN; training requires explicit --action train."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dean_integration.adapters import OfficialProtocolAdapter
from dean_integration.config import config_digest, load_config
from dean_integration.external import bound_entrypoint, verify_checkout
from dean_integration.preflight import check_assets, sha256
from dean_integration.provenance import execute_run


def worker_command(args, checkout, directory, seed):
    bounded = getattr(args, "action", None) == "online-smoke"
    command = [sys.executable, str(Path(__file__).resolve()), "--action", "smoke-worker" if bounded else "worker",
               "--config", str(args.config.resolve()), "--upstream", str(checkout),
               "--dataset-parent", str(args.dataset_parent.resolve()),
               "--checkpoint", str(args.checkpoint.resolve()), "--output", str(directory),
               "--seed", str(seed)]
    for flag, value in (("--base-checkpoint", args.base_checkpoint), ("--boundary-rng", args.boundary_rng),
                        ("--selex-config", args.selex_config)):
        if value is not None:
            command += [flag, str(value.resolve())]
    if args.online_only:
        command.append("--online-only")
    if getattr(args, "paired_online", False):
        command += ["--paired-online", "--online_seed", str(args.online_seed)]
    if getattr(args, "stream_manifest", None) is not None:
        command += ["--stream-manifest", str(args.stream_manifest.resolve())]
    if args.use_selex_online:
        command.append("--use_selex_online")
    if getattr(args, "selex_implementation", "synthetic") != "synthetic":
        command += ["--selex-implementation", args.selex_implementation]
    if bounded:
        command += ["--smoke-updates", str(args.smoke_updates)]
        if args.replay_manifest is not None:
            command += ["--replay-manifest", str(args.replay_manifest), "--replay-manifest-sha256", args.replay_manifest_sha256]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["plan", "verify", "smoke", "train", "worker", "online-smoke", "smoke-worker"], default="plan")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dean/cub_official_v0.yaml")
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--dataset-parent", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, help="Default: all five official seeds")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--online-only", action="store_true", help="Strict replay; requires historical RNG sidecar")
    modes.add_argument("--paired-online", action="store_true", help="New paired stream; skips base, resets online RNG")
    parser.add_argument("--online_seed", type=int, help="Required seed for paired online boundary/stream")
    parser.add_argument("--stream-manifest", type=Path, help="Read existing paired stream; omit to generate a new one")
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--boundary-rng", type=Path)
    parser.add_argument("--use_selex_online", action="store_true", help="Enable the explicitly selected SelEx implementation")
    parser.add_argument("--selex-implementation", choices=["synthetic", "reference_v1"], default="synthetic",
                        help="Synthetic scaffolding or SelEx adapted to DEAN online discovery")
    parser.add_argument("--selex-config", type=Path)
    parser.add_argument("--smoke-updates", type=int, default=1)
    parser.add_argument("--replay-manifest", type=Path)
    parser.add_argument("--replay-manifest-sha256")
    args = parser.parse_args()
    for name in ("config", "upstream", "dataset_parent", "checkpoint", "output", "base_checkpoint", "boundary_rng", "selex_config", "stream_manifest", "replay_manifest"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    try:
        config = load_config(args.config)
        if config["mode"] != "official":
            raise ValueError("SSB draft is NOT RUN and cannot enter the official runner")
        checkout = verify_checkout(args.upstream)
        seeds = [args.seed] if args.seed is not None else config["seeds"]
        from dean_integration.experiment_config import load_selex_config
        selex = load_selex_config(args.selex_config, args.selex_implementation)
        real_active = args.use_selex_online and args.selex_implementation == "reference_v1" and selex.gamma > 0
        bounded = args.action in ("online-smoke", "smoke-worker")
        if bounded:
            from dean_integration.selex_smoke_gate import SmokeBudget
            smoke_budget = SmokeBudget(args.smoke_updates)
            if not real_active or selex.hierarchy_source != "hssk_online" or not args.paired_online:
                raise ValueError("Online smoke requires active hssk_online and explicit paired mode")
        elif args.replay_manifest is not None or args.replay_manifest_sha256 is not None or args.smoke_updates != 1:
            raise ValueError("Replay manifest/smoke budget options require explicit online-smoke")
        if args.online_only and (args.seed is None or not args.base_checkpoint or not args.boundary_rng):
            raise ValueError("--online-only requires explicit --seed, --base-checkpoint and --boundary-rng; never retrains base")
        if args.paired_online:
            from dean_integration.paired_stream import validate_online_seed
            validate_online_seed(args.online_seed)
            if args.seed is None or not args.base_checkpoint:
                raise ValueError("--paired-online requires explicit --seed and --base-checkpoint; never retrains base")
            if args.boundary_rng:
                raise ValueError("--paired-online does not accept --boundary-rng; choose strict replay explicitly")
        elif args.online_seed is not None or args.stream_manifest is not None:
            raise ValueError("--online_seed and --stream-manifest require --paired-online")
        if not (args.online_only or args.paired_online) and (args.base_checkpoint or args.boundary_rng):
            raise ValueError("Base checkpoint/RNG options require an explicit online mode")
        for seed in seeds:
            OfficialProtocolAdapter.arguments(config, seed, "<DINO_FULL_CHECKPOINT>")
        report = {"config": config, "config_sha256": config_digest(config),
                  "environment_spec_sha256": sha256(ROOT / "environments/dean/environment.yml"),
                  "checkout": str(checkout), "seeds": seeds,
                  "arguments": {str(s): OfficialProtocolAdapter.arguments(
                      config, s, args.checkpoint or "<DINO_FULL_CHECKPOINT>") for s in seeds},
                  "status": "NOT RUN"}
        report.update(upstream_sha=config["upstream"]["sha"],
                      online_only=args.online_only, use_selex_online=args.use_selex_online,
                      paired_online=args.paired_online,
                      online_mode="paired" if args.paired_online else "strict_replay" if args.online_only else "full",
                      online_seed=args.online_seed,
                      selex_implementation=args.selex_implementation, real_selex_active=real_active,
                      selex_config=selex.__dict__,
                      selex_config_sha256=config_digest(selex.__dict__),
                      runtime_source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                                             [Path(__file__).resolve(), *sorted((ROOT / "dean_integration").glob("*.py"))]})
        report["wrapper_arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        if bounded:
            report.update(git_sha=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                          git_dirty=bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()))
        if args.selex_implementation == "reference_v1":
            from dean_integration.selex_hierarchy_provider import provider_metadata, ensure_provider_ready
            report.update(provider_metadata(selex))
        if args.action == "plan":
            print(json.dumps(report, indent=2))
            return 0
        if real_active:
            if selex.hierarchy_source == "hssk_online" and not bounded:
                report.update(status="BLOCKED", error="Full training blocked: Phase 4 resume/provenance readiness required")
                print(json.dumps(report, indent=2))
            ensure_provider_ready(selex, bounded)
        if args.action == "smoke":
            from dean_integration.smoke import run_smoke
            print(json.dumps(run_smoke(checkout), indent=2))
            return 0
        if not all((args.dataset_parent, args.checkpoint, args.output)):
            raise ValueError("verify/train require --dataset-parent, --checkpoint and --output")
        output = args.output.resolve()
        if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "runs/dean"):
            raise ValueError("Inside this repo output must be below runs/dean")
        assets = check_assets(args.dataset_parent, args.checkpoint, output, upstream=checkout)
        report["preflight"] = assets
        report["checkpoint_sha256"] = sha256(args.checkpoint)
        boundary_rng = None
        context = {"seed": args.seed, "upstream_sha": config["upstream"]["sha"],
                   "config_sha256": report["config_sha256"], "dino_sha256": report["checkpoint_sha256"],
                   "dataset_metadata": assets.get("cub", {}).get("metadata_sha256"),
                   "arguments": report["arguments"].get(str(args.seed))}
        paired = None
        replay_manager = None
        replay_record = None
        if bounded and not assets["errors"]:
            from dean_integration.selex_replay_manifest import verified_manifest
            from dean_integration.selex_labeled_replay import VerifiedLabeledReplay
            from dean_integration.selex_hierarchy_provider import digest
            replay_record, eligible = verified_manifest(args.dataset_parent/"CUB_200_2011", args.replay_manifest,
                                                         args.replay_manifest_sha256)
            replay_manager = VerifiedLabeledReplay(eligible)
            replay_manager.admit(eligible.entry(k) for k in sorted(eligible._records))
            report["verified_replay"] = dict(canonical_sha256=digest(replay_record),
                dataset=eligible.dataset_fingerprint, split=eligible.split_fingerprint, source=replay_record["source"])
            report["bounded_smoke"] = dict(incoming_batches=smoke_budget.max_incoming_batches, optimizer_updates=args.smoke_updates, full_train_blocked=True)
        if args.online_only or args.paired_online:
            report.update(base_checkpoint=str(args.base_checkpoint.resolve()),
                          base_checkpoint_sha256=sha256(args.base_checkpoint))
            if args.online_only:
                from dean_integration.base_checkpoint import read_boundary
                boundary_rng = read_boundary(args.base_checkpoint, args.boundary_rng, context)
                report.update(boundary_rng=str(args.boundary_rng.resolve()), boundary_rng_sha256=sha256(args.boundary_rng))
            else:
                from dean_integration.paired_stream import PairedOnlineStream, read_manifest
                paired_context = {**context, "online_seed": args.online_seed,
                                  "base_checkpoint_sha256": report["base_checkpoint_sha256"]}
                report["stream_context"] = paired_context
                if args.stream_manifest:
                    # Exact dataset IDs are checked when upstream builds the actual stream datasets.
                    read_manifest(args.stream_manifest, paired_context, batch_size=64)
                    report.update(input_stream_manifest=str(args.stream_manifest),
                                  input_stream_manifest_sha256=sha256(args.stream_manifest))
                paired = PairedOnlineStream(paired_context, output / "stream_manifest.json", args.stream_manifest)
            if not assets["errors"]:
                from dean_integration.base_checkpoint import audit_base
                report["base_checkpoint_validation"] = audit_base(args.base_checkpoint, checkout)
        if args.action == "verify" or assets["errors"]:
            print(json.dumps(report, indent=2))
            return int(bool(assets["errors"]))
        if args.action in ("worker", "smoke-worker"):
            if args.seed is None:
                raise ValueError("Internal worker requires a single seed")
            if any((output / name).exists() for name in ("Ours", "report_dict", "logs")):
                raise ValueError("Worker output contains training artifacts; choose a fresh output directory")
            synthetic_active = args.use_selex_online and args.selex_implementation == "synthetic"
            code = bound_entrypoint(checkout, args.dataset_parent, online_only=args.online_only,
                                    use_selex_online=synthetic_active, reference_selex=real_active,
                                    capture_boundary=True, paired_online=args.paired_online,
                                    **({"bounded_smoke": True} if bounded else {}))
            from dean_integration.base_checkpoint import load_base, restore_rng, save_boundary
            from dean_integration.selex_online import augment_online_loss
            real = None
            if real_active:
                from dean_integration.selex_real_online import RealSelExOnline
                options = {}
                if bounded:
                    from dean_integration.selex_replay_manifest import write_manifest
                    write_manifest(output/"replay_eligibility.json", replay_record)
                    options = dict(bounded_smoke=True, replay_manager=replay_manager, context=dict(
                        model_lineage=report["base_checkpoint_sha256"], dataset=report["verified_replay"]["dataset"],
                        feature_version="dino768-eval-v1", seed_protocol="paired-hssk-v1"))
                real = RealSelExOnline(selex, args.online_seed if args.paired_online else args.seed, output, **options)
                if bounded:
                    smoke_budget.bind_engine(real, dict(git_sha=report["git_sha"], upstream_sha=report["upstream_sha"],
                        config_sha256=report["config_sha256"], selex_config_sha256=report["selex_config_sha256"],
                        checkpoint_sha256=report["checkpoint_sha256"], base_checkpoint_sha256=report["base_checkpoint_sha256"],
                        replay_manifest_sha256=report["verified_replay"]["canonical_sha256"],
                        dataset_sha256=report["verified_replay"]["dataset"], seed=args.seed, online_seed=args.online_seed))
            sys.path.insert(0, str(checkout))
            sys.argv = [str(checkout / "Ours_train_online.py")] + report["arguments"][str(args.seed)]
            os.chdir(output)
            namespace_values = {"__name__": "__main__", "__file__": str(checkout / "Ours_train_online.py"),
                        "_dean_load_base": load_base, "_dean_restore": restore_rng,
                        "_dean_save_boundary": save_boundary, "_dean_context": context,
                        "_dean_base_path": str(args.base_checkpoint.resolve()) if args.base_checkpoint else None,
                        "_dean_boundary_rng": boundary_rng, "_dean_augment": augment_online_loss,
                        "_dean_paired": paired,
                        "_dean_real": real,
                        "_dean_selex_config": selex}
            if bounded:
                namespace_values["_dean_smoke"] = smoke_budget
                from dean_integration.selex_smoke_gate import BoundedSmokeComplete
                try:
                    exec(code, namespace_values)
                except BoundedSmokeComplete:
                    real._event("bounded_smoke_complete", updates=smoke_budget.steps, full_train_blocked=True)
                except BaseException as exc:
                    smoke_budget.failed(exc)
                    real._event("bounded_smoke_failed", exit_code=1, **smoke_budget.batch_summary())
                    raise
                else:
                    error = ValueError("Smoke worker exited without audited post-step budget termination")
                    smoke_budget.failed(error)
                    real._event("bounded_smoke_failed", exit_code=1, **smoke_budget.batch_summary())
                    raise error
            else:
                exec(code, namespace_values)
            return 0
        namespace = ("experiments/cub_hssk_smoke_v1" if bounded else "experiments/cub_paired_v1" if args.paired_online else
                     "experiments/cub_online_v1" if args.online_only or args.use_selex_online else "official/cub_v0")
        run = output / namespace
        run.mkdir(parents=True, exist_ok=False)
        if bounded and args.replay_manifest is None:
            from dean_integration.selex_replay_manifest import write_manifest
            from dean_integration.selex_hierarchy_provider import digest
            args.replay_manifest = run/"replay_eligibility.json"
            write_manifest(args.replay_manifest, replay_record)
            args.replay_manifest_sha256 = digest(replay_record)
        jobs = [(seed, run / f"seed{seed}", worker_command(args, checkout, run / f"seed{seed}", seed)) for seed in seeds]
        return execute_run(run, report, jobs,
                           lambda command, **kw: subprocess.run(command, stderr=subprocess.STDOUT, **kw))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"DEAN error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
