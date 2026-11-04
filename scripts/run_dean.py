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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["plan", "verify", "smoke", "train", "worker"], default="plan")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dean/cub_official_v0.yaml")
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--dataset-parent", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, help="Default: all five official seeds")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if config["mode"] != "official":
            raise ValueError("SSB draft is NOT RUN and cannot enter the official runner")
        checkout = verify_checkout(args.upstream)
        seeds = [args.seed] if args.seed is not None else config["seeds"]
        for seed in seeds:
            OfficialProtocolAdapter.arguments(config, seed, "<DINO_FULL_CHECKPOINT>")
        if args.action == "smoke":
            from dean_integration.smoke import run_smoke
            print(json.dumps(run_smoke(checkout), indent=2))
            return 0
        report = {"config": config, "config_sha256": config_digest(config),
                  "environment_spec_sha256": sha256(ROOT / "environments/dean/environment.yml"),
                  "checkout": str(checkout), "seeds": seeds,
                  "arguments": {str(s): OfficialProtocolAdapter.arguments(
                      config, s, args.checkpoint or "<DINO_FULL_CHECKPOINT>") for s in seeds},
                  "status": "NOT RUN"}
        if args.action == "plan":
            print(json.dumps(report, indent=2))
            return 0
        if not all((args.dataset_parent, args.checkpoint, args.output)):
            raise ValueError("verify/train require --dataset-parent, --checkpoint and --output")
        output = args.output.resolve()
        if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "runs/dean"):
            raise ValueError("Inside this repo output must be below runs/dean")
        assets = check_assets(args.dataset_parent, args.checkpoint, output, upstream=checkout)
        report["preflight"] = assets
        if args.action == "verify" or assets["errors"]:
            print(json.dumps(report, indent=2))
            return int(bool(assets["errors"]))
        if args.action == "worker":
            if args.seed is None:
                raise ValueError("Internal worker requires a single seed")
            # Runtime AST changes only the dataset-parent string. Source stays pristine.
            code = bound_entrypoint(checkout, args.dataset_parent)
            sys.path.insert(0, str(checkout))
            sys.argv = [str(checkout / "Ours_train_online.py")] + report["arguments"][str(args.seed)]
            os.chdir(output)
            exec(code, {"__name__": "__main__", "__file__": str(checkout / "Ours_train_online.py")})
            return 0
        run = output / "official/cub_v0"
        run.mkdir(parents=True, exist_ok=False)
        (run / "provenance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for seed in seeds:
            seed_dir = run / f"seed{seed}"
            seed_dir.mkdir()
            command = [sys.executable, str(Path(__file__).resolve()), "--action", "worker",
                       "--config", str(args.config.resolve()), "--upstream", str(checkout),
                       "--dataset-parent", str(args.dataset_parent.resolve()),
                       "--checkpoint", str(args.checkpoint.resolve()), "--output", str(seed_dir),
                       "--seed", str(seed)]
            status = {"seed": seed, "status": "STARTED", "command": command}
            status_path = seed_dir / "execution.json"
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            try:
                with (seed_dir / "console.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                status.update(status="COMPLETE" if result.returncode == 0 else "FAILED", returncode=result.returncode)
            except KeyboardInterrupt:
                status.update(status="INTERRUPTED")
                raise
            finally:
                status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            if result.returncode:
                return result.returncode
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"DEAN error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
