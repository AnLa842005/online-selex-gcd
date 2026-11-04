"""Produce a DEAN-1A prerequisite report. This command never trains or runs GPU kernels."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dean_integration.config import UPSTREAM_SHA, config_digest, load_config
from dean_integration.external import verify_checkout
from dean_integration.preflight import check_assets, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--dataset-parent", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trusted local checkpoint only")
    parser.add_argument("--output", required=True, type=Path, help="Temporary writable smoke output root")
    parser.add_argument("--report", required=True, type=Path, help="New JSON report; never overwritten")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists; choose a new report path")
    config = load_config(ROOT / "configs/dean/cub_official_v0.yaml")
    report = {"schema": "dean-1a-prerequisites-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "project_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "project_dirty_at_audit": bool(subprocess.check_output(
                  ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
              "config_sha256": config_digest(config),
              "environment_spec_sha256": sha256(ROOT / "environments/dean/environment.yml"),
              "gpu_smoke": {"status": "NOT RUN", "optimizer_steps": 0, "peak_vram_bytes": None,
                            "forward_seconds": None, "backward_seconds": None,
                            "reason": "Prerequisites must pass; a separate bounded GPU run is still required"},
              "full_training": "NOT RUN", "new_accuracy": None,
              "parameter_update_audit": {"status": "NOT RUN", "reason": "No GPU forward/backward/step executed"}}
    try:
        checkout = verify_checkout(args.upstream)
        report["upstream"] = {"sha": UPSTREAM_SHA, "path": str(checkout), "verification": "PASS"}
        report["preflight"] = check_assets(args.dataset_parent, args.checkpoint, args.output, upstream=checkout)
        from dean_integration.smoke import run_smoke
        report["cpu_components"] = run_smoke(checkout)
        report["upstream"]["clean_after_audit"] = bool(verify_checkout(checkout))
        report["errors"] = report["preflight"]["errors"]
        report["status"] = "BLOCKED" if report["errors"] else "PREREQUISITES_PASS_GPU_SMOKE_PENDING"
    except Exception as exc:
        report.update(status="BLOCKED", errors=[f"{type(exc).__name__}: {exc}"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "report": str(args.report.resolve()),
                      "errors": report["errors"]}, indent=2))
    return 1 if report["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
