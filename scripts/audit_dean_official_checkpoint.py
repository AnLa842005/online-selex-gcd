"""Audit actual official full-checkpoint loading on CPU; no training, epochs or steps."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dean_integration.official_load_audit import audit_full_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trusted-local", action="store_true", required=True,
                        help="Authorize unpickling this trusted local full checkpoint on CPU")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists; select a new path")
    try:
        report = audit_full_checkpoint(args.checkpoint, args.upstream)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Checkpoint audit blocked: {exc}\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    # No tensor dump; samples are capped at 20 in the JSON report as well.
    print(json.dumps({key: report[key] for key in (
        "size", "sha256", "top_level_keys", "teacher_key_count", "prefix_distribution",
        "sample_teacher_keys", "weights_changed_after_official_load", "decision", "blockers")}, indent=2))
    return 0 if report["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
