"""Read-only CUB split preflight; creates a new eligibility manifest, no training."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dean_integration.external import verify_checkout
from dean_integration.preflight import cub_metadata
from dean_integration.selex_hierarchy_provider import digest
from dean_integration.selex_replay_manifest import verified_manifest, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prop-samples", type=float, default=.8)
    parser.add_argument("--base-classes", type=int, default=160)
    args = parser.parse_args()
    try:
        verify_checkout(args.upstream)
        cub_metadata(args.dataset_root)  # Real official CUB inventory; fixtures use the library API only.
        value, _ = verified_manifest(args.dataset_root, prop_samples=args.prop_samples, base_classes=args.base_classes)
        write_manifest(args.output, value)
        print(json.dumps(dict(path=str(args.output.resolve()), canonical_sha256=digest(value),
                              eligible_references=len(value["eligibility"]["records"])), indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(1, str(exc)+"\n")


if __name__ == "__main__":
    main()
