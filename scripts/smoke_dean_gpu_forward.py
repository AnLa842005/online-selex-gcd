"""Future Linux DEAN-1A one-batch forward only; no epochs, backward or optimizer."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trusted-local", action="store_true", required=True)
    parser.add_argument("--dataset-parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists; choose a new path")
    import torch
    from PIL import Image
    from dean_integration.external import verify_checkout
    from dean_integration.preflight import check_assets
    from dean_integration.official_load_audit import audit_full_checkpoint
    from dean_integration.checkpoint_audit import build_reference_model

    checkout = verify_checkout(args.upstream)
    prerequisite = check_assets(args.dataset_parent, args.checkpoint, args.report.parent, upstream=checkout)
    if prerequisite["errors"]:
        parser.exit(1, json.dumps(prerequisite["errors"], indent=2) + "\n")
    audit = audit_full_checkpoint(args.checkpoint, checkout)
    if audit["decision"] != "PASS":
        parser.exit(1, "Raw official checkpoint audit did not PASS\n")
    torch.manual_seed(1)
    model = build_reference_model(checkout)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["teacher"], strict=False)
    del checkpoint
    # Same post-load replacements as the official entrypoint, before base training.
    model.backbone.embedding = torch.nn.Linear(model.backbone.embed_dim, 768)
    model.head = torch.nn.Identity()
    model.requires_grad_(False)
    model.eval()
    sys.path.insert(0, str(checkout))
    try:
        from dataset.utils import make_transform
    finally:
        sys.path.pop(0)
    root = args.dataset_parent / "CUB_200_2011"
    images = dict(line.split(maxsplit=1) for line in (root / "images.txt").read_text().splitlines())
    train_id = next(line.split()[0] for line in (root / "train_test_split.txt").read_text().splitlines()
                    if line.split()[1] == "1")
    with Image.open(root / "images" / images[train_id]) as picture:
        batch = make_transform(is_train=False)(picture.convert("RGB")).unsqueeze(0)
    model = model.cuda()
    batch = batch.cuda()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        features = model(batch)  # Exactly one real-image batch, one forward call.
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    finite = bool(torch.isfinite(features).all().item())
    report = {"status": "PASS" if finite else "FAIL", "scope": "initialization-path forward only",
              "checkpoint_sha256": audit["sha256"], "environment": prerequisite,
              "input_shape": list(batch.shape), "output_shape": list(features.shape),
              "output_finite": finite, "forward_seconds": elapsed,
              "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
              "batches": 1, "forward_calls": 1, "backward_calls": 0, "optimizer_steps": 0,
              "base_training": "NOT RUN", "full_training": "NOT RUN",
              "embedding": "fresh untrained official post-load Linear",
              "not_validated": ["base-trained head", "online EGD/loss/backward", "PET updates"],
              "upstream_clean_after": bool(verify_checkout(checkout))}
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "input_shape", "output_shape", "forward_seconds",
                                             "peak_vram_allocated_bytes", "optimizer_steps")}, indent=2))
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
