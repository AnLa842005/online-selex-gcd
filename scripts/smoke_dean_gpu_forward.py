"""Linux one-batch forward smoke; opt-in --online-step performs exactly one update, zero epochs."""
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
    parser.add_argument("--online-step", action="store_true", help="Opt in to one online loss/backward/AdamW step")
    parser.add_argument("--batch-size", type=int, choices=range(1, 5), help="Default 1 forward / 2 online-step; maximum 4")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists; choose a new path")
    batch_size = args.batch_size or (2 if args.online_step else 1)
    if args.online_step and batch_size < 2:
        parser.error("--online-step requires batch-size >=2 for the mixed pseudo target fixture")
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
    sys.path.insert(0, str(checkout))
    try:
        from dataset.utils import make_transform
    finally:
        sys.path.pop(0)
    root = args.dataset_parent / "CUB_200_2011"
    images = dict(line.split(maxsplit=1) for line in (root / "images.txt").read_text().splitlines())
    train_ids = [line.split()[0] for line in (root / "train_test_split.txt").read_text().splitlines()
                 if line.split()[1] == "1"][:batch_size]
    inputs = []
    transform = make_transform(is_train=args.online_step)
    for train_id in train_ids:
        with Image.open(root / "images" / images[train_id]) as picture:
            inputs.append(transform(picture.convert("RGB")))
    batch = torch.stack(inputs)
    report = {"status": "FAIL", "scope": "online one-step fixture" if args.online_step else "initialization-path forward only",
              "checkpoint_sha256": audit["sha256"], "environment": prerequisite,
              "input_shape": list(batch.shape), "forward_seconds": None, "backward_seconds": None,
              "step_seconds": None, "peak_vram_allocated_bytes": None, "peak_vram_reserved_bytes": None,
              "batches": 1, "backward_calls": 0, "optimizer_steps": 0, "optimizer_step_attempts": 0, "epochs": 0,
              "transform": "official train" if args.online_step else "official test",
              "base_training": "NOT RUN", "full_training": "NOT RUN",
              "embedding": "fresh untrained official post-load Linear",
              "not_validated": ["base-trained head", "discovery pseudo-labelling", "reproduction accuracy"], "errors": []}
    if not args.online_step:
        report["not_validated"].extend(["online loss/backward", "PET updates"])
    try:
        torch.cuda.reset_peak_memory_stats()
        batch = batch.cuda()
        if args.online_step:
            from dean_integration.online_step_smoke import build_online_pair, run_online_step
            models = tuple(module.cuda() for module in build_online_pair(model, checkout))
            report.update(run_online_step(models, batch, checkout))
        else:
            model.requires_grad_(False)
            model.eval()
            model = model.cuda()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                features = model(batch)  # Exactly one batch, one forward call.
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            finite = bool(torch.isfinite(features).all().item())
            report.update(status="PASS" if finite else "FAIL", forward_seconds=elapsed,
                          output_shape=list(features.shape), output_finite=finite, forward_calls=1)
            if not finite:
                report["errors"].append("Nonfinite forward output")
    except (RuntimeError, ValueError) as exc:
        report.update(status="FAIL", errors=[f"{type(exc).__name__}: {exc}"])
    report.update(peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_vram_reserved_bytes=torch.cuda.max_memory_reserved())
    try:
        report["upstream_clean_after"] = bool(verify_checkout(checkout))
    except (RuntimeError, ValueError) as exc:
        report["status"] = "FAIL"
        report["errors"].append(str(exc))
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: report.get(k) for k in ("status", "input_shape", "output_shape", "forward_seconds",
                                                 "peak_vram_allocated_bytes", "optimizer_steps", "errors")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
