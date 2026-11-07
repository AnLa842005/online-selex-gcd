"""Verify official source provenance and bind only deployment paths at runtime."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from .config import ROOT, UPSTREAM_SHA, UPSTREAM_URL


def git(checkout, *args):
    return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()


def verify_checkout(checkout):
    checkout = Path(checkout).resolve()
    if git(checkout, "rev-parse", "HEAD") != UPSTREAM_SHA:
        raise ValueError("External checkout must be at the pinned SHA")
    if git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("External checkout is dirty; do not patch upstream")
    if git(checkout, "remote", "get-url", "origin").removesuffix(".git") != UPSTREAM_URL.removesuffix(".git"):
        raise ValueError("External origin is not the official repository")
    lock = json.loads((ROOT / "configs/dean/upstream_lock.json").read_text())
    files = git(checkout, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    if sorted(files) != sorted(lock["files"]):
        raise ValueError("Upstream tracked file tree mismatch")
    for name, digest in lock["files"].items():
        # Git blob hashing avoids checkout CRLF differences on Windows.
        blob = subprocess.check_output(["git", "-C", str(checkout), "show", f"HEAD:{name}"])
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError(f"Upstream blob mismatch: {name}")
    return checkout


def bound_entrypoint(checkout, dataset_parent, *, online_only=False, use_selex_online=False, capture_boundary=False, paired_online=False, reference_selex=False, bounded_smoke=False, lifecycle=False):
    path = Path(checkout) / "Ours_train_online.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    literals = [node for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and node.value == "Your data path.."]
    if len(literals) != 1:
        raise ValueError("Expected exactly one audited dataset path literal")
    literals[0].value = str(Path(dataset_parent).resolve())
    if online_only or paired_online or use_selex_online or reference_selex:
        from .runtime import patch_experiment
        tree = patch_experiment(tree, online_only=online_only, use_selex_online=use_selex_online, paired_online=paired_online,
                                reference_selex=reference_selex)
    # Sidecar instrumentation is RNG-neutral; official loss and loop are untouched.
    if online_only or paired_online or capture_boundary:
        from .runtime import instrument_boundary
        tree = instrument_boundary(tree, online_only=online_only, paired_online=paired_online)
    if lifecycle:
        if bounded_smoke or not paired_online or not reference_selex:
            raise ValueError("Lifecycle requires separate active paired HSSK path")
        from .runtime import patch_lifecycle
        tree = patch_lifecycle(tree)
    if bounded_smoke:
        if not paired_online or not reference_selex:
            raise ValueError("Bounded HSSK smoke requires paired online and active reference losses")
        from .runtime import patch_bounded_smoke
        tree = patch_bounded_smoke(tree)
    ast.fix_missing_locations(tree)
    return compile(tree, str(path), "exec")
