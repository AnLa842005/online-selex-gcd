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


def bound_entrypoint(checkout, dataset_parent):
    path = Path(checkout) / "Ours_train_online.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    literals = [node for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and node.value == "Your data path.."]
    if len(literals) != 1:
        raise ValueError("Expected exactly one audited dataset path literal")
    literals[0].value = str(Path(dataset_parent).resolve())
    return compile(tree, str(path), "exec")
