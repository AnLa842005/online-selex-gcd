"""Versioned experiment contracts; importing this module never imports DEAN."""
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_URL = "https://github.com/KU-VGI/OCGCD.git"
UPSTREAM_SHA = "28b9384cd7ea9b123b03d045bac2e92d8d0b2870"
BASE_SHA = "031bd87a7c7636036c6ccb7e5540739c721b5398"
OFFICIAL_ARGS = [
    "--model", "VIT", "--dataset", "cub", "--alpha", "32", "--mrg", "0.1",
    "--lr", "1e-4", "--warm", "5", "--base_epochs", "30", "--inc_epochs", "15",
    "--batch-size", "64", "--inc_batch-size", "64", "--num_base_classes", "160",
    "--num_inc_sessions", "1", "--lora_layers", "7", "8", "9", "10", "11",
    "--energy_hp", "1.", "--n_replay", "5", "--use_EC", "--use_VFA",
]


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")
    mode = config.get("mode")
    if mode not in ("official", "ssb100_draft"):
        raise ValueError("Unknown DEAN protocol mode")
    reference = ROOT / "configs/dean" / (
        "cub_official_v0.yaml" if mode == "official" else "cub_ssb100_v0.yaml")
    expected = yaml.safe_load(reference.read_text(encoding="utf-8"))
    if config != expected:
        raise ValueError("Versioned config differs from its audited contract; create a new version")
    lock = json.loads((ROOT / "configs/dean/upstream_lock.json").read_text())
    if config_digest(config) != lock["config_sha256"][reference.name]:
        raise ValueError("Config digest differs from audited lock")
    if config["upstream"] != {"url": UPSTREAM_URL, "sha": UPSTREAM_SHA, "branch": "main"}:
        raise ValueError("Upstream provenance mismatch")
    return config


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
