"""Read-only asset/runtime checks except an explicit output write probe."""
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cub_metadata(root):
    root = Path(root)
    tables = {}
    for name in ("images", "image_class_labels", "train_test_split"):
        rows = [line.split(maxsplit=1) for line in (root / f"{name}.txt").read_text().splitlines()]
        table = {int(key): value for key, value in rows}
        if len(table) != len(rows):
            raise ValueError(f"Duplicate CUB IDs in {name}")
        tables[name] = table
    images, labels, splits = (tables[n] for n in ("images", "image_class_labels", "train_test_split"))
    if set(images) != set(range(1, 11789)) or images.keys() != labels.keys() or images.keys() != splits.keys():
        raise ValueError("Expected 11788 aligned official CUB image IDs")
    if set(map(int, labels.values())) != set(range(1, 201)):
        raise ValueError("Expected official CUB labels 1..200")
    if set(splits.values()) != {"0", "1"} or sum(v == "1" for v in splits.values()) != 5994:
        raise ValueError("Expected official 5994 train / 5794 test split")
    for relative in images.values():
        path = (root / "images" / relative).resolve()
        if not path.is_relative_to((root / "images").resolve()) or not path.is_file():
            raise ValueError("CUB image missing or outside image root")
    per_class = [sum(int(labels[i]) == cls and splits[i] == "1" for i in images)
                 for cls in range(1, 201)]
    return {"train": 5994, "test": 5794, "known": 160, "novel": 40,
            "base_labelled": sum(int(n * .8) for n in per_class[:160]),
            "metadata_sha256": {name: sha256(root / f"{name}.txt") for name in tables}}


def check_assets(dataset_parent, checkpoint, output):
    report, errors = {"python": platform.python_version()}, []
    expected = {"torch": "2.0.0", "torchvision": "0.15.1", "timm": "0.9.8",
                "scikit-learn": "1.3.2", "numpy": "1.24.4", "scipy": "1.10.1",
                "pandas": "2.0.3", "Pillow": "9.5.0", "matplotlib": "3.7.3",
                "tqdm": "4.66.1", "einops": "0.7.0", "pytorch-metric-learning": "2.3.0",
                "PyYAML": "6.0.1"}
    if platform.python_version() != "3.9.18":
        errors.append("Expected isolated Python 3.9.18 environment")
    report["packages"] = {}
    for name, version in expected.items():
        try:
            actual = importlib.metadata.version(name)
            report["packages"][name] = actual
            if actual.split("+")[0] != version:
                errors.append(f"{name}: expected {version}, found {actual}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing package: {name}")
    try:
        import torch
        report.update(cuda=torch.version.cuda, gpu_available=torch.cuda.is_available())
        if not torch.cuda.is_available() or torch.version.cuda != "11.8":
            errors.append("Official launcher requires working CUDA 11.8 GPU runtime")
        else:
            report["gpu"] = torch.cuda.get_device_name(0)
        # Only load explicitly supplied, trusted local checkpoints; no downloads.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not isinstance(state.get("teacher"), dict) or not state["teacher"]:
            raise ValueError("Official DEAN expects a nonempty full-checkpoint ['teacher'] mapping")
        report["checkpoint"] = {"path": str(Path(checkpoint).resolve()), "sha256": sha256(checkpoint),
                                "teacher_keys": len(state["teacher"]),
                                "model_key_compatibility": "NOT VERIFIED; upstream strict=False needs audit"}
    except Exception as exc:
        errors.append(f"Checkpoint/runtime: {exc}")
    try:
        report["cub"] = cub_metadata(Path(dataset_parent) / "CUB_200_2011")
    except (OSError, ValueError) as exc:
        errors.append(f"CUB metadata: {exc}")
    try:
        Path(output).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=output) as handle:
            handle.write(b"DEAN output probe")
    except OSError as exc:
        errors.append(f"Output is not writable: {exc}")
    report["errors"] = errors
    return report
