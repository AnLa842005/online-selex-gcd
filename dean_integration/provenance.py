"""Durable run lifecycle records without changing the training process."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .preflight import sha256


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def execute_run(run, report, jobs, runner):
    """Finalize aggregate and per-seed records even on spawn errors/interrupts."""
    run = Path(run)
    started = time.monotonic()
    report.update(status="STARTED", start_time=timestamp(), executions=[], returncode=None)
    write_json(run / "provenance.json", report)
    code = 0
    try:
        for seed, directory, command in jobs:
            directory.mkdir()
            status = dict(seed=seed, status="STARTED", command=command,
                          start_time=timestamp(), returncode=None)
            report["executions"].append(status)
            path = directory / "execution.json"
            write_json(path, status)
            tick = time.monotonic()
            try:
                with (directory / "console.log").open("w", encoding="utf-8") as log:
                    code = runner(command, stdout=log).returncode
                summary_path = directory / "selex_reference_summary.json"
                if report.get("real_selex_active"):
                    if summary_path.is_file():
                        try:
                            summary = json.loads(summary_path.read_text(encoding="utf-8"))
                            if not isinstance(summary, dict) or summary.get("implementation") != "reference_v1":
                                raise ValueError("Invalid real SelEx telemetry identity")
                            if "selex_config" in report and summary.get("config") != report["selex_config"]:
                                raise ValueError("Real SelEx telemetry config differs from provenance")
                            if code == 0 and (summary.get("step", 0) < 1 or not summary.get("optimizer_groups")):
                                raise ValueError("Real SelEx worker completed without optimizer updates")
                            status["selex_reference"] = summary
                            status["selex_reference_summary_sha256"] = sha256(summary_path)
                        except (OSError, ValueError, TypeError) as exc:
                            if code == 0:
                                raise
                            status["selex_reference_error"] = str(exc)
                    elif code == 0:
                        raise ValueError("Real SelEx worker completed without research telemetry")
                manifest = directory / "stream_manifest.json"
                if report.get("online_mode") == "paired":
                    if manifest.is_file():
                        from .paired_stream import manifest_summary, read_manifest
                        try:
                            if "stream_context" in report:
                                read_manifest(manifest, report["stream_context"], batch_size=64)
                            summary = manifest_summary(manifest)
                            if summary["online_seed"] != report["online_seed"]:
                                raise ValueError("Worker manifest online seed differs from provenance")
                            status["stream_manifest"] = summary
                        except (OSError, ValueError, KeyError, TypeError) as exc:
                            if code == 0:
                                raise
                            status["stream_manifest_error"] = str(exc)
                    elif code == 0:
                        raise ValueError("Paired worker completed without a stream manifest")
                status.update(status="COMPLETE" if code == 0 else "FAILED", returncode=code)
            except BaseException as exc:
                code = 130 if isinstance(exc, KeyboardInterrupt) else 1
                status.update(status="FAILED", returncode=code, error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                status.update(end_time=timestamp(), duration_seconds=time.monotonic() - tick,
                              output_paths=sorted(str(p.resolve()) for p in directory.rglob("*") if p.is_file()),
                              checkpoint_hashes={str(p.resolve()): sha256(p) for p in directory.rglob("*.pth") if p.is_file()})
                write_json(path, status)
            if code:
                break
        return code
    except BaseException as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if report.get("online_mode") == "paired":
            manifests = [s["stream_manifest"] for s in report["executions"] if "stream_manifest" in s]
            report["stream_manifests"] = manifests
            report["stream_manifest_sha256"] = manifests[0]["sha256"] if len(manifests) == 1 else None
        report.update(status="COMPLETE" if code == 0 else "FAILED", returncode=code,
                      end_time=timestamp(), duration_seconds=time.monotonic() - started,
                      output_root=str(run.resolve()),
                      output_paths=sorted(str(p.resolve()) for p in run.rglob("*") if p.is_file()))
        write_json(run / "provenance.json", report)
