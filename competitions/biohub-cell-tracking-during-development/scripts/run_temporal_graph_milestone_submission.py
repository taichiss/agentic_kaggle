#!/usr/bin/env python
"""Run the restart-safe EXP-0009 epoch-5/30 Kaggle submission workflow."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

COMPETITION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = COMPETITION_ROOT.parents[1]
REPOSITORY_SOURCE = REPOSITORY_ROOT / "src"
if str(REPOSITORY_SOURCE) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_SOURCE))

from agentic_kaggle.leaderboard import load_manifest, validate_submission  # noqa: E402

DEFAULT_CONFIG = COMPETITION_ROOT / "configs/exp-0009-host-tgraph3-residual-30e.toml"
PACKAGER = COMPETITION_ROOT / "scripts/prepare_temporal_graph_submission.py"
BIOHUB_VALIDATOR = COMPETITION_ROOT / "src/submission.py"
COMPETITION_SLUG = "biohub-cell-tracking-during-development"
ALLOWED_COMPLETED_EPOCHS = frozenset({5, 30})
KERNEL_SUCCESS_STATUSES = frozenset({"complete", "completed"})
KERNEL_FAILURE_STATUSES = frozenset(
    {"cancelled", "canceled", "error", "failed", "failure"}
)
KERNEL_STATUS_PATTERN = re.compile(r'has status\s+"([^"]+)"', re.IGNORECASE)
STATE_SCHEMA_VERSION = 1

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class MilestonePlan:
    config_path: Path
    experiment_id: str
    completed_epoch: int
    checkpoint: Path
    variant: str
    dataset_id: str
    kernel_id: str
    postprocess_profile: str
    bundle_root: Path
    state_path: Path
    kernel_output_dir: Path
    test_dir: Path
    message: str

    @property
    def dataset_dir(self) -> Path:
        return self.bundle_root / "dataset"

    @property
    def kernel_dir(self) -> Path:
        return self.bundle_root / "kernel"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_competition_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (COMPETITION_ROOT / path).resolve()


def _submission_milestone(
    config: Mapping[str, Any], completed_epoch: int
) -> dict[str, Any]:
    if completed_epoch not in ALLOWED_COMPLETED_EPOCHS:
        raise ValueError("completed epoch must be one of the EXP-0009 milestones: 5 or 30")
    submission = config.get("submission")
    if not isinstance(submission, Mapping):
        raise ValueError("config is missing [submission]")
    if submission.get("competition") != COMPETITION_SLUG:
        raise ValueError("submission competition does not match the Biohub workspace")
    milestones = submission.get("milestones")
    if not isinstance(milestones, list):
        raise ValueError("config is missing [[submission.milestones]]")
    matches = [
        dict(item)
        for item in milestones
        if isinstance(item, Mapping)
        and int(item.get("completed_epoch", -1)) == completed_epoch
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one configured milestone for completed epoch "
            f"{completed_epoch}, found {len(matches)}"
        )
    required = {
        "checkpoint",
        "variant",
        "dataset_id",
        "kernel_id",
        "postprocess_profile",
    }
    missing = sorted(required - matches[0].keys())
    if missing:
        raise ValueError(f"submission milestone is missing fields: {missing}")
    return matches[0]


def build_plan(
    config_path: Path,
    completed_epoch: int,
    *,
    bundle_root: Path | None = None,
    state_path: Path | None = None,
    kernel_output_dir: Path | None = None,
) -> MilestonePlan:
    config_path = config_path.expanduser().resolve()
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    milestone = _submission_milestone(config, completed_epoch)
    experiment_id = str(config.get("experiment_id", "")).strip()
    if experiment_id != "EXP-0009":
        raise ValueError(f"this workflow only accepts EXP-0009, found {experiment_id!r}")
    output = config.get("output")
    if not isinstance(output, Mapping) or "artifact_dir" not in output:
        raise ValueError("config is missing output.artifact_dir")
    data = config.get("data")
    if not isinstance(data, Mapping) or "test_dir" not in data:
        raise ValueError("config is missing data.test_dir")
    artifact_dir = _resolve_competition_path(str(output["artifact_dir"]))
    resolved_bundle = (
        bundle_root.expanduser().resolve()
        if bundle_root is not None
        else (
            COMPETITION_ROOT
            / f"data/kaggle-submission-{experiment_id}-epoch{completed_epoch}"
        ).resolve()
    )
    resolved_state = (
        state_path.expanduser().resolve()
        if state_path is not None
        else artifact_dir / f"milestone-submission-epoch-{completed_epoch:04d}.json"
    )
    resolved_output = (
        kernel_output_dir.expanduser().resolve()
        if kernel_output_dir is not None
        else resolved_bundle / "kernel-output"
    )
    variant = str(milestone["variant"])
    profile = str(milestone["postprocess_profile"])
    message = f"{experiment_id} e{completed_epoch} {variant} {profile}"
    return MilestonePlan(
        config_path=config_path,
        experiment_id=experiment_id,
        completed_epoch=completed_epoch,
        checkpoint=(artifact_dir / str(milestone["checkpoint"])).resolve(),
        variant=variant,
        dataset_id=str(milestone["dataset_id"]),
        kernel_id=str(milestone["kernel_id"]),
        postprocess_profile=profile,
        bundle_root=resolved_bundle,
        state_path=resolved_state,
        kernel_output_dir=resolved_output,
        test_dir=_resolve_competition_path(str(data["test_dir"])),
        message=message,
    )


def _root_uv_command(*args: str) -> list[str]:
    return [
        "uv",
        "run",
        "--project",
        str(REPOSITORY_ROOT),
        "--frozen",
        *args,
    ]


def kaggle_environment(*, runner: Runner = subprocess.run) -> dict[str, str]:
    """Return inherited auth, securely hydrating the cached OAuth access token."""
    environment = os.environ.copy()
    if environment.get("KAGGLE_API_TOKEN"):
        return environment
    command = _root_uv_command("kaggle", "auth", "print-access-token")
    try:
        result = runner(
            command,
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Kaggle CLI is unavailable for OAuth token hydration") from error
    token = result.stdout.strip() if result.returncode == 0 else ""
    if not token:
        # Never include stdout: a malformed CLI response could still contain credential material.
        detail = result.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"could not obtain a Kaggle OAuth access token{suffix}")
    environment["KAGGLE_API_TOKEN"] = token
    return environment


def packager_command(plan: MilestonePlan) -> list[str]:
    return [
        sys.executable,
        str(PACKAGER),
        "--config",
        str(plan.config_path),
        "--completed-epoch",
        str(plan.completed_epoch),
        "--output-root",
        str(plan.bundle_root),
        "--generate-only",
    ]


def packager_publish_command(plan: MilestonePlan) -> list[str]:
    command = packager_command(plan)
    command[-1] = "--publish"
    return command


def dataset_publish_command(plan: MilestonePlan) -> list[str]:
    return _root_uv_command(
        "kaggle", "datasets", "create", "-p", str(plan.dataset_dir)
    )


def kernel_publish_command(plan: MilestonePlan) -> list[str]:
    return _root_uv_command("kaggle", "kernels", "push", "-p", str(plan.kernel_dir))


def kernel_status_command(plan: MilestonePlan) -> list[str]:
    return _root_uv_command("kaggle", "kernels", "status", plan.kernel_id)


def kernel_output_command(plan: MilestonePlan) -> list[str]:
    return _root_uv_command(
        "kaggle",
        "kernels",
        "output",
        plan.kernel_id,
        "-p",
        str(plan.kernel_output_dir),
        "--force",
        "--quiet",
        "--file-pattern",
        r"(^|/)submission\.csv$",
    )


def leaderboard_submit_command(
    plan: MilestonePlan, submission_csv: Path, *, timeout: float, poll_interval: float
) -> list[str]:
    return _root_uv_command(
        "kaggle-lb",
        "submit",
        COMPETITION_SLUG,
        "--file",
        str(submission_csv),
        "--experiment-id",
        plan.experiment_id,
        "--message",
        plan.message,
        "--timeout",
        str(timeout),
        "--poll-interval",
        str(poll_interval),
    )


def _run_command(
    command: Sequence[str],
    *,
    runner: Runner,
    accepted_returncodes: frozenset[int] = frozenset({0}),
    echo_stdout: bool = True,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, Any] = {
        "cwd": REPOSITORY_ROOT,
        "text": True,
        "capture_output": True,
        "check": False,
    }
    if environment is not None:
        kwargs["env"] = dict(environment)
    result = runner(list(command), **kwargs)
    if echo_stdout and result.stdout.strip():
        print(result.stdout.rstrip(), flush=True)
    if result.returncode not in accepted_returncodes:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: "
            f"{' '.join(command)}\n{detail}"
        )
    return result


def wait_for_checkpoint(
    checkpoint: Path,
    *,
    enabled: bool,
    timeout: float,
    poll_interval: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if checkpoint.is_file() and checkpoint.stat().st_size > 0:
        return
    if not enabled:
        raise FileNotFoundError(
            f"milestone checkpoint is not ready: {checkpoint}; "
            "pass --wait-for-checkpoint to launch alongside training"
        )
    if timeout <= 0 or poll_interval <= 0:
        raise ValueError("checkpoint timeout and poll interval must be positive")
    deadline = monotonic() + timeout
    print(f"Waiting for checkpoint: {checkpoint}", flush=True)
    while True:
        if checkpoint.is_file() and checkpoint.stat().st_size > 0:
            print(f"Checkpoint ready: {checkpoint}", flush=True)
            return
        if monotonic() >= deadline:
            raise TimeoutError(f"checkpoint wait timed out after {timeout:g}s: {checkpoint}")
        sleep(min(poll_interval, max(0.0, deadline - monotonic())))


def parse_kernel_status(output: str) -> str:
    match = KERNEL_STATUS_PATTERN.search(output)
    if match is None:
        raise RuntimeError(f"could not parse Kaggle kernel status: {output[:300]!r}")
    return match.group(1).strip().lower()


def wait_for_kernel(
    plan: MilestonePlan,
    *,
    timeout: float,
    poll_interval: float,
    runner: Runner,
    environment: Mapping[str, str] | None = None,
    on_status: Callable[[str], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    if timeout <= 0 or poll_interval <= 0:
        raise ValueError("kernel timeout and poll interval must be positive")
    deadline = monotonic() + timeout
    while True:
        result = _run_command(
            kernel_status_command(plan), runner=runner, environment=environment
        )
        status = parse_kernel_status(result.stdout)
        if on_status is not None:
            on_status(status)
        if status in KERNEL_SUCCESS_STATUSES:
            return status
        if status in KERNEL_FAILURE_STATUSES:
            raise RuntimeError(f"Kaggle Notebook ended with status {status!r}")
        if monotonic() >= deadline:
            raise TimeoutError(
                f"Kaggle Notebook did not complete within {timeout:g}s; last status={status!r}"
            )
        sleep(min(poll_interval, max(0.0, deadline - monotonic())))


def _load_biohub_validator() -> Callable[[Path, Sequence[str]], dict[str, int]]:
    spec = importlib.util.spec_from_file_location(
        "biohub_milestone_submission_validator", BIOHUB_VALIDATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Biohub submission validator: {BIOHUB_VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.validate_submission


def local_test_datasets(test_dir: Path) -> tuple[str, ...]:
    if not test_dir.is_dir():
        raise ValueError(f"local test directory does not exist: {test_dir}")
    datasets = tuple(
        sorted(path.name.removesuffix(".zarr") for path in test_dir.glob("*.zarr"))
    )
    if not datasets:
        raise ValueError(f"no local test .zarr datasets found in {test_dir}")
    return datasets


def find_and_validate_submission(
    output_dir: Path, *, expected_datasets: Sequence[str]
) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(path.resolve() for path in output_dir.rglob("submission.csv"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one submission.csv below {output_dir}, found {len(candidates)}"
    )
    submission_csv = candidates[0]
    biohub_metadata = _load_biohub_validator()(
        submission_csv, tuple(sorted(expected_datasets))
    )
    _, manifest = load_manifest(REPOSITORY_ROOT, COMPETITION_SLUG)
    generic_metadata = validate_submission(submission_csv, manifest)
    return submission_csv, {
        **generic_metadata,
        **biohub_metadata,
        "path": str(submission_csv),
    }


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _identity(plan: MilestonePlan) -> dict[str, Any]:
    return {
        "experiment_id": plan.experiment_id,
        "completed_epoch": plan.completed_epoch,
        "checkpoint": str(plan.checkpoint),
        "dataset_id": plan.dataset_id,
        "kernel_id": plan.kernel_id,
        "postprocess_profile": plan.postprocess_profile,
        "bundle_root": str(plan.bundle_root),
        "test_dir": str(plan.test_dir),
        "message": plan.message,
    }


def _new_state(plan: MilestonePlan) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "identity": _identity(plan),
        "package": {},
        "dataset_publish": {},
        "kernel_publish": {},
        "kernel_run": {},
        "output": {},
        "submission": {},
    }


def _load_state(plan: MilestonePlan) -> dict[str, Any]:
    if not plan.state_path.is_file():
        return _new_state(plan)
    state = json.loads(plan.state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported workflow state schema: {plan.state_path}")
    if state.get("identity") != _identity(plan):
        raise ValueError(
            f"workflow state belongs to different milestone inputs: {plan.state_path}"
        )
    return state


def _save_state(plan: MilestonePlan, state: Mapping[str, Any]) -> None:
    plan.state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = plan.state_path.with_name(plan.state_path.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(plan.state_path)


@contextmanager
def _workflow_lock(state_path: Path):
    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - this workflow runs under WSL/Linux
        raise RuntimeError("workflow locking requires a POSIX environment") from error
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_name(state_path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _validate_packaged_bundle(plan: MilestonePlan) -> dict[str, Any]:
    manifest_path = plan.bundle_root / "bundle-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"packager did not write {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": plan.experiment_id,
        "completed_epoch": plan.completed_epoch,
        "dataset_id": plan.dataset_id,
        "kernel_id": plan.kernel_id,
        "postprocess_profile": plan.postprocess_profile,
    }
    mismatches = {
        key: {"expected": value, "found": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"packaged milestone manifest mismatch: {mismatches}")
    for path in (plan.dataset_dir, plan.kernel_dir):
        if not path.is_dir():
            raise ValueError(f"packager output directory is missing: {path}")
    packaged_checkpoint = plan.dataset_dir / "temporal_graph_checkpoint.pth"
    dataset_manifest = plan.dataset_dir / "manifest.json"
    for path in (packaged_checkpoint, dataset_manifest):
        if not path.is_file():
            raise ValueError(f"packager output file is missing: {path}")
    actual_graph_sha256 = _sha256(packaged_checkpoint)
    if actual_graph_sha256 != manifest.get("graph_checkpoint_sha256"):
        raise ValueError("packaged temporal-graph checkpoint SHA-256 mismatch")
    if actual_graph_sha256 != _sha256(plan.checkpoint):
        raise ValueError("packaged temporal-graph checkpoint differs from the milestone source")
    if _sha256(dataset_manifest) != manifest.get("manifest_sha256"):
        raise ValueError("packaged Dataset manifest SHA-256 mismatch")
    return manifest


def _begin_step(plan: MilestonePlan, state: dict[str, Any], key: str) -> bool:
    """Record an intent and return True only for the process that first recorded it."""
    step = state[key]
    if step.get("completed_at"):
        return False
    fresh = not bool(step.get("started_at"))
    if fresh:
        step["started_at"] = _timestamp()
        _save_state(plan, state)
    return fresh


def _finish_step(
    plan: MilestonePlan,
    state: dict[str, Any],
    key: str,
    **details: Any,
) -> None:
    state[key].update(details)
    state[key]["completed_at"] = _timestamp()
    _save_state(plan, state)


def _remote_resource_exists(
    command: Sequence[str],
    *,
    runner: Runner,
    resource_name: str,
    environment: Mapping[str, str] | None = None,
) -> bool:
    kwargs: dict[str, Any] = {
        "cwd": REPOSITORY_ROOT,
        "text": True,
        "capture_output": True,
        "check": False,
    }
    if environment is not None:
        kwargs["env"] = dict(environment)
    result = runner(list(command), **kwargs)
    if result.returncode == 0:
        return True
    detail = (result.stderr or result.stdout).strip().lower()
    missing_markers = (
        "404",
        "not found",
        "could not find",
        "does not exist",
        "no dataset found",
        "no kernel found",
    )
    if any(marker in detail for marker in missing_markers):
        return False
    raise RuntimeError(f"could not recover {resource_name} publish state: {detail}")


def _dataset_status_command(plan: MilestonePlan) -> list[str]:
    return _root_uv_command(
        "kaggle", "datasets", "status", plan.dataset_id, "--format", "json"
    )


def _recover_existing_submission(
    plan: MilestonePlan, *, file_metadata: Mapping[str, Any]
) -> dict[str, Any] | None:
    ledger = COMPETITION_ROOT / "strategy/lb-submissions.jsonl"
    if not ledger.is_file():
        return None
    matches: list[dict[str, Any]] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            record.get("experiment_id") == plan.experiment_id
            and record.get("description") == plan.message
            and record.get("file_sha256") == file_metadata.get("sha256")
            and str(record.get("ref", "")).strip()
        ):
            matches.append(record)
    if not matches:
        return None
    record = matches[-1]
    return {
        "ref": str(record["ref"]),
        "status": str(record.get("status", "")),
        "public_score": record.get("public_score"),
        "private_score": record.get("private_score"),
        "ledger": str(ledger),
        "recovered": True,
    }


def _submissions_command() -> list[str]:
    return _root_uv_command(
        "kaggle",
        "competitions",
        "submissions",
        COMPETITION_SLUG,
        "--format",
        "json",
        "--page-size",
        "200",
        "--quiet",
    )


def _recover_remote_submission(
    plan: MilestonePlan,
    *,
    submission_csv: Path,
    runner: Runner,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Recover an accepted submit after a wrapper interruption, without resubmitting."""
    result = _run_command(
        _submissions_command(),
        runner=runner,
        echo_stdout=False,
        environment=environment,
    )
    output = result.stdout.strip()
    if not output or output == "No submissions found":
        return None
    try:
        submissions = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError("Kaggle returned invalid submission JSON during resume") from error
    if not isinstance(submissions, list):
        raise RuntimeError("Kaggle submission history must be a JSON list")
    matches = [
        item
        for item in submissions
        if isinstance(item, Mapping)
        and str(item.get("description", "")) == plan.message
        and str(item.get("fileName", "")) == submission_csv.name
        and str(item.get("ref", "")).strip()
    ]
    if not matches:
        return None
    item = matches[0]

    def optional(value: Any) -> str | None:
        normalized = str(value).strip() if value is not None else ""
        return None if normalized in {"", "None", "null"} else normalized

    return {
        "ref": str(item["ref"]),
        "status": str(item.get("status", "")),
        "public_score": optional(item.get("publicScore")),
        "private_score": optional(item.get("privateScore")),
        "recovered": True,
        "recovery_source": "kaggle-submission-history",
    }


def _parse_leaderboard_output(output: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    supported = {"ref", "status", "public_score", "private_score", "ledger"}
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in supported:
            normalized = value.strip()
            parsed[key.strip()] = None if normalized == "not_available" else normalized
    if not str(parsed.get("ref", "")).strip():
        raise RuntimeError("kaggle-lb completed without reporting a submission ref")
    return parsed


def execute(
    plan: MilestonePlan,
    *,
    wait_for_checkpoint_enabled: bool,
    checkpoint_timeout: float,
    checkpoint_poll_interval: float,
    kernel_timeout: float,
    kernel_poll_interval: float,
    leaderboard_timeout: float,
    leaderboard_poll_interval: float,
    runner: Runner = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    with _workflow_lock(plan.state_path):
        state = _load_state(plan)
        recorded_ref = str(state["submission"].get("ref", "")).strip()
        if recorded_ref:
            print(
                f"Submission already recorded as ref {recorded_ref}; no external action taken.",
                flush=True,
            )
            return state

        expected_datasets = local_test_datasets(plan.test_dir)

        wait_for_checkpoint(
            plan.checkpoint,
            enabled=wait_for_checkpoint_enabled,
            timeout=checkpoint_timeout,
            poll_interval=checkpoint_poll_interval,
            monotonic=monotonic,
            sleep=sleep,
        )
        checkpoint_sha256 = _sha256(plan.checkpoint)
        recorded_checkpoint = state["package"].get("checkpoint_sha256")
        if recorded_checkpoint and recorded_checkpoint != checkpoint_sha256:
            raise ValueError(
                "milestone checkpoint bytes changed after packaging; refusing to republish"
            )
        environment = kaggle_environment(runner=runner)

        try:
            packaged_manifest = _validate_packaged_bundle(plan)
        except (OSError, ValueError, json.JSONDecodeError):
            packaged_manifest = None
        if not state["package"].get("completed_at") or packaged_manifest is None:
            if state["dataset_publish"].get("completed_at"):
                raise RuntimeError(
                    "local package is missing after Dataset publication; restore the bundle "
                    "instead of silently changing published bytes"
                )
            _begin_step(plan, state, "package")
            _run_command(
                packager_command(plan), runner=runner, environment=environment
            )
            packaged_manifest = _validate_packaged_bundle(plan)
            _finish_step(
                plan,
                state,
                "package",
                checkpoint_sha256=checkpoint_sha256,
                graph_checkpoint_sha256=packaged_manifest["graph_checkpoint_sha256"],
                manifest_sha256=packaged_manifest["manifest_sha256"],
            )

        if (
            not state["dataset_publish"].get("completed_at")
            and not state["kernel_publish"].get("completed_at")
            and not state["dataset_publish"].get("started_at")
            and not state["kernel_publish"].get("started_at")
        ):
            _begin_step(plan, state, "dataset_publish")
            _begin_step(plan, state, "kernel_publish")
            _run_command(
                packager_publish_command(plan),
                runner=runner,
                environment=environment,
            )
            _finish_step(plan, state, "dataset_publish", recovered=False)
            _finish_step(plan, state, "kernel_publish", recovered=False)

        if not state["dataset_publish"].get("completed_at"):
            fresh = _begin_step(plan, state, "dataset_publish")
            recovered = False
            if not fresh:
                recovered = _remote_resource_exists(
                    _dataset_status_command(plan),
                    runner=runner,
                    resource_name=f"Dataset {plan.dataset_id}",
                    environment=environment,
                )
            if not recovered:
                _run_command(
                    dataset_publish_command(plan),
                    runner=runner,
                    environment=environment,
                )
            _finish_step(plan, state, "dataset_publish", recovered=recovered)

        if not state["kernel_publish"].get("completed_at"):
            fresh = _begin_step(plan, state, "kernel_publish")
            recovered = False
            if not fresh:
                recovered = _remote_resource_exists(
                    kernel_status_command(plan),
                    runner=runner,
                    resource_name=f"Notebook {plan.kernel_id}",
                    environment=environment,
                )
            if not recovered:
                _run_command(
                    kernel_publish_command(plan),
                    runner=runner,
                    environment=environment,
                )
            _finish_step(plan, state, "kernel_publish", recovered=recovered)

        if state["kernel_run"].get("status") not in KERNEL_SUCCESS_STATUSES:

            def record_kernel_status(status: str) -> None:
                state["kernel_run"]["status"] = status
                state["kernel_run"]["checked_at"] = _timestamp()
                _save_state(plan, state)

            status = wait_for_kernel(
                plan,
                timeout=kernel_timeout,
                poll_interval=kernel_poll_interval,
                runner=runner,
                environment=environment,
                on_status=record_kernel_status,
                monotonic=monotonic,
                sleep=sleep,
            )
            _finish_step(plan, state, "kernel_run", status=status)

        output_metadata: dict[str, Any] | None = None
        submission_csv: Path | None = None
        if state["output"].get("completed_at"):
            candidate = Path(str(state["output"].get("path", "")))
            if candidate.is_file() and _sha256(candidate) == state["output"].get("sha256"):
                submission_csv, output_metadata = find_and_validate_submission(
                    plan.kernel_output_dir, expected_datasets=expected_datasets
                )
        if submission_csv is None or output_metadata is None:
            fresh = _begin_step(plan, state, "output")
            if not fresh:
                try:
                    submission_csv, output_metadata = find_and_validate_submission(
                        plan.kernel_output_dir, expected_datasets=expected_datasets
                    )
                except (OSError, ValueError):
                    submission_csv = None
                    output_metadata = None
            if submission_csv is None or output_metadata is None:
                plan.kernel_output_dir.mkdir(parents=True, exist_ok=True)
                _run_command(
                    kernel_output_command(plan),
                    runner=runner,
                    environment=environment,
                )
                submission_csv, output_metadata = find_and_validate_submission(
                    plan.kernel_output_dir, expected_datasets=expected_datasets
                )
            _finish_step(plan, state, "output", **output_metadata)

        recovered_submission = _recover_existing_submission(
            plan, file_metadata=output_metadata
        )
        if recovered_submission is not None:
            _finish_step(plan, state, "submission", **recovered_submission)
            return state

        fresh_submission_intent = _begin_step(plan, state, "submission")
        if not fresh_submission_intent:
            recovered_submission = _recover_remote_submission(
                plan,
                submission_csv=submission_csv,
                runner=runner,
                environment=environment,
            )
            if recovered_submission is not None:
                _finish_step(plan, state, "submission", **recovered_submission)
                return state
        command = leaderboard_submit_command(
            plan,
            submission_csv,
            timeout=leaderboard_timeout,
            poll_interval=leaderboard_poll_interval,
        )
        result = _run_command(
            command,
            runner=runner,
            accepted_returncodes=frozenset({0, 2}),
            environment=environment,
        )
        submission = _parse_leaderboard_output(result.stdout)
        _finish_step(plan, state, "submission", **submission)
        if result.returncode != 0:
            raise RuntimeError(
                f"kaggle-lb stopped with status {submission.get('status')!r}; "
                f"submission ref {submission['ref']} was recorded and will not be resubmitted"
            )
        return state


def _command_strings(commands: Sequence[Sequence[str]]) -> list[list[str]]:
    return [[str(part) for part in command] for command in commands]


def dry_run_plan(
    plan: MilestonePlan,
    *,
    wait_for_checkpoint_enabled: bool,
    checkpoint_timeout: float,
    checkpoint_poll_interval: float,
    kernel_timeout: float,
    kernel_poll_interval: float,
    leaderboard_timeout: float,
    leaderboard_poll_interval: float,
) -> dict[str, Any]:
    placeholder_csv = plan.kernel_output_dir / "submission.csv"
    return {
        "mode": "plan",
        "external_mutations": False,
        "milestone": {
            **asdict(plan),
            "config_path": str(plan.config_path),
            "checkpoint": str(plan.checkpoint),
            "bundle_root": str(plan.bundle_root),
            "state_path": str(plan.state_path),
            "kernel_output_dir": str(plan.kernel_output_dir),
            "test_dir": str(plan.test_dir),
            "checkpoint_exists": plan.checkpoint.is_file(),
        },
        "wait_for_checkpoint": {
            "enabled": wait_for_checkpoint_enabled,
            "timeout_seconds": checkpoint_timeout,
            "poll_interval_seconds": checkpoint_poll_interval,
        },
        "kernel_wait": {
            "timeout_seconds": kernel_timeout,
            "poll_interval_seconds": kernel_poll_interval,
        },
        "commands": _command_strings(
            [
                packager_command(plan),
                packager_publish_command(plan),
                kernel_status_command(plan),
                kernel_output_command(plan),
                leaderboard_submit_command(
                    plan,
                    placeholder_csv,
                    timeout=leaderboard_timeout,
                    poll_interval=leaderboard_poll_interval,
                ),
            ]
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--completed-epoch", type=int, choices=(5, 30), required=True)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--kernel-output-dir", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Publish Dataset/Notebook and submit to Kaggle. The default only prints a plan.",
    )
    parser.add_argument(
        "--wait-for-checkpoint",
        action="store_true",
        help="Wait for the configured periodic checkpoint before packaging.",
    )
    parser.add_argument("--checkpoint-timeout", type=float, default=43200)
    parser.add_argument("--checkpoint-poll-interval", type=float, default=30)
    parser.add_argument("--kernel-timeout", type=float, default=43200)
    parser.add_argument("--kernel-poll-interval", type=float, default=60)
    parser.add_argument("--leaderboard-timeout", type=float, default=43200)
    parser.add_argument("--leaderboard-poll-interval", type=float, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_plan(
            args.config,
            args.completed_epoch,
            bundle_root=args.bundle_root,
            state_path=args.state_file,
            kernel_output_dir=args.kernel_output_dir,
        )
        if not args.execute:
            print(
                json.dumps(
                    dry_run_plan(
                        plan,
                        wait_for_checkpoint_enabled=args.wait_for_checkpoint,
                        checkpoint_timeout=args.checkpoint_timeout,
                        checkpoint_poll_interval=args.checkpoint_poll_interval,
                        kernel_timeout=args.kernel_timeout,
                        kernel_poll_interval=args.kernel_poll_interval,
                        leaderboard_timeout=args.leaderboard_timeout,
                        leaderboard_poll_interval=args.leaderboard_poll_interval,
                    ),
                    indent=2,
                    default=str,
                )
            )
            return 0
        state = execute(
            plan,
            wait_for_checkpoint_enabled=args.wait_for_checkpoint,
            checkpoint_timeout=args.checkpoint_timeout,
            checkpoint_poll_interval=args.checkpoint_poll_interval,
            kernel_timeout=args.kernel_timeout,
            kernel_poll_interval=args.kernel_poll_interval,
            leaderboard_timeout=args.leaderboard_timeout,
            leaderboard_poll_interval=args.leaderboard_poll_interval,
        )
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
