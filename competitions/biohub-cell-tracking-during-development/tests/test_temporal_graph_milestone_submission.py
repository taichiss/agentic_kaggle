"""Focused tests for the restart-safe EXP-0009 Kaggle milestone workflow."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

COMPETITION_ROOT = Path(__file__).parents[1]
SCRIPT = COMPETITION_ROOT / "scripts/run_temporal_graph_milestone_submission.py"
SPEC = importlib.util.spec_from_file_location(
    "temporal_graph_milestone_submission_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
workflow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = workflow
SPEC.loader.exec_module(workflow)


def _config(path: Path, artifact_dir: Path, *, include_epoch_5: bool = True) -> Path:
    milestone_5 = ""
    if include_epoch_5:
        milestone_5 = '''
[[submission.milestones]]
completed_epoch = 5
checkpoint = "checkpoint_epoch_0005.pth"
variant = "early-lb-probe"
dataset_id = "fixture-owner/biohub-exp-0009-e5"
kernel_id = "fixture-owner/biohub-exp-0009-e5-submit"
postprocess_profile = "public-applicable-v1"
'''
    path.write_text(
        f'''schema_version = 1
experiment_id = "EXP-0009"

[submission]
competition = "biohub-cell-tracking-during-development"
{milestone_5}
[[submission.milestones]]
completed_epoch = 30
checkpoint = "checkpoint_epoch_0030.pth"
variant = "final-30e"
dataset_id = "fixture-owner/biohub-exp-0009-e30"
kernel_id = "fixture-owner/biohub-exp-0009-e30-submit"
postprocess_profile = "public-applicable-v1"

[data]
test_dir = "{(artifact_dir.parent / 'test').as_posix()}"

[output]
artifact_dir = "{artifact_dir.as_posix()}"
''',
        encoding="utf-8",
    )
    return path


def _plan(tmp_path: Path) -> workflow.MilestonePlan:
    artifact_dir = tmp_path / "artifacts"
    (tmp_path / "test/fixture.zarr").mkdir(parents=True)
    config = _config(tmp_path / "experiment.toml", artifact_dir)
    return workflow.build_plan(
        config,
        5,
        bundle_root=tmp_path / "bundle",
        state_path=tmp_path / "state.json",
        kernel_output_dir=tmp_path / "output",
    )


def _completed(command: list[str], stdout: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")


def _write_submission(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "id,dataset,row_type,node_id,t,z,y,x,source_id,target_id\n"
        "0,fixture,node,1,0,0,0,0,-1,-1\n",
        encoding="utf-8",
    )


class SuccessfulRunner:
    def __init__(self, plan: workflow.MilestonePlan) -> None:
        self.plan = plan
        self.calls: list[list[str]] = []
        self.call_kwargs: list[dict] = []
        self.statuses = iter(("RUNNING", "COMPLETE"))

    def __call__(self, command, **kwargs):
        command = list(command)
        self.calls.append(command)
        self.call_kwargs.append(kwargs)
        if "auth" in command and "print-access-token" in command:
            return _completed(command, "fixture-oauth-token\n")
        if str(workflow.PACKAGER) in command:
            self.plan.dataset_dir.mkdir(parents=True, exist_ok=True)
            self.plan.kernel_dir.mkdir(parents=True, exist_ok=True)
            packaged_checkpoint = (
                self.plan.dataset_dir / "temporal_graph_checkpoint.pth"
            )
            packaged_checkpoint.write_bytes(self.plan.checkpoint.read_bytes())
            dataset_manifest = self.plan.dataset_dir / "manifest.json"
            dataset_manifest.write_text('{"fixture": true}\n', encoding="utf-8")
            (self.plan.bundle_root / "bundle-manifest.json").write_text(
                json.dumps(
                    {
                        "experiment_id": "EXP-0009",
                        "completed_epoch": 5,
                        "dataset_id": self.plan.dataset_id,
                        "kernel_id": self.plan.kernel_id,
                        "postprocess_profile": "public-applicable-v1",
                        "graph_checkpoint_sha256": workflow._sha256(
                            packaged_checkpoint
                        ),
                        "manifest_sha256": workflow._sha256(dataset_manifest),
                    }
                ),
                encoding="utf-8",
            )
            return _completed(command)
        if "datasets" in command and "create" in command:
            return _completed(command, "Dataset created\n")
        if "kernels" in command and "push" in command:
            return _completed(command, "Kernel version 1 successfully pushed\n")
        if "kernels" in command and "status" in command:
            status = next(self.statuses)
            return _completed(
                command, f'{self.plan.kernel_id} has status "{status}"\n'
            )
        if "kernels" in command and "output" in command:
            _write_submission(self.plan.kernel_output_dir / "submission.csv")
            return _completed(command, "Output downloaded\n")
        if "kaggle-lb" in command and "submit" in command:
            return _completed(
                command,
                "ref: 123456\n"
                "status: COMPLETE\n"
                "public_score: 0.901\n"
                "private_score: not_available\n"
                f"ledger: {COMPETITION_ROOT / 'strategy/lb-submissions.jsonl'}\n",
            )
        raise AssertionError(f"unexpected command: {command}")


def _execute(plan: workflow.MilestonePlan, runner, **overrides):
    arguments = {
        "wait_for_checkpoint_enabled": False,
        "checkpoint_timeout": 10,
        "checkpoint_poll_interval": 1,
        "kernel_timeout": 10,
        "kernel_poll_interval": 1,
        "leaderboard_timeout": 10,
        "leaderboard_poll_interval": 1,
        "runner": runner,
        "sleep": lambda _seconds: None,
    }
    arguments.update(overrides)
    return workflow.execute(plan, **arguments)


def test_default_cli_is_a_side_effect_free_plan(tmp_path: Path, capsys) -> None:
    plan = _plan(tmp_path)
    result = workflow.main(
        [
            "--config",
            str(plan.config_path),
            "--completed-epoch",
            "5",
            "--bundle-root",
            str(plan.bundle_root),
            "--state-file",
            str(plan.state_path),
            "--kernel-output-dir",
            str(plan.kernel_output_dir),
            "--wait-for-checkpoint",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "plan"
    assert payload["external_mutations"] is False
    assert payload["milestone"]["completed_epoch"] == 5
    assert any("kaggle-lb" in command for command in payload["commands"])
    assert not plan.state_path.exists()
    assert not plan.bundle_root.exists()


def test_workflow_packages_publishes_waits_validates_and_submits_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    plan = _plan(tmp_path)
    plan.checkpoint.parent.mkdir(parents=True)
    plan.checkpoint.write_bytes(b"epoch five graph head")
    runner = SuccessfulRunner(plan)

    state = _execute(plan, runner)

    assert state["kernel_run"]["status"] == "complete"
    assert state["output"]["nodes"] == 1
    assert state["output"]["edges"] == 0
    assert state["submission"]["ref"] == "123456"
    assert state["submission"]["public_score"] == "0.901"
    assert sum(
        str(workflow.PACKAGER) in call and "--publish" in call
        for call in runner.calls
    ) == 1
    assert not any("datasets" in call and "create" in call for call in runner.calls)
    assert not any("kernels" in call and "push" in call for call in runner.calls)
    assert sum("kaggle-lb" in call and "submit" in call for call in runner.calls) == 1
    leaderboard_call = next(
        index
        for index, call in enumerate(runner.calls)
        if "kaggle-lb" in call and "submit" in call
    )
    assert (
        runner.call_kwargs[leaderboard_call]["env"]["KAGGLE_API_TOKEN"]
        == "fixture-oauth-token"
    )
    assert "fixture-oauth-token" not in plan.state_path.read_text(encoding="utf-8")

    def forbidden_runner(command, **_kwargs):
        raise AssertionError(f"resume attempted an external command: {command}")

    resumed = _execute(plan, forbidden_runner)
    assert resumed["submission"]["ref"] == "123456"


def test_resume_recovers_dataset_created_before_state_commit(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    plan = _plan(tmp_path)
    plan.checkpoint.parent.mkdir(parents=True)
    plan.checkpoint.write_bytes(b"epoch five graph head")
    first_runner = SuccessfulRunner(plan)

    def fail_after_dataset_create(command, **kwargs):
        result = first_runner(command, **kwargs)
        if str(workflow.PACKAGER) in command and "--publish" in command:
            return _completed(list(command), "upload interrupted", returncode=1)
        return result

    with pytest.raises(RuntimeError, match="command failed"):
        _execute(plan, fail_after_dataset_create)

    resumed_runner = SuccessfulRunner(plan)
    original = resumed_runner.__call__
    kernel_pushed = False

    def recover_runner(command, **kwargs):
        nonlocal kernel_pushed
        command = list(command)
        if "datasets" in command and "status" in command:
            resumed_runner.calls.append(command)
            return _completed(command, '{"status":"ready","currentVersionNumber":1}\n')
        if "kernels" in command and "status" in command and not kernel_pushed:
            resumed_runner.calls.append(command)
            return subprocess.CompletedProcess(
                command, 1, stdout="", stderr="Kernel not found"
            )
        if "kernels" in command and "push" in command:
            kernel_pushed = True
        return original(command, **kwargs)

    state = _execute(plan, recover_runner)

    assert state["dataset_publish"]["recovered"] is True
    assert not any(
        "datasets" in call and "create" in call for call in resumed_runner.calls
    )
    assert sum(
        "kernels" in call and "push" in call for call in resumed_runner.calls
    ) == 1
    assert state["submission"]["ref"] == "123456"


def test_resume_recovers_remote_submission_after_wrapper_interruption(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    plan = _plan(tmp_path)
    plan.checkpoint.parent.mkdir(parents=True)
    plan.checkpoint.write_bytes(b"epoch five graph head")
    first_runner = SuccessfulRunner(plan)

    def interrupt_after_submit(command, **kwargs):
        if "kaggle-lb" in command and "submit" in command:
            first_runner.calls.append(list(command))
            return _completed(list(command), "connection interrupted", returncode=1)
        return first_runner(command, **kwargs)

    with pytest.raises(RuntimeError, match="command failed"):
        _execute(plan, interrupt_after_submit)

    calls: list[list[str]] = []

    def recovery_runner(command, **_kwargs):
        command = list(command)
        calls.append(command)
        if "auth" in command and "print-access-token" in command:
            return _completed(command, "fixture-oauth-token\n")
        assert "competitions" in command and "submissions" in command
        return _completed(
            command,
            json.dumps(
                [
                    {
                        "ref": "654321",
                        "fileName": "submission.csv",
                        "description": plan.message,
                        "status": "complete",
                        "publicScore": "0.902",
                        "privateScore": None,
                    }
                ]
            ),
        )

    state = _execute(plan, recovery_runner)

    assert state["submission"]["ref"] == "654321"
    assert state["submission"]["public_score"] == "0.902"
    assert state["submission"]["recovery_source"] == "kaggle-submission-history"
    assert not any("kaggle-lb" in call and "submit" in call for call in calls)


def test_validation_rejects_a_submission_missing_a_local_test_dataset(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    (plan.test_dir / "second-dataset.zarr").mkdir()
    _write_submission(plan.kernel_output_dir / "submission.csv")
    expected = workflow.local_test_datasets(plan.test_dir)

    assert expected == ("fixture", "second-dataset")
    with pytest.raises(ValueError, match="missing test datasets.*second-dataset"):
        workflow.find_and_validate_submission(
            plan.kernel_output_dir, expected_datasets=expected
        )


def test_execute_rejects_an_empty_local_test_inventory_before_external_calls(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    (plan.test_dir / "fixture.zarr").rmdir()

    def forbidden_runner(command, **_kwargs):
        raise AssertionError(f"empty test inventory reached an external command: {command}")

    with pytest.raises(ValueError, match="no local test .zarr datasets"):
        _execute(plan, forbidden_runner)


def test_kaggle_environment_preserves_an_explicit_token(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", "already-configured")

    def forbidden_runner(command, **_kwargs):
        raise AssertionError(f"explicit token unexpectedly triggered auth CLI: {command}")

    environment = workflow.kaggle_environment(runner=forbidden_runner)

    assert environment["KAGGLE_API_TOKEN"] == "already-configured"


def test_wait_for_checkpoint_can_follow_periodic_training_output(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_epoch_0005.pth"
    clock = {"now": 0.0}

    def sleep(seconds: float) -> None:
        clock["now"] += seconds
        checkpoint.write_bytes(b"atomic checkpoint")

    workflow.wait_for_checkpoint(
        checkpoint,
        enabled=True,
        timeout=10,
        poll_interval=1,
        monotonic=lambda: clock["now"],
        sleep=sleep,
    )

    assert checkpoint.read_bytes() == b"atomic checkpoint"


def test_kernel_wait_has_a_bounded_timeout(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    clock = {"now": 0.0}

    def runner(command, **_kwargs):
        return _completed(
            list(command), f'{plan.kernel_id} has status "RUNNING"\n'
        )

    def sleep(seconds: float) -> None:
        clock["now"] += seconds

    with pytest.raises(TimeoutError, match="last status='running'"):
        workflow.wait_for_kernel(
            plan,
            timeout=2,
            poll_interval=1,
            runner=runner,
            monotonic=lambda: clock["now"],
            sleep=sleep,
        )


def test_epoch_must_exist_as_a_configured_exp0009_milestone(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    config = _config(
        tmp_path / "experiment.toml", artifact_dir, include_epoch_5=False
    )

    with pytest.raises(ValueError, match="exactly one configured milestone"):
        workflow.build_plan(config, 5)
    with pytest.raises(ValueError, match="5 or 30"):
        workflow.build_plan(config, 10)
