from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from lighteval.harbor.agent_spec import HarborAgentSpec
from lighteval.harbor.job_runner import harbor_task_path_for_run, run_harbor_jobs_for_docs


def test_harbor_task_path_for_run_relative_to_job_dir(tmp_path):
    job_dir = tmp_path / "results" / "harbor" / "job"
    task_dir = job_dir / "tasks" / "lexam_mcq_4__en__0__463"
    task_dir.mkdir(parents=True)

    assert harbor_task_path_for_run(task_dir, job_dir) == "tasks/lexam_mcq_4__en__0__463"
    assert harbor_task_path_for_run(task_dir.resolve(), job_dir) == "tasks/lexam_mcq_4__en__0__463"


def _task(spec: HarborAgentSpec):
    return SimpleNamespace(config=SimpleNamespace(harbor_agent=spec))


@patch("lighteval.harbor.job_runner.ingest_trial_outputs")
@patch("lighteval.harbor.job_runner.scaffold_harbor_tasks_from_docs")
@patch("lighteval.harbor.job_runner.subprocess.run")
def test_job_uses_agent_import_path_for_lexam(mock_run, mock_scaffold, mock_ingest, tmp_path):
    job_dir = tmp_path / "job"
    mock_scaffold.return_value = [job_dir / "tasks" / "sample-1"]
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    mock_ingest.return_value = [object()]

    spec = HarborAgentSpec(
        runner_module="lighteval.tasks.multilingual.tasks.swiss_legal.lexam_subprocess_runner",
        harbor_agent_import="pkg.agent:LexamHarborAgent",
    )
    doc = SimpleNamespace(
        task_name="lexam_mcq_4:en|0",
        id="1",
        query="Q",
        choices=["A", "B"],
        gold_index=0,
        specific={},
    )
    run_harbor_jobs_for_docs(
        tasks_dict={"lexam_mcq_4:en|0": _task(spec)},
        documents_dict={"lexam_mcq_4:en|0": [doc]},
        harbor_job_dir=job_dir,
        harbor_agent=None,
        harbor_agent_import=None,
        harbor_model=None,
        harbor_env=None,
        harbor_timeout_seconds=30.0,
    )

    command = mock_run.call_args.args[0]
    assert command[2:4] == ["-p", "tasks/sample-1"]
    assert mock_run.call_args.kwargs["cwd"] == str(job_dir)
    assert "--agent-import-path" in command
    assert "pkg.agent:LexamHarborAgent" in command
    assert "-a" not in command


@patch("lighteval.harbor.job_runner.ingest_trial_outputs")
@patch("lighteval.harbor.job_runner.scaffold_harbor_tasks_from_docs")
@patch("lighteval.harbor.job_runner.subprocess.run")
def test_job_uses_builtin_agent_flag(mock_run, mock_scaffold, mock_ingest, tmp_path):
    job_dir = tmp_path / "job"
    mock_scaffold.return_value = [job_dir / "tasks" / "sample-1"]
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    mock_ingest.return_value = [object()]

    spec = HarborAgentSpec(
        runner_module="lighteval.tasks.multilingual.tasks.swiss_legal.lexam_subprocess_runner",
        harbor_agent_import="pkg.agent:LexamHarborAgent",
    )
    doc = SimpleNamespace(
        task_name="lexam_mcq_4:en|0",
        id="1",
        query="Q",
        choices=["A", "B"],
        gold_index=0,
        specific={},
    )
    run_harbor_jobs_for_docs(
        tasks_dict={"lexam_mcq_4:en|0": _task(spec)},
        documents_dict={"lexam_mcq_4:en|0": [doc]},
        harbor_job_dir=job_dir,
        harbor_agent="terminus-2",
        harbor_agent_import=None,
        harbor_model="openai/gpt-4o-mini",
        harbor_env=None,
        harbor_timeout_seconds=30.0,
    )

    command = mock_run.call_args.args[0]
    assert command[:4] == ["harbor", "run", "-p", "tasks/sample-1"]
    assert "-a" in command
    assert "terminus-2" in command
    assert "--agent-import-path" not in command
