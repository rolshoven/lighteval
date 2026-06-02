"""Harbor sandbox job orchestration (no CLI dependencies)."""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from lighteval.harbor.execution import HarborExecutionSpec, build_harbor_run_command, resolve_harbor_execution
from lighteval.harbor.ingest import ingest_trial_outputs
from lighteval.harbor.scaffold import scaffold_harbor_tasks_from_docs


def harbor_task_path_for_run(task_path: Path, harbor_job_dir: Path) -> str:
    """Path for `harbor run -p` when the subprocess cwd is ``harbor_job_dir``."""
    resolved_task = task_path.resolve()
    resolved_job = harbor_job_dir.resolve()
    try:
        return str(resolved_task.relative_to(resolved_job))
    except ValueError:
        return str(resolved_task)


def _harbor_run_error_message(harbor_job_dir: Path, stdout: str) -> str | None:
    """Return a helpful message when ``harbor run`` finished but trials errored."""
    result_path: Path | None = None
    match = re.search(r"Results written to (jobs/[^\s]+)/result\.json", stdout)
    if match:
        candidate = harbor_job_dir / match.group(1) / "result.json"
        if candidate.exists():
            result_path = candidate

    if result_path is None:
        jobs_dir = harbor_job_dir / "jobs"
        if not jobs_dir.is_dir():
            return None
        job_runs = sorted(
            (p for p in jobs_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for job_run in job_runs:
            candidate = job_run / "result.json"
            if candidate.exists():
                result_path = candidate
                break

    if result_path is None:
        return None

    try:
        payload = json.loads(result_path.read_text())
    except json.JSONDecodeError:
        return None

    stats = payload.get("stats", {})
    if stats.get("n_errored_trials", 0) == 0:
        return None

    details: list[str] = []
    for trial_config in sorted(result_path.parent.rglob("config.json")):
        trial_dir = trial_config.parent
        if trial_dir == result_path.parent:
            continue
        exception_path = trial_dir / "exception.txt"
        if exception_path.exists():
            details.append(f"{trial_dir.name}:\n{exception_path.read_text().strip()}")

    summary = f"Harbor job reported {stats.get('n_errored_trials', 0)} errored trial(s). See {result_path}."
    if details:
        return f"{summary}\n\n" + "\n\n".join(details)
    return summary


def apply_harbor_execution_env(base_env: dict[str, str], execution: HarborExecutionSpec) -> None:
    base_env.update(execution.environment)
    if execution.skills_dir:
        base_env["LIGHTEVAL_HARBOR_SKILLS_DIR"] = execution.skills_dir
        base_env["LIGHEVAL_HARBOR_SKILLS_DIR"] = execution.skills_dir
    if execution.mcp_config_path:
        base_env["LIGHTEVAL_HARBOR_MCP_CONFIG_PATH"] = execution.mcp_config_path
        base_env["LIGHEVAL_HARBOR_MCP_CONFIG_PATH"] = execution.mcp_config_path


def run_harbor_jobs_for_docs(
    *,
    tasks_dict: dict[str, Any],
    documents_dict: dict[str, list[Any]],
    harbor_job_dir: Path,
    harbor_agent: str | None,
    harbor_agent_import: str | None,
    harbor_model: str | None,
    harbor_env: str | None,
    harbor_timeout_seconds: float,
) -> Path:
    execution = resolve_harbor_execution(
        tasks_dict,
        harbor_agent=harbor_agent,
        override_agent_import=harbor_agent_import,
    )
    tasks_dir = harbor_job_dir / "tasks"
    logs_dir = harbor_job_dir / "logs"
    answers_dir = harbor_job_dir / "answers"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)

    task_paths = scaffold_harbor_tasks_from_docs(documents_dict, tasks_dir)
    base_env = os.environ.copy()
    apply_harbor_execution_env(base_env, execution)

    for idx, task_path in enumerate(task_paths):
        command = build_harbor_run_command(
            task_path=harbor_task_path_for_run(Path(task_path), harbor_job_dir),
            execution=execution,
            harbor_model=harbor_model,
            harbor_env=harbor_env,
        )

        process = subprocess.run(
            command,
            cwd=str(harbor_job_dir),
            env=base_env,
            timeout=harbor_timeout_seconds,
            capture_output=True,
            text=True,
        )
        (logs_dir / f"{idx:05d}.stdout.log").write_text(process.stdout)
        (logs_dir / f"{idx:05d}.stderr.log").write_text(process.stderr)
        if process.returncode != 0:
            raise RuntimeError(
                f"Harbor job command failed for {task_path}.\nCommand: {' '.join(command)}\n"
                f"Return code: {process.returncode}\nStdout: {process.stdout}\nStderr: {process.stderr}"
            )
        harbor_error = _harbor_run_error_message(harbor_job_dir, process.stdout)
        if harbor_error:
            raise RuntimeError(harbor_error)

    ingested = ingest_trial_outputs(job_dir=harbor_job_dir, answers_dir=answers_dir)
    if not ingested:
        raise RuntimeError(
            "No Harbor answers were ingested from job outputs. "
            "Please verify your Harbor agent writes answer artifacts compatible with lighteval."
        )
    return answers_dir
