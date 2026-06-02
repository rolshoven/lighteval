"""Resolve Harbor CLI execution mode (built-in agent vs custom import path)."""

from dataclasses import dataclass
from typing import Any, Literal

from lighteval.harbor.agent_spec import HarborAgentSpec


def _task_harbor_spec(task: Any) -> HarborAgentSpec | None:
    config = getattr(task, "config", None)
    if config is None:
        return None
    return getattr(config, "harbor_agent", None)

@dataclass(frozen=True)
class HarborExecutionSpec:
    mode: Literal["builtin_agent", "custom_import"]
    harbor_agent: str | None
    agent_import_path: str | None
    skills_dir: str | None
    mcp_config_path: str | None
    environment: dict[str, str]


def build_harbor_run_command(
    *,
    task_path: str,
    execution: HarborExecutionSpec,
    harbor_model: str | None,
    harbor_env: str | None,
) -> list[str]:
    command = ["harbor", "run", "-p", task_path]
    if execution.mode == "builtin_agent":
        if not execution.harbor_agent:
            raise ValueError("builtin_agent mode requires harbor_agent to be set.")
        command.extend(["-a", execution.harbor_agent])
    else:
        if not execution.agent_import_path:
            raise ValueError("custom_import mode requires agent_import_path to be set.")
        command.extend(["--agent-import-path", execution.agent_import_path])
    if harbor_model:
        command.extend(["--model", harbor_model])
    if harbor_env:
        command.extend(["--env", harbor_env])
    return command


def resolve_harbor_execution(
    tasks_dict: dict[str, Any],
    harbor_agent: str | None = None,
    override_agent_import: str | None = None,
) -> HarborExecutionSpec:
    merged = _merge_task_harbor_specs(tasks_dict)

    if override_agent_import:
        return HarborExecutionSpec(
            mode="custom_import",
            harbor_agent=None,
            agent_import_path=override_agent_import,
            skills_dir=merged.skills_dir,
            mcp_config_path=merged.mcp_config_path,
            environment=merged.environment,
        )

    if harbor_agent and ":" in harbor_agent:
        return HarborExecutionSpec(
            mode="custom_import",
            harbor_agent=None,
            agent_import_path=harbor_agent,
            skills_dir=merged.skills_dir,
            mcp_config_path=merged.mcp_config_path,
            environment=merged.environment,
        )

    if harbor_agent is not None and harbor_agent != merged.default_harbor_agent:
        return HarborExecutionSpec(
            mode="builtin_agent",
            harbor_agent=harbor_agent,
            agent_import_path=None,
            skills_dir=merged.skills_dir,
            mcp_config_path=merged.mcp_config_path,
            environment=merged.environment,
        )

    if not merged.harbor_agent_import:
        raise ValueError(
            "Selected tasks do not define `harbor_agent_import`. "
            "Use `lighteval harbor run` for subprocess mode or pass --harbor-agent for a built-in Harbor agent."
        )

    return HarborExecutionSpec(
        mode="custom_import",
        harbor_agent=None,
        agent_import_path=merged.harbor_agent_import,
        skills_dir=merged.skills_dir,
        mcp_config_path=merged.mcp_config_path,
        environment=merged.environment,
    )


@dataclass(frozen=True)
class _MergedTaskHarborSpec:
    harbor_agent_import: str | None
    default_harbor_agent: str | None
    skills_dir: str | None
    mcp_config_path: str | None
    environment: dict[str, str]


def _merge_task_harbor_specs(tasks_dict: dict[str, Any]) -> _MergedTaskHarborSpec:
    specs_by_task: dict[str, HarborAgentSpec] = {}
    for task_name, task in tasks_dict.items():
        spec = _task_harbor_spec(task)
        if spec is None:
            raise ValueError(
                f"Task '{task_name}' does not define `harbor_agent`. "
                "Set a task-level harbor_agent on the task config."
            )
        specs_by_task[task_name] = spec

    unique_imports = {spec.harbor_agent_import for spec in specs_by_task.values() if spec.harbor_agent_import}
    if len(unique_imports) > 1:
        details = ", ".join(
            f"{task_name}={spec.harbor_agent_import}"
            for task_name, spec in sorted(specs_by_task.items())
            if spec.harbor_agent_import
        )
        raise ValueError(
            "Harbor job mode currently supports one custom agent import per run. "
            f"Found multiple task agent imports: {details}"
        )

    unique_default_agents = {
        spec.default_harbor_agent for spec in specs_by_task.values() if spec.default_harbor_agent
    }
    if len(unique_default_agents) > 1:
        details = ", ".join(
            f"{task_name}={spec.default_harbor_agent}"
            for task_name, spec in sorted(specs_by_task.items())
            if spec.default_harbor_agent
        )
        raise ValueError(
            "Harbor job mode currently supports one default_harbor_agent alias per run. "
            f"Found multiple task default agents: {details}"
        )

    first_spec = next(iter(specs_by_task.values()))
    merged_env: dict[str, str] = {}
    for spec in specs_by_task.values():
        merged_env.update(spec.environment)

    harbor_agent_import = next(iter(unique_imports)) if unique_imports else None
    default_harbor_agent = next(iter(unique_default_agents)) if unique_default_agents else None
    return _MergedTaskHarborSpec(
        harbor_agent_import=harbor_agent_import,
        default_harbor_agent=default_harbor_agent,
        skills_dir=first_spec.skills_dir,
        mcp_config_path=first_spec.mcp_config_path,
        environment=merged_env,
    )
