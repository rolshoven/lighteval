from dataclasses import dataclass
from typing import Any

from lighteval.harbor.agent_spec import HarborAgentSpec
from lighteval.harbor.execution import HarborExecutionSpec, resolve_harbor_execution


@dataclass(frozen=True)
class ResolvedHarborJobAgent:
    """Backward-compatible view of a custom-import Harbor execution."""

    harbor_agent_import: str
    skills_dir: str | None
    mcp_config_path: str | None
    environment: dict[str, str]


def resolve_job_agent(tasks_dict: dict[str, Any], override_agent_import: str | None = None) -> ResolvedHarborJobAgent:
    execution = resolve_harbor_execution(
        tasks_dict,
        harbor_agent=None,
        override_agent_import=override_agent_import,
    )
    if execution.mode != "custom_import" or not execution.agent_import_path:
        raise ValueError(
            "resolve_job_agent only supports custom-import execution. "
            "Use resolve_harbor_execution for built-in Harbor agents."
        )
    return ResolvedHarborJobAgent(
        harbor_agent_import=execution.agent_import_path,
        skills_dir=execution.skills_dir,
        mcp_config_path=execution.mcp_config_path,
        environment=execution.environment,
    )


def execution_to_resolved(execution: HarborExecutionSpec) -> ResolvedHarborJobAgent | None:
    if execution.mode != "custom_import" or not execution.agent_import_path:
        return None
    return ResolvedHarborJobAgent(
        harbor_agent_import=execution.agent_import_path,
        skills_dir=execution.skills_dir,
        mcp_config_path=execution.mcp_config_path,
        environment=execution.environment,
    )
