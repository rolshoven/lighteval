from dataclasses import dataclass, field


@dataclass(frozen=True)
class HarborAgentSpec:
    """Task-level declaration of how to execute Harbor-backed agents.

    The spec is attached to task configs, allowing each benchmark family to ship
    its own reference agent implementation without coupling global Harbor code to
    any specific benchmark.
    """

    runner_module: str
    runner_callable: str = "main"
    environment: dict[str, str] = field(default_factory=dict)
    harbor_agent_import: str | None = None
    default_harbor_agent: str | None = None
    utility_quality_metrics: tuple[str, ...] | None = None
    skills_dir: str | None = None
    mcp_config_path: str | None = None
