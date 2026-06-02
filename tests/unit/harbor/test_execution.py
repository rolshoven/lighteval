from types import SimpleNamespace

import pytest

from lighteval.harbor.agent_spec import HarborAgentSpec
from lighteval.harbor.execution import build_harbor_run_command, resolve_harbor_execution


def _task(spec: HarborAgentSpec | None):
    return SimpleNamespace(config=SimpleNamespace(harbor_agent=spec))


def test_resolve_default_uses_task_import():
    spec = HarborAgentSpec(
        runner_module="lighteval.tasks.multilingual.tasks.swiss_legal.lexam_subprocess_runner",
        harbor_agent_import="pkg.agent:LexamHarborAgent",
    )
    execution = resolve_harbor_execution({"lexam_mcq_4:en|0": _task(spec)})
    assert execution.mode == "custom_import"
    assert execution.agent_import_path == "pkg.agent:LexamHarborAgent"


def test_resolve_builtin_agent_terminus():
    spec = HarborAgentSpec(
        runner_module="module.path",
        harbor_agent_import="pkg.agent:LexamHarborAgent",
    )
    execution = resolve_harbor_execution({"lexam_mcq_4:en|0": _task(spec)}, harbor_agent="terminus-2")
    assert execution.mode == "builtin_agent"
    assert execution.harbor_agent == "terminus-2"
    assert execution.agent_import_path is None


def test_resolve_default_agent_alias_uses_task_import():
    spec = HarborAgentSpec(
        runner_module="module.path",
        harbor_agent_import="pkg.agent:LexamHarborAgent",
        default_harbor_agent="my-agent",
    )
    execution = resolve_harbor_execution({"task_a:en|0": _task(spec)}, harbor_agent="my-agent")
    assert execution.mode == "custom_import"
    assert execution.agent_import_path == "pkg.agent:LexamHarborAgent"


def test_resolve_import_override():
    spec = HarborAgentSpec(
        runner_module="module.path",
        harbor_agent_import="pkg.agent:LexamHarborAgent",
    )
    execution = resolve_harbor_execution(
        {"lexam_mcq_4:en|0": _task(spec)},
        override_agent_import="other.agent:OtherAgent",
    )
    assert execution.agent_import_path == "other.agent:OtherAgent"


def test_resolve_colon_agent_string_is_custom_import():
    spec = HarborAgentSpec(
        runner_module="module.path",
        harbor_agent_import="pkg.agent:LexamHarborAgent",
    )
    execution = resolve_harbor_execution(
        {"lexam_mcq_4:en|0": _task(spec)},
        harbor_agent="custom.agent:CustomAgent",
    )
    assert execution.mode == "custom_import"
    assert execution.agent_import_path == "custom.agent:CustomAgent"


def test_build_harbor_run_command_builtin_vs_import():
    builtin = resolve_harbor_execution(
        {"t": _task(HarborAgentSpec(runner_module="m", harbor_agent_import="pkg.a:A"))},
        harbor_agent="terminus-2",
    )
    custom = resolve_harbor_execution(
        {"t": _task(HarborAgentSpec(runner_module="m", harbor_agent_import="pkg.a:A"))},
    )

    builtin_cmd = build_harbor_run_command(
        task_path="/tmp/task",
        execution=builtin,
        harbor_model="openai/gpt-4o-mini",
        harbor_env="docker",
    )
    custom_cmd = build_harbor_run_command(
        task_path="/tmp/task",
        execution=custom,
        harbor_model=None,
        harbor_env=None,
    )

    assert builtin_cmd == [
        "harbor",
        "run",
        "-p",
        "/tmp/task",
        "-a",
        "terminus-2",
        "--model",
        "openai/gpt-4o-mini",
        "--env",
        "docker",
    ]
    assert custom_cmd == [
        "harbor",
        "run",
        "-p",
        "/tmp/task",
        "--agent-import-path",
        "pkg.a:A",
    ]


def test_resolve_rejects_multiple_import_paths():
    first = HarborAgentSpec(runner_module="a", harbor_agent_import="pkg.a:Agent")
    second = HarborAgentSpec(runner_module="b", harbor_agent_import="pkg.b:Agent")
    with pytest.raises(ValueError, match="one custom agent import"):
        resolve_harbor_execution({"task_a": _task(first), "task_b": _task(second)})


def test_resolve_rejects_multiple_default_harbor_agents():
    first = HarborAgentSpec(
        runner_module="a",
        harbor_agent_import="pkg.a:Agent",
        default_harbor_agent="agent-a",
    )
    second = HarborAgentSpec(
        runner_module="b",
        harbor_agent_import="pkg.a:Agent",
        default_harbor_agent="agent-b",
    )
    with pytest.raises(ValueError, match="default_harbor_agent"):
        resolve_harbor_execution({"task_a": _task(first), "task_b": _task(second)})
