from types import SimpleNamespace

import pytest

from lighteval.harbor.agent_spec import HarborAgentSpec
from lighteval.harbor.resolve import resolve_job_agent


def _task(spec: HarborAgentSpec | None):
    return SimpleNamespace(config=SimpleNamespace(harbor_agent=spec))


def test_resolve_job_agent_happy_path():
    spec = HarborAgentSpec(
        runner_module="module.path",
        harbor_agent_import="pkg.agent:LexamHarborAgent",
        skills_dir="/tmp/skills",
        environment={"FOO": "bar"},
    )
    resolved = resolve_job_agent({"lexam_mcq_4:en|0": _task(spec)})
    assert resolved.harbor_agent_import == "pkg.agent:LexamHarborAgent"
    assert resolved.skills_dir == "/tmp/skills"
    assert resolved.environment == {"FOO": "bar"}


def test_resolve_job_agent_requires_harbor_spec():
    with pytest.raises(ValueError, match="does not define `harbor_agent`"):
        resolve_job_agent({"lexam_mcq_4:en|0": _task(None)})


def test_resolve_job_agent_rejects_multiple_import_paths():
    first = HarborAgentSpec(runner_module="a", harbor_agent_import="pkg.a:Agent")
    second = HarborAgentSpec(runner_module="b", harbor_agent_import="pkg.b:Agent")
    with pytest.raises(ValueError, match="one custom agent import"):
        resolve_job_agent({"task_a": _task(first), "task_b": _task(second)})
