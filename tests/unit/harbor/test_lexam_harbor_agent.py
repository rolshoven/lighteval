import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from harbor.models.agent.context import AgentContext

from lighteval.harbor import agent_skills
from lighteval.tasks.multilingual.tasks.swiss_legal.lexam_harbor_agent import LexamHarborAgent
from lighteval.tasks.multilingual.tasks.swiss_legal.lexam_harbor_common import ensure_mcq_final_answer_format

FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "skills"
# Production skills dir: only optional domain skills (opencaselaw). LEXam exam rules are in the system prompt.
HARBOR_SKILLS_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src/lighteval/tasks/multilingual/tasks/swiss_legal/harbor_skills"
)


def test_mcq_output_is_forced_to_final_answer_format():
    text = ensure_mcq_final_answer_format(
        task_name="lexam_mcq_4:en|0",
        text="I believe it is B.",
        choices=["A", "B", "C", "D"],
    )
    assert "Final Answer:" in text
    assert "###B###" in text


def test_oq_output_not_modified_by_mcq_formatter():
    text = ensure_mcq_final_answer_format(
        task_name="lexam_oq:en|0",
        text="A free-form legal analysis.",
        choices=["ignored"],
    )
    assert text == "A free-form legal analysis."


def test_lexam_harbor_agent_has_stable_name():
    assert LexamHarborAgent.name() == "lexam-harbor-agent"


def test_lexam_harbor_agent_supports_atif():
    assert LexamHarborAgent.SUPPORTS_ATIF is True


@patch("lighteval.tasks.multilingual.tasks.swiss_legal.lexam_harbor_agent._litellm_completion")
def test_run_agent_loop_includes_skills_catalog(mock_completion):
    mock_message = MagicMock()
    mock_message.content = "Final Answer: ###A###"
    mock_message.tool_calls = []
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=mock_message)],
    )

    agent = LexamHarborAgent(skills_dir=HARBOR_SKILLS_ROOT)
    sample = {"task_name": "lexam_mcq_4:en", "choices": ["A", "B"], "prompt": "Pick A or B."}
    environment = AsyncMock()

    final_text, _, _ = asyncio.run(
        agent._run_agent_loop(sample=sample, model_name="openai/gpt-4o-mini", environment=environment)
    )

    assert final_text == "Final Answer: ###A###"
    system_message = mock_completion.call_args.kwargs["messages"][0]["content"]
    assert "LEXam exam answering" in system_message
    assert "Final Answer: ###X###" in system_message
    assert "<name>opencaselaw</name>" in system_message
    assert "<name>lexam</name>" not in system_message
    tool_names = [tool["function"]["name"] for tool in mock_completion.call_args.kwargs["tools"]]
    assert agent_skills.ACTIVATE_SKILL_TOOL in tool_names
    activate_schema = next(
        tool
        for tool in mock_completion.call_args.kwargs["tools"]
        if tool["function"]["name"] == agent_skills.ACTIVATE_SKILL_TOOL
    )
    assert activate_schema["function"]["parameters"]["properties"]["name"]["enum"] == ["opencaselaw"]


@patch("lighteval.tasks.multilingual.tasks.swiss_legal.lexam_harbor_agent._litellm_completion")
def test_run_agent_loop_activate_skill_injects_body(mock_completion):
    first_message = MagicMock()
    first_message.content = ""
    activate_call = MagicMock()
    activate_call.id = "call-activate"
    activate_call.function.name = agent_skills.ACTIVATE_SKILL_TOOL
    activate_call.function.arguments = json.dumps({"name": "opencaselaw"})
    first_message.tool_calls = [activate_call]

    second_message = MagicMock()
    second_message.content = "Final Answer: ###B###"
    second_message.tool_calls = []

    mock_completion.side_effect = [
        MagicMock(choices=[MagicMock(message=first_message)]),
        MagicMock(choices=[MagicMock(message=second_message)]),
    ]

    agent = LexamHarborAgent(skills_dir=HARBOR_SKILLS_ROOT)
    sample = {"task_name": "lexam_mcq_4:en", "choices": ["A", "B"], "prompt": "Question?"}
    environment = AsyncMock()

    asyncio.run(agent._run_agent_loop(sample=sample, model_name="openai/gpt-4o-mini", environment=environment))

    injected = [message for message in mock_completion.call_args_list[1].kwargs["messages"] if message["role"] == "user"]
    assert any("OpenCaseLaw.ch" in message["content"] for message in injected)


@patch("lighteval.tasks.multilingual.tasks.swiss_legal.lexam_harbor_agent._litellm_completion")
def test_run_agent_loop_writes_trajectory_when_logs_dir_set(mock_completion, tmp_path):
    mock_message = MagicMock()
    mock_message.content = "done"
    mock_message.tool_calls = []
    mock_completion.return_value = MagicMock(choices=[MagicMock(message=mock_message)])

    agent = LexamHarborAgent(logs_dir=tmp_path, skills_dir=HARBOR_SKILLS_ROOT)
    sample = {"task_name": "lexam_oq:en", "choices": [], "prompt": "Explain."}
    environment = AsyncMock()

    asyncio.run(agent._run_agent_loop(sample=sample, model_name="openai/gpt-4o-mini", environment=environment))

    trajectory_path = tmp_path / "trajectory.json"
    assert trajectory_path.exists()
    trajectory = json.loads(trajectory_path.read_text())
    assert trajectory["schema_version"].startswith("ATIF-v")
    assert trajectory["agent"]["name"] == "lexam-harbor-agent"
    assert len(trajectory["steps"]) >= 2


@patch.object(LexamHarborAgent, "_run_agent_loop", new_callable=AsyncMock)
def test_lexam_harbor_agent_writes_answer_via_environment(mock_loop):
    mock_loop.return_value = (
        "Final Answer: ###A###",
        {"tool_calls": 1.0, "tokens_in": 42.0, "tokens_out": 7.0, "wall_time_sec": 1.5},
        ["reasoning"],
    )

    agent = LexamHarborAgent()
    agent.model_name = "openai/gpt-4o-mini"
    environment = AsyncMock()
    environment.exec = AsyncMock(return_value=type("ExecResult", (), {"stdout": "", "stderr": ""})())
    context = AgentContext()

    asyncio.run(agent.run("Answer the question.", environment, context))

    assert mock_loop.await_count == 1
    assert environment.exec.await_count >= 2
    assert any("answer.json" in str(call.args[0]) for call in environment.exec.await_args_list)
    assert context.n_input_tokens == 42
    assert context.n_output_tokens == 7
    assert context.metadata is not None
    assert context.metadata["tool_calls"] == 1.0
    assert context.metadata["wall_time_sec"] >= 0.0
