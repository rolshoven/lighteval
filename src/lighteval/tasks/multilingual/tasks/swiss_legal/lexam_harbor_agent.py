"""Harbor sandbox agent for LEXam legal exam tasks."""

from __future__ import annotations

import json
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from lighteval.harbor import agent_skills
from lighteval.harbor.atif import (
    append_agent_step,
    append_system_step,
    append_user_step,
    new_trajectory,
    set_final_metrics,
    write_trajectory,
)
from lighteval.tasks.multilingual.tasks.swiss_legal.lexam_harbor_common import (
    build_sample_output,
    extract_usage_metrics,
    format_answer_payload,
)
from lighteval.tasks.multilingual.tasks.swiss_legal.lexam_opencaselaw import run_opencaselaw_tool

AGENT_VERSION = "1.0.0"
MAX_AGENT_ITERATIONS = 32
EXECUTE_BASH_TOOL = "execute_bash"
ANSWER_PATH = "/logs/agent/answer.json"

LEXAM_SYSTEM_PROMPT = """You are a LEXam exam answering assistant for Swiss legal exams.

Use structured legal reasoning: clarify facts, identify issues, state applicable rules, apply them to the facts, and conclude clearly.

For multiple-choice tasks (task names starting with `lexam_mcq_`), you must end your response with exactly one line in this form:
Final Answer: ###X###
where X is a single uppercase letter matching one of the provided choices.

For open-ended tasks (`lexam_oq`), provide a complete exam-style legal analysis in the language of the question.

Optional domain skills are listed below. When a skill applies, call activate_skill before relying on it."""


def _litellm_completion(**kwargs: Any) -> object:
    try:
        from litellm import completion
    except ImportError as exc:
        raise RuntimeError(
            "LexamHarborAgent requires `litellm`. Install lighteval with litellm extras."
        ) from exc
    return completion(**kwargs)


def _opencaselaw_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "opencaselaw_search_decisions",
                "description": "Search Swiss court decisions on OpenCaseLaw.ch.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                        "canton": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "opencaselaw_get_doctrine",
                "description": "Fetch statute text and doctrine excerpts from OpenCaseLaw.ch.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "http_get",
                "description": "HTTP GET JSON from allowlisted OpenCaseLaw hosts.",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
        },
    ]


def _execute_bash_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": EXECUTE_BASH_TOOL,
            "description": "Run a bash command in the Harbor task sandbox.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Shell command to run."}},
                "required": ["command"],
            },
        },
    }


class LexamHarborAgent(BaseAgent):
    """LiteLLM agent with LEXam prompts, optional skills, and OpenCaseLaw tools."""

    SUPPORTS_ATIF = True

    def __init__(
        self,
        logs_dir: Path | None = None,
        model_name: str | None = None,
        skills_dir: str | None = None,
        mcp_config_path: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            logs_dir=logs_dir or Path("/tmp/lexam-harbor-agent"),
            model_name=model_name,
            skills_dir=skills_dir,
            **kwargs,
        )
        self._mcp_config_path = mcp_config_path or os.getenv("LIGHTEVAL_HARBOR_MCP_CONFIG_PATH")
        self._skills_root = agent_skills.resolve_skills_dir(skills_dir, self.skills_dir)
        self._skills = agent_skills.discover_skills(self._skills_root) if self._skills_root else []
        self._skills_by_name = {skill.name: skill for skill in self._skills}

    @staticmethod
    def name() -> str:
        return "lexam-harbor-agent"

    def version(self) -> str:
        return AGENT_VERSION

    def _build_system_prompt(self) -> str:
        parts = [LEXAM_SYSTEM_PROMPT, agent_skills.skills_catalog_instructions()]
        catalog = agent_skills.build_skills_catalog(self._skills)
        if catalog:
            parts.append(catalog)
        return "\n\n".join(parts)

    def _build_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = [_execute_bash_tool_schema(), *_opencaselaw_tool_schemas()]
        activate_schema = agent_skills.build_activate_skill_tool_schema(list(self._skills_by_name))
        if activate_schema:
            tools.append(activate_schema)
        return tools

    def _load_metadata(self) -> dict[str, Any]:
        for parent in [self.logs_dir, *self.logs_dir.parents]:
            candidate = parent / "metadata.json"
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
        return {}

    def _sample_from_instruction(self, instruction: str) -> dict[str, Any]:
        metadata = self._load_metadata()
        task_name = str(metadata.get("task_name", ""))
        if not task_name:
            match = re.search(r"from `([^`]+)`", instruction)
            if match:
                task_name = match.group(1)

        prompt = instruction
        if "## Prompt" in instruction:
            prompt = instruction.split("## Prompt", 1)[1].strip()

        return {
            "task_name": task_name,
            "doc_id": str(metadata.get("doc_id", "")),
            "choices": list(metadata.get("choices", [])),
            "prompt": prompt,
        }

    async def setup(self, environment: BaseEnvironment) -> None:
        if self._skills_root:
            await environment.exec(agent_skills.sandbox_skills_copy_shell(str(self._skills_root)))
        if self._mcp_config_path and Path(self._mcp_config_path).is_file():
            mcp_text = Path(self._mcp_config_path).read_text(encoding="utf-8")
            await environment.exec(
                "mkdir -p /workspace && cat > /workspace/mcp.json <<'LEXAM_MCP_EOF'\n"
                f"{mcp_text}\n"
                "LEXAM_MCP_EOF"
            )

    async def _run_bash(self, environment: BaseEnvironment, command: str) -> str:
        result = await environment.exec(command)
        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        exit_code = getattr(result, "exit_code", getattr(result, "returncode", 0))
        payload = {"exit_code": exit_code, "stdout": stdout[:8000], "stderr": stderr[:4000]}
        return json.dumps(payload, indent=2)

    async def _dispatch_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        environment: BaseEnvironment,
        activated: set[str],
        messages: list[dict[str, Any]],
    ) -> str:
        if tool_name == agent_skills.ACTIVATE_SKILL_TOOL:
            return agent_skills.activate_skill_in_messages(
                skill_name=str(arguments.get("name", "")),
                skills_by_name=self._skills_by_name,
                activated=activated,
                messages=messages,
            )
        if tool_name == EXECUTE_BASH_TOOL:
            return await self._run_bash(environment, str(arguments.get("command", "")))
        if tool_name in {"opencaselaw_search_decisions", "opencaselaw_get_doctrine", "http_get"}:
            return run_opencaselaw_tool(tool_name, arguments)
        return json.dumps({"error": "unknown_tool", "tool": tool_name})

    async def _run_agent_loop(
        self,
        *,
        sample: dict[str, Any],
        model_name: str,
        environment: BaseEnvironment,
    ) -> tuple[str, dict[str, float], list[str]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": str(sample["prompt"])},
        ]
        tools = self._build_tools()
        activated: set[str] = set()
        tool_call_count = 0.0
        aggregated_metrics: dict[str, float] = {}
        reasoning: list[str] = []
        trajectory = None
        if self.logs_dir:
            trajectory = new_trajectory(
                agent_name=self.name(),
                agent_version=self.version(),
                model_name=model_name,
            )
            append_system_step(trajectory, messages[0]["content"])
            append_user_step(trajectory, messages[1]["content"])

        for _ in range(MAX_AGENT_ITERATIONS):
            response = _litellm_completion(
                model=model_name,
                messages=messages,
                tools=tools,
            )
            for key, value in extract_usage_metrics(response).items():
                aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + value

            choice = response.choices[0]
            message = choice.message
            content = getattr(message, "content", None) or ""
            tool_calls = getattr(message, "tool_calls", None) or []

            if content.strip():
                reasoning.append(content)

            if trajectory is not None:
                append_agent_step(
                    trajectory,
                    message=content,
                    model_name=model_name,
                    tool_calls=[
                        {
                            "id": call.id,
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        }
                        for call in tool_calls
                    ]
                    or None,
                )

            if not tool_calls:
                if trajectory is not None:
                    set_final_metrics(
                        trajectory,
                        {
                            "prompt_tokens": aggregated_metrics.get("tokens_in", 0.0),
                            "completion_tokens": aggregated_metrics.get("tokens_out", 0.0),
                        },
                    )
                    write_trajectory(self.logs_dir / "trajectory.json", trajectory)
                aggregated_metrics["tool_calls"] = tool_call_count
                return content, aggregated_metrics, reasoning

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ],
            }
            messages.append(assistant_message)

            for call in tool_calls:
                tool_call_count += 1.0
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}

                tool_result = await self._dispatch_tool(
                    tool_name=call.function.name,
                    arguments=arguments,
                    environment=environment,
                    activated=activated,
                    messages=messages,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_result,
                    }
                )

        raise RuntimeError(f"LexamHarborAgent exceeded {MAX_AGENT_ITERATIONS} tool iterations.")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await self.setup(environment)
        sample = self._sample_from_instruction(instruction)
        model_name = (
            self.model_name
            or os.getenv("HARBOR_REFERENCE_MODEL")
            or os.getenv("REFERENCE_MODEL")
            or os.getenv("LITELLM_MODEL")
            or "openai/gpt-4o-mini"
        )

        start = time.perf_counter()
        final_text, metrics, _ = await self._run_agent_loop(
            sample=sample,
            model_name=model_name,
            environment=environment,
        )
        metrics["wall_time_sec"] = time.perf_counter() - start

        output = build_sample_output(sample=sample, text=final_text, metrics=metrics)
        payload = format_answer_payload(output)
        await environment.exec("mkdir -p /logs/agent")
        await environment.exec(
            f"cat > {shlex.quote(ANSWER_PATH)} <<'LEXAM_ANSWER_EOF'\n{payload}\nLEXAM_ANSWER_EOF"
        )

        tokens_in = metrics.get("tokens_in")
        tokens_out = metrics.get("tokens_out")
        if tokens_in is not None:
            context.n_input_tokens = int(tokens_in)
        if tokens_out is not None:
            context.n_output_tokens = int(tokens_out)
        context.metadata = {
            key: value
            for key, value in metrics.items()
            if key not in {"tokens_in", "tokens_out"}
        }
