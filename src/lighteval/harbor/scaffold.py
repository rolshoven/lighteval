import json
import re
from pathlib import Path
from typing import Any

from lighteval.harbor.integration import build_harbor_prompt


def _task_toml(task_name: str) -> str:
    return f"""[metadata]
name = "{task_name}"
description = "Harbor task wrapper for {task_name} executed and scored through lighteval."
author = "lighteval"

[agent]
timeout_sec = 1800

[verifier]
timeout_sec = 1800

[environment]
os = "linux"
"""


def _instruction_md(task_name: str) -> str:
    return f"""# Task

Run the configured agent against `{task_name}` through the Harbor execution harness.
Write the final answer JSON to the path provided by the runner.
"""


def _sample_instruction_md(task_name: str, prompt: str) -> str:
    return f"""# Task

Solve the following sample from `{task_name}`.
Write the final answer JSON to `/logs/agent/answer.json`.

## Prompt

{prompt}
"""


# Harbor copies task ``tests/`` into the container at verifier time (see TaskPaths).
# The image build context is ``environment/`` only, so do not COPY tests here.
DOCKERFILE = """FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends bash ca-certificates && rm -rf /var/lib/apt/lists/*
"""


TEST_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail

# Harbor requires the verifier to write /logs/verifier/reward.txt or reward.json.
# Canonical lighteval metrics run after trial ingest; this reward only signals
# that the agent produced the expected answer artifact.
ANSWER_PATH="${HARBOR_AGENT_ANSWER_PATH:-/logs/agent/answer.json}"
REWARD_PATH="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_PATH")"

if [ -f "$ANSWER_PATH" ]; then
  echo 1 > "$REWARD_PATH"
else
  echo 0 > "$REWARD_PATH"
  echo "missing answer json: $ANSWER_PATH" >&2
  exit 1
fi
"""


def safe_harbor_name(value: str) -> str:
    sanitized = value.replace(":", "__").replace("|", "__")
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", sanitized)
    return sanitized.strip("_") or "sample"


def scaffold_harbor_tasks(task_names: list[str], output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    generated_paths: list[Path] = []

    for task_name in task_names:
        safe_name = safe_harbor_name(task_name)
        task_dir = output_dir / safe_name
        env_dir = task_dir / "environment"
        tests_dir = task_dir / "tests"

        env_dir.mkdir(parents=True, exist_ok=True)
        tests_dir.mkdir(parents=True, exist_ok=True)

        (task_dir / "task.toml").write_text(_task_toml(task_name))
        (task_dir / "instruction.md").write_text(_instruction_md(task_name))
        (env_dir / "Dockerfile").write_text(DOCKERFILE)
        (tests_dir / "test.sh").write_text(TEST_SCRIPT)

        generated_paths.append(task_dir)

    return generated_paths


def scaffold_harbor_tasks_from_docs(docs_by_task: dict[str, list[Any]], output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    generated_paths: list[Path] = []

    for task_name, docs in docs_by_task.items():
        for doc in docs:
            sample_name = f"{task_name}|{doc.id}"
            safe_name = safe_harbor_name(sample_name)
            task_dir = output_dir / safe_name
            env_dir = task_dir / "environment"
            tests_dir = task_dir / "tests"

            env_dir.mkdir(parents=True, exist_ok=True)
            tests_dir.mkdir(parents=True, exist_ok=True)

            prompt = build_harbor_prompt(doc)
            metadata = {
                "task_name": task_name,
                "doc_id": doc.id,
                "choices": doc.choices,
                "generation_size": doc.generation_size,
                "stop_sequences": doc.stop_sequences,
            }

            (task_dir / "task.toml").write_text(_task_toml(task_name))
            (task_dir / "instruction.md").write_text(_sample_instruction_md(task_name=task_name, prompt=prompt))
            (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
            (env_dir / "Dockerfile").write_text(DOCKERFILE)
            (tests_dir / "test.sh").write_text(TEST_SCRIPT)

            generated_paths.append(task_dir)

    return generated_paths
