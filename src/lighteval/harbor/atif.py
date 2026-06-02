"""Minimal ATIF trajectory builders (optional Harbor/atif validation)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ATIF_SCHEMA_VERSION = "ATIF-v1.6"


def new_trajectory(
    *,
    agent_name: str,
    agent_version: str,
    model_name: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": ATIF_SCHEMA_VERSION,
        "session_id": session_id or str(uuid.uuid4()),
        "agent": {
            "name": agent_name,
            "version": agent_version,
            **({"model_name": model_name} if model_name else {}),
        },
        "steps": [],
    }


def append_user_step(trajectory: dict[str, Any], message: str) -> None:
    trajectory["steps"].append(
        {
            "step_id": len(trajectory["steps"]) + 1,
            "timestamp": _iso_timestamp(),
            "source": "user",
            "message": message,
        }
    )


def append_system_step(trajectory: dict[str, Any], message: str) -> None:
    trajectory["steps"].append(
        {
            "step_id": len(trajectory["steps"]) + 1,
            "timestamp": _iso_timestamp(),
            "source": "system",
            "message": message,
        }
    )


def append_agent_step(
    trajectory: dict[str, Any],
    *,
    message: str,
    model_name: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    observation_results: list[dict[str, Any]] | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    step: dict[str, Any] = {
        "step_id": len(trajectory["steps"]) + 1,
        "timestamp": _iso_timestamp(),
        "source": "agent",
        "message": message,
    }
    if model_name:
        step["model_name"] = model_name
    if tool_calls:
        step["tool_calls"] = tool_calls
    if observation_results:
        step["observation"] = {"results": observation_results}
    if metrics:
        step["metrics"] = metrics
    trajectory["steps"].append(step)


def set_final_metrics(trajectory: dict[str, Any], metrics: dict[str, float]) -> None:
    final: dict[str, Any] = {}
    if "prompt_tokens" in metrics:
        final["total_prompt_tokens"] = int(metrics["prompt_tokens"])
    if "completion_tokens" in metrics:
        final["total_completion_tokens"] = int(metrics["completion_tokens"])
    if "cost_usd" in metrics:
        final["total_cost_usd"] = float(metrics["cost_usd"])
    if final:
        trajectory["final_metrics"] = final


def write_trajectory(path: Path, trajectory: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trajectory, indent=2, ensure_ascii=False), encoding="utf-8")


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
