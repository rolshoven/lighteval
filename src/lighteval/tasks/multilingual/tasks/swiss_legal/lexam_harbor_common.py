"""Shared helpers for LEXam Harbor subprocess and sandbox agents."""

import json
import re
import time
from typing import Any, cast


def extract_text_from_litellm_response(response: object) -> str:
    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0]
        message = getattr(first, "message", None)
        if message is not None:
            content = getattr(message, "content", "")
            if content:
                return str(content)
        text = getattr(first, "text", None)
        if text:
            return str(text)
    if isinstance(response, dict):
        response_dict = cast(dict[str, Any], response)
        choices = response_dict.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if content:
                return str(content)
            text = choices[0].get("text")
            if text:
                return str(text)
    return ""


def extract_usage_metrics(response: object) -> dict[str, float]:
    metrics: dict[str, float] = {}
    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if prompt_tokens is not None:
            metrics["tokens_in"] = float(prompt_tokens)
        if completion_tokens is not None:
            metrics["tokens_out"] = float(completion_tokens)
    elif isinstance(response, dict):
        response_dict = cast(dict[str, Any], response)
        usage = response_dict.get("usage", {})
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if prompt_tokens is not None:
                metrics["tokens_in"] = float(prompt_tokens)
            if completion_tokens is not None:
                metrics["tokens_out"] = float(completion_tokens)
    return metrics


def ensure_mcq_final_answer_format(task_name: str, text: str, choices: list[str]) -> str:
    if not task_name.startswith("lexam_mcq_"):
        return text
    if re.search(r"###\s*[A-Z]\s*###", text):
        return text

    letter = None
    final_answer_match = re.search(r"final\s+answer\s*[:\-]?\s*([A-Z])", text, flags=re.IGNORECASE)
    if final_answer_match:
        candidate = final_answer_match.group(1).upper()
        if candidate in choices:
            letter = candidate

    if letter is None:
        standalone_letters = re.findall(r"\b([A-Z])\b", text)
        for candidate in reversed(standalone_letters):
            if candidate in choices:
                letter = candidate
                break

    if letter is None:
        letter = choices[0] if choices else "A"

    return f"{text.rstrip()}\n\nFinal Answer: ###{letter}###"


def run_litellm_baseline(sample: dict[str, Any], model_name: str) -> tuple[str, dict[str, float]]:
    try:
        from litellm import completion
    except ImportError as exc:
        raise RuntimeError(
            "The LEXam subprocess runner requires `litellm`. "
            "Install lighteval with endpoint extras or install litellm manually."
        ) from exc

    start = time.perf_counter()
    response = completion(
        model=model_name,
        messages=[{"role": "user", "content": sample["prompt"]}],
    )
    wall_time_sec = time.perf_counter() - start

    text = extract_text_from_litellm_response(response)
    metrics = extract_usage_metrics(response)
    metrics["wall_time_sec"] = wall_time_sec
    return text, metrics


def build_sample_output(sample: dict[str, Any], text: str, metrics: dict[str, float]) -> dict[str, Any]:
    text = ensure_mcq_final_answer_format(
        task_name=str(sample.get("task_name", "")),
        text=text,
        choices=list(sample.get("choices", [])),
    )
    return {
        "text": text,
        "reasoning": None,
        "metrics": metrics,
    }


def format_answer_payload(output: dict[str, Any]) -> str:
    return json.dumps(output, indent=2, sort_keys=True)
