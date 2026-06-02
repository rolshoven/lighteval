import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lighteval.harbor.agent_spec import HarborAgentSpec

if TYPE_CHECKING:
    from lighteval.tasks.requests import Doc
    from lighteval.logging.info_loggers import DetailsLogger


@dataclass
class HarborSamplePayload:
    task_name: str
    doc_id: str
    prompt: str
    query: str
    instruction: str | None
    choices: list[str]
    generation_size: int | None
    stop_sequences: list[str] | None
    metadata: dict[str, Any]


@dataclass
class HarborSampleResult:
    text: str
    reasonings: list[str] | None
    operational_metrics: dict[str, float]


def build_harbor_prompt(doc: Any) -> str:
    if doc.instruction:
        return f"{doc.instruction.strip()}\n\n{doc.query}"
    return doc.query


def build_harbor_payload(doc: Any) -> HarborSamplePayload:
    metadata = {}
    if doc.specific:
        metadata.update(doc.specific)
    return HarborSamplePayload(
        task_name=doc.task_name,
        doc_id=doc.id,
        prompt=build_harbor_prompt(doc),
        query=doc.query,
        instruction=doc.instruction,
        choices=doc.choices,
        generation_size=doc.generation_size,
        stop_sequences=doc.stop_sequences,
        metadata=metadata,
    )


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def aggregate_operational_metrics(details: dict[str, list[Any]]) -> dict[str, Any]:
    task_aggregates: dict[str, dict[str, float]] = {}
    global_values: dict[str, list[float]] = defaultdict(list)

    for task_name, task_details in details.items():
        metric_values: dict[str, list[float]] = defaultdict(list)
        for detail in task_details:
            specific = detail.doc.specific or {}
            harbor_payload = specific.get("harbor", {})
            metrics = harbor_payload.get("operational_metrics", {})
            for metric_name, metric_value in metrics.items():
                parsed = _safe_float(metric_value)
                if parsed is not None:
                    metric_values[metric_name].append(parsed)
                    global_values[metric_name].append(parsed)
        task_aggregates[task_name] = {
            metric_name: (sum(values) / len(values)) for metric_name, values in metric_values.items() if values
        }

    all_aggregates = {
        metric_name: (sum(values) / len(values)) for metric_name, values in global_values.items() if values
    }

    return {"all": all_aggregates, "by_task": task_aggregates}


def _task_harbor_spec(task: Any) -> HarborAgentSpec | None:
    config = getattr(task, "config", None)
    if config is None:
        return None
    return getattr(config, "harbor_agent", None)


def merge_utility_quality_metrics(tasks_dict: dict[str, Any]) -> tuple[str, ...]:
    """Merge task-level utility metric priorities; default to ``(\"acc\",)`` when unset."""
    metrics_by_task: dict[str, tuple[str, ...]] = {}
    for task_name, task in tasks_dict.items():
        spec = _task_harbor_spec(task)
        if spec is None or spec.utility_quality_metrics is None:
            continue
        metrics_by_task[task_name] = spec.utility_quality_metrics

    if not metrics_by_task:
        return ("acc",)

    unique_metrics = set(metrics_by_task.values())
    if len(unique_metrics) > 1:
        details = ", ".join(
            f"{task_name}={metrics}" for task_name, metrics in sorted(metrics_by_task.items())
        )
        raise ValueError(
            "Harbor reward utility currently supports one utility_quality_metrics list per run. "
            f"Found multiple task metric priorities: {details}"
        )
    return next(iter(unique_metrics))


def extract_canonical_metrics(results_or_metrics: dict[str, Any]) -> dict[str, Any]:
    """Return aggregated task metrics from a pipeline final_dict or pass through flat metrics."""
    nested = results_or_metrics.get("results")
    if isinstance(nested, dict):
        return nested
    return results_or_metrics


def compute_harbor_utility(
    canonical_metrics: dict[str, dict[str, Any]],
    operational_metrics: dict[str, Any],
    quality_metric_priority: tuple[str, ...] = ("acc",),
    alpha: float = 1.0,
    beta: float = 0.02,
    gamma: float = 0.001,
) -> dict[str, Any]:
    quality = 0.0
    quality_metric = None

    all_task_values = canonical_metrics.get("all", {})
    for candidate in quality_metric_priority:
        candidate_value = all_task_values.get(candidate)
        if isinstance(candidate_value, (int, float)):
            quality = float(candidate_value)
            quality_metric = candidate
            break

    wall_time_sec = float(operational_metrics.get("all", {}).get("wall_time_sec", 0.0))
    cost_usd = float(operational_metrics.get("all", {}).get("cost_usd", 0.0))

    utility = alpha * quality - beta * cost_usd - gamma * wall_time_sec
    return {
        "utility": utility,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "quality_metric": quality_metric,
        "quality": quality,
        "cost_usd": cost_usd,
        "wall_time_sec": wall_time_sec,
    }


def write_harbor_reward_json(
    output_path: str | Path,
    canonical_metrics: dict[str, Any],
    details: dict[str, list[Any]],
    quality_metric_priority: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = extract_canonical_metrics(canonical_metrics)
    operational_metrics = aggregate_operational_metrics(details)
    utility_metrics = compute_harbor_utility(
        canonical_metrics=metrics,
        operational_metrics=operational_metrics,
        quality_metric_priority=quality_metric_priority or ("acc",),
    )

    sample_count = sum(len(task_details) for task_details in details.values())
    reward_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": sample_count,
        "canonical_metrics": metrics,
        "harbor_operational_metrics": operational_metrics,
        "harbor_aggregate_metrics": utility_metrics,
    }
    output_path.write_text(json.dumps(reward_payload, indent=2, sort_keys=True, default=str))
    return reward_payload


def compare_canonical_metric_drift(
    baseline_results: dict[str, Any],
    harbor_results: dict[str, Any],
    abs_tol: float = 1e-6,
) -> dict[str, Any]:
    drift = {"within_tolerance": True, "differences": []}

    shared_tasks = sorted(set(baseline_results.keys()) & set(harbor_results.keys()))
    for task_name in shared_tasks:
        baseline_metrics = baseline_results.get(task_name, {})
        harbor_metrics = harbor_results.get(task_name, {})
        shared_metrics = sorted(set(baseline_metrics.keys()) & set(harbor_metrics.keys()))
        for metric_name in shared_metrics:
            baseline_value = baseline_metrics[metric_name]
            harbor_value = harbor_metrics[metric_name]
            if not isinstance(baseline_value, (int, float)) or not isinstance(harbor_value, (int, float)):
                continue
            delta = float(harbor_value) - float(baseline_value)
            if abs(delta) > abs_tol:
                drift["within_tolerance"] = False
                drift["differences"].append(
                    {
                        "task": task_name,
                        "metric": metric_name,
                        "baseline": baseline_value,
                        "harbor": harbor_value,
                        "delta": delta,
                    }
                )
    return drift


def details_to_jsonable(details: dict[str, list[Any]]) -> dict[str, Any]:
    serializable: dict[str, list[dict[str, Any]]] = {}
    for task_name, task_details in details.items():
        serializable[task_name] = []
        for detail in task_details:
            serializable[task_name].append(
                {
                    "doc": asdict(detail.doc),
                    "model_response": asdict(detail.model_response),
                    "metric": detail.metric,
                }
            )
    return serializable
