"""Ingest Harbor trial directories into lighteval answer JSON.

Harbor agents should write answer artifacts at one of:

- ``<trial>/answer.json``
- ``<trial>/sample_output.json``
- ``<trial>/logs/agent/answer.json``
- ``<trial>/logs/agent/sample_output.json``
- ``<trial>/agent/answer.json`` (Harbor bind-mount layout under ``jobs/``)

Task metadata (``task_name``, ``doc_id``) lives in ``tasks/<sample>/metadata.json``.
After ``harbor run``, answers are typically under ``jobs/<run>/<trial>/agent/``.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lighteval.harbor.scaffold import safe_harbor_name


@dataclass(frozen=True)
class IngestedHarborAnswer:
    task_name: str
    doc_id: str
    text: str
    reasoning: list[str] | None
    metrics: dict[str, float]


def _as_reasoning(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _as_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    parsed: dict[str, float] = {}
    for key, raw in value.items():
        try:
            parsed[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return parsed


def _parse_answer_payload(answer_path: Path) -> tuple[str, list[str] | None, dict[str, float]]:
    try:
        payload = json.loads(answer_path.read_text())
    except json.JSONDecodeError:
        return answer_path.read_text(), None, {}

    if isinstance(payload, dict):
        if "text" in payload:
            return str(payload.get("text", "")), _as_reasoning(payload.get("reasoning")), _as_metrics(payload.get("metrics"))

        if "final_answer" in payload:
            return str(payload.get("final_answer", "")), _as_reasoning(payload.get("reasoning")), _as_metrics(
                payload.get("metrics")
            )

    return str(payload), None, {}


def _answer_candidates(sample_dir: Path) -> list[Path]:
    return [
        sample_dir / "answer.json",
        sample_dir / "sample_output.json",
        sample_dir / "logs" / "agent" / "answer.json",
        sample_dir / "logs" / "agent" / "sample_output.json",
        sample_dir / "agent" / "answer.json",
        sample_dir / "agent" / "sample_output.json",
    ]


def _find_metadata_files(root_dir: Path) -> list[Path]:
    return sorted(root_dir.rglob("metadata.json"))


def _resolve_harbor_root(job_dir: Path) -> Path | None:
    if (job_dir / "jobs").is_dir() and (job_dir / "tasks").is_dir():
        return job_dir
    if job_dir.name == "tasks" and (job_dir.parent / "jobs").is_dir():
        return job_dir.parent
    return None


def _write_ingested_answer(
    *,
    metadata_path: Path,
    answer_path: Path,
    answers_dir: Path,
    ingested: list[IngestedHarborAnswer],
    seen: set[tuple[str, str]],
) -> None:
    metadata = json.loads(metadata_path.read_text())
    task_name = str(metadata["task_name"])
    doc_id = str(metadata["doc_id"])
    key = (task_name, doc_id)
    if key in seen:
        return

    text, reasoning, metrics = _parse_answer_payload(answer_path)
    output_payload = {"text": text, "reasoning": reasoning, "metrics": metrics}

    task_dir = answers_dir / safe_harbor_name(task_name)
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / f"{doc_id}.json").write_text(json.dumps(output_payload, indent=2, sort_keys=True))

    ingested.append(
        IngestedHarborAnswer(
            task_name=task_name,
            doc_id=doc_id,
            text=text,
            reasoning=reasoning,
            metrics=metrics,
        )
    )
    seen.add(key)


def _ingest_colocated_samples(job_dir: Path, answers_dir: Path, ingested: list[IngestedHarborAnswer], seen: set[tuple[str, str]]) -> None:
    for metadata_path in _find_metadata_files(job_dir):
        sample_dir = metadata_path.parent
        answer_path = next((candidate for candidate in _answer_candidates(sample_dir) if candidate.exists()), None)
        if answer_path is None:
            continue
        _write_ingested_answer(
            metadata_path=metadata_path,
            answer_path=answer_path,
            answers_dir=answers_dir,
            ingested=ingested,
            seen=seen,
        )


def _ingest_harbor_job_trials(harbor_root: Path, answers_dir: Path, ingested: list[IngestedHarborAnswer], seen: set[tuple[str, str]]) -> None:
    jobs_dir = harbor_root / "jobs"
    if not jobs_dir.is_dir():
        return

    for config_path in sorted(jobs_dir.rglob("config.json")):
        trial_dir = config_path.parent
        if trial_dir.parent == jobs_dir:
            continue

        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            continue

        task_path = config.get("task", {}).get("path")
        if not task_path:
            continue

        metadata_path = harbor_root / task_path / "metadata.json"
        if not metadata_path.exists():
            continue

        answer_path = next((candidate for candidate in _answer_candidates(trial_dir) if candidate.exists()), None)
        if answer_path is None:
            continue

        _write_ingested_answer(
            metadata_path=metadata_path,
            answer_path=answer_path,
            answers_dir=answers_dir,
            ingested=ingested,
            seen=seen,
        )


def ingest_trial_outputs(job_dir: str | Path, answers_dir: str | Path) -> list[IngestedHarborAnswer]:
    job_dir = Path(job_dir)
    answers_dir = Path(answers_dir)
    answers_dir.mkdir(parents=True, exist_ok=True)

    ingested: list[IngestedHarborAnswer] = []
    seen: set[tuple[str, str]] = set()

    _ingest_colocated_samples(job_dir, answers_dir, ingested, seen)

    harbor_root = _resolve_harbor_root(job_dir)
    if harbor_root is not None:
        _ingest_harbor_job_trials(harbor_root, answers_dir, ingested, seen)

    return ingested
