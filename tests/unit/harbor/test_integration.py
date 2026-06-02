import json
from dataclasses import dataclass
from pathlib import Path

from lighteval.harbor.agent_spec import HarborAgentSpec
from lighteval.harbor.integration import (
    compare_canonical_metric_drift,
    extract_canonical_metrics,
    merge_utility_quality_metrics,
    write_harbor_reward_json,
)


class _NonSerializableModelConfig:
    """Stand-in for HarborModelConfig (Pydantic) in serialization tests."""

    model_name = "harbor-agent"
from lighteval.harbor.scaffold import scaffold_harbor_tasks, scaffold_harbor_tasks_from_docs
from lighteval.tasks.requests import Doc


@dataclass
class FakeDetail:
    doc: object
    model_response: object
    metric: dict


def _make_detail(task_name: str, wall_time: float, cost: float) -> FakeDetail:
    doc = type(
        "DocProxy",
        (),
        {"specific": {"harbor": {"operational_metrics": {"wall_time_sec": wall_time, "cost_usd": cost}}}},
    )()
    model_response = type("ModelResponseProxy", (), {"text": ["A"]})()
    return FakeDetail(doc=doc, model_response=model_response, metric={"acc": 1.0})


def test_compare_canonical_metric_drift_detects_differences():
    baseline = {"all": {"acc": 0.7}, "lexam_mcq_4:en|0": {"acc": 0.7}}
    harbor = {"all": {"acc": 0.72}, "lexam_mcq_4:en|0": {"acc": 0.72}}
    diff = compare_canonical_metric_drift(baseline_results=baseline, harbor_results=harbor, abs_tol=1e-4)
    assert diff["within_tolerance"] is False
    assert diff["differences"]


def test_extract_canonical_metrics_from_pipeline_final_dict():
    final_dict = {
        "config_general": {"model_config": _NonSerializableModelConfig()},
        "results": {"all": {"acc": 0.5}, "lexam_mcq_4:en|0": {"acc": 0.5}},
    }
    assert extract_canonical_metrics(final_dict) == final_dict["results"]


def test_write_harbor_reward_json_outputs_combined_payload(tmp_path: Path):
    details = {"lexam_mcq_4:en|0": [_make_detail("lexam_mcq_4:en|0", wall_time=1.0, cost=0.1)]}
    reward_path = tmp_path / "reward.json"
    canonical = {"all": {"acc": 0.8}, "lexam_mcq_4:en|0": {"acc": 0.8}}

    payload = write_harbor_reward_json(output_path=reward_path, canonical_metrics=canonical, details=details)
    on_disk = json.loads(reward_path.read_text())

    assert payload["sample_count"] == 1
    assert on_disk["canonical_metrics"]["all"]["acc"] == 0.8
    assert "harbor_operational_metrics" in on_disk
    assert "harbor_aggregate_metrics" in on_disk


def test_write_harbor_reward_json_uses_quality_metric_priority(tmp_path: Path):
    details = {"demo_task:en|0": [_make_detail("demo_task:en|0", wall_time=1.0, cost=0.0)]}
    reward_path = tmp_path / "reward.json"
    canonical = {
        "all": {"acc": 0.1, "custom_score": 0.9},
        "demo_task:en|0": {"acc": 0.1, "custom_score": 0.9},
    }

    payload = write_harbor_reward_json(
        output_path=reward_path,
        canonical_metrics=canonical,
        details=details,
        quality_metric_priority=("custom_score", "acc"),
    )

    assert payload["harbor_aggregate_metrics"]["quality_metric"] == "custom_score"
    assert payload["harbor_aggregate_metrics"]["quality"] == 0.9


def test_merge_utility_quality_metrics_from_tasks():
    spec = HarborAgentSpec(
        runner_module="m",
        utility_quality_metrics=("custom_score", "acc"),
    )
    task = type("Task", (), {"config": type("Config", (), {"harbor_agent": spec})()})()
    assert merge_utility_quality_metrics({"demo:en|0": task}) == ("custom_score", "acc")


def test_merge_utility_quality_metrics_defaults_to_acc():
    task = type("Task", (), {"config": type("Config", (), {"harbor_agent": None})()})()
    assert merge_utility_quality_metrics({"demo:en|0": task}) == ("acc",)


def test_write_harbor_reward_json_accepts_pipeline_final_dict(tmp_path: Path):
    details = {"lexam_mcq_4:en|0": [_make_detail("lexam_mcq_4:en|0", wall_time=2.0, cost=0.2)]}
    reward_path = tmp_path / "reward.json"
    final_dict = {
        "config_general": {"model_config": _NonSerializableModelConfig()},
        "results": {"all": {"acc": 1.0}, "lexam_mcq_4:en|0": {"acc": 1.0}},
    }

    write_harbor_reward_json(output_path=reward_path, canonical_metrics=final_dict, details=details)
    on_disk = json.loads(reward_path.read_text())

    assert on_disk["canonical_metrics"]["all"]["acc"] == 1.0
    assert on_disk["harbor_aggregate_metrics"]["quality"] == 1.0


def test_scaffold_harbor_tasks_creates_expected_structure(tmp_path: Path):
    generated = scaffold_harbor_tasks(task_names=["demo_task:en", "demo_task:de"], output_dir=tmp_path)
    assert len(generated) == 2
    first = generated[0]
    assert (first / "task.toml").exists()
    assert (first / "instruction.md").exists()
    assert (first / "environment" / "Dockerfile").exists()
    test_script = (first / "tests" / "test.sh").read_text()
    assert (first / "tests" / "test.sh").exists()
    assert "/logs/verifier/reward.txt" in test_script


def test_scaffold_harbor_tasks_from_docs_writes_metadata(tmp_path: Path):
    doc = Doc(
        task_name="demo_task:en|0",
        query="Question?",
        instruction="Answer carefully.",
        choices=["A", "B"],
        gold_index=0,
        id="42",
    )
    generated = scaffold_harbor_tasks_from_docs({"demo_task:en|0": [doc]}, output_dir=tmp_path)
    assert len(generated) == 1
    first = generated[0]
    metadata = json.loads((first / "metadata.json").read_text())
    assert metadata["task_name"] == "demo_task:en|0"
    assert metadata["doc_id"] == "42"
