import json
from pathlib import Path

from lighteval.harbor.ingest import ingest_trial_outputs
from lighteval.harbor.scaffold import safe_harbor_name


def test_ingest_trial_outputs_writes_contract_json(tmp_path: Path):
    job_dir = tmp_path / "job"
    sample_dir = job_dir / "sample-1"
    sample_dir.mkdir(parents=True)

    (sample_dir / "metadata.json").write_text(
        json.dumps({"task_name": "lexam_mcq_4:en|0", "doc_id": "7"}, indent=2, sort_keys=True)
    )
    (sample_dir / "answer.json").write_text(
        json.dumps({"text": "Final Answer: ###B###", "reasoning": None, "metrics": {"wall_time_sec": 1.2}})
    )

    answers_dir = tmp_path / "answers"
    ingested = ingest_trial_outputs(job_dir=job_dir, answers_dir=answers_dir)

    assert len(ingested) == 1
    answer_file = answers_dir / safe_harbor_name("lexam_mcq_4:en|0") / "7.json"
    payload = json.loads(answer_file.read_text())
    assert payload["text"] == "Final Answer: ###B###"
    assert payload["metrics"]["wall_time_sec"] == 1.2


def test_ingest_harbor_jobs_trial_agent_answer(tmp_path: Path):
    harbor_root = tmp_path / "harbor_job"
    task_dir = harbor_root / "tasks" / "lexam_mcq_4__en__0__463"
    trial_dir = harbor_root / "jobs" / "2026-06-01__run" / "lexam_mcq_4__en__0__463__trial"
    task_dir.mkdir(parents=True)
    trial_dir.mkdir(parents=True)
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True)

    (task_dir / "metadata.json").write_text(
        json.dumps({"task_name": "lexam_mcq_4:en|0", "doc_id": "463"}, indent=2, sort_keys=True)
    )
    (trial_dir / "config.json").write_text(
        json.dumps({"task": {"path": "tasks/lexam_mcq_4__en__0__463"}}, indent=2, sort_keys=True)
    )
    (agent_dir / "answer.json").write_text(
        json.dumps({"text": "Final Answer: ###A###", "reasoning": None, "metrics": {}})
    )

    answers_dir = tmp_path / "answers"
    ingested = ingest_trial_outputs(job_dir=harbor_root, answers_dir=answers_dir)

    assert len(ingested) == 1
    answer_file = answers_dir / safe_harbor_name("lexam_mcq_4:en|0") / "463.json"
    payload = json.loads(answer_file.read_text())
    assert payload["text"] == "Final Answer: ###A###"


def test_ingest_reads_logs_agent_answer_path(tmp_path: Path):
    job_dir = tmp_path / "job"
    sample_dir = job_dir / "trial-1"
    sample_dir.mkdir(parents=True)
    agent_logs = sample_dir / "logs" / "agent"
    agent_logs.mkdir(parents=True)

    (sample_dir / "metadata.json").write_text(
        json.dumps({"task_name": "lexam_oq:en|0", "doc_id": "3"}, indent=2, sort_keys=True)
    )
    (agent_logs / "answer.json").write_text(
        json.dumps({"text": "Legal analysis answer.", "reasoning": ["step 1"], "metrics": {}})
    )

    answers_dir = tmp_path / "answers"
    ingested = ingest_trial_outputs(job_dir=job_dir, answers_dir=answers_dir)

    assert len(ingested) == 1
    answer_file = answers_dir / safe_harbor_name("lexam_oq:en|0") / "3.json"
    payload = json.loads(answer_file.read_text())
    assert payload["text"] == "Legal analysis answer."
    assert payload["reasoning"] == ["step 1"]
