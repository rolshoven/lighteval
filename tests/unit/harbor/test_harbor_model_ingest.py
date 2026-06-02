import json
from pathlib import Path

from lighteval.harbor.scaffold import safe_harbor_name
from lighteval.models.harbor.harbor_model import HarborModel, HarborModelConfig
from lighteval.tasks.requests import Doc


def test_harbor_model_reads_cached_answers(tmp_path: Path):
    task_name = "lexam_mcq_4:en|0"
    answers_dir = tmp_path / "answers" / safe_harbor_name(task_name)
    answers_dir.mkdir(parents=True)
    (answers_dir / "0.json").write_text(
        json.dumps({"text": "Final Answer: ###A###", "reasoning": None, "metrics": {"wall_time_sec": 0.5}})
    )

    model = HarborModel(HarborModelConfig(model_name="harbor-agent", answers_dir=str(tmp_path / "answers")))
    doc = Doc(
        task_name=task_name,
        query="Q",
        choices=["A", "B", "C", "D"],
        gold_index=0,
        id="0",
        specific={},
    )
    response = model.greedy_until([doc])[0]
    assert response.text[0] == "Final Answer: ###A###"
    assert doc.specific["harbor"]["operational_metrics"]["wall_time_sec"] == 0.5
