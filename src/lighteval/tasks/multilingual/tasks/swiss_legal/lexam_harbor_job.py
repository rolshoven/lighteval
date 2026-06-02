import json
from pathlib import Path
from typing import Any

from lighteval.harbor.ingest import ingest_trial_outputs
from lighteval.harbor.scaffold import scaffold_harbor_tasks_from_docs


def export_lexam_harbor_tasks(documents_dict: dict[str, list[Any]], output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = scaffold_harbor_tasks_from_docs(documents_dict, output_dir)
    manifest = [{"task_dir": str(path)} for path in generated]
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return generated


def ingest_lexam_harbor_trials(job_dir: str | Path, answers_dir: str | Path):
    return ingest_trial_outputs(job_dir=job_dir, answers_dir=answers_dir)
