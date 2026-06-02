from pathlib import Path

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.harbor.harbor_model import HarborModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.multilingual.tasks.swiss_legal.lexam_harbor_job import export_lexam_harbor_tasks


def lexam_task_list() -> str:
    from lighteval.tasks.multilingual.tasks.swiss_legal.main import LEXAM_LANGUAGES, LEXAM_MCQ_NUM_CHOICES

    task_names = [f"lexam_oq:{lang}" for lang in LEXAM_LANGUAGES]
    task_names.extend(
        f"lexam_mcq_{num_choices}{'_idk' if with_idk else ''}:{lang}"
        for lang in LEXAM_LANGUAGES
        for num_choices in LEXAM_MCQ_NUM_CHOICES
        for with_idk in (False, True)
    )
    return ",".join(task_names)


def main(output_dir: str, tasks: str | None = None, max_samples: int | None = 10):
    selected_tasks = tasks or lexam_task_list()
    pipeline = Pipeline(
        tasks=selected_tasks,
        pipeline_parameters=PipelineParameters(
            launcher_type=ParallelismManager.NONE,
            load_tasks_multilingual=True,
            max_samples=max_samples,
        ),
        evaluation_tracker=EvaluationTracker(output_dir=output_dir),
        model_config=HarborModelConfig(),
    )
    generated = export_lexam_harbor_tasks(pipeline.documents_dict, output_dir)
    print(f"Generated {len(generated)} Harbor task directories in {Path(output_dir)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate doc-backed Harbor task folders for LEXam.")
    parser.add_argument("output_dir", help="Output directory where Harbor task folders will be created.")
    parser.add_argument("--tasks", default=None, help="Comma-separated lighteval task list (defaults to all LEXam variants).")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum samples per task to scaffold.")
    args = parser.parse_args()
    main(output_dir=args.output_dir, tasks=args.tasks, max_samples=args.max_samples)
