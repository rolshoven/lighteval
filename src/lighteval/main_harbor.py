import json
import os
from pathlib import Path

import typer
from typer import Argument, Option
from typing_extensions import Annotated

from lighteval.cli_args import (
    custom_tasks,
    dataset_loading_processes,
    job_id,
    load_responses_from_details_date_id,
    load_tasks_multilingual,
    max_samples,
    model_args,
    num_fewshot_seeds,
    output_dir,
    public_run,
    push_to_hub,
    push_to_tensorboard,
    reasoning_tags,
    remove_reasoning_tags,
    results_org,
    results_path_template,
    save_details,
    tasks,
    wandb,
)
from lighteval.harbor.integration import (
    compare_canonical_metric_drift,
    merge_utility_quality_metrics,
    write_harbor_reward_json,
)
from lighteval.harbor.job_runner import run_harbor_jobs_for_docs
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.harbor.harbor_model import HarborModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters


app = typer.Typer(help="Harbor-backed lighteval workflows.")


def _wire_harbor_task_specs(pipeline: Pipeline) -> None:
    if hasattr(pipeline.model, "set_task_harbor_specs"):
        task_harbor_specs = {task.full_name: task.config.harbor_agent for task in pipeline.tasks_dict.values()}
        pipeline.model.set_task_harbor_specs(task_harbor_specs)


def _resolve_reward_quality_metrics(pipeline: Pipeline) -> tuple[str, ...]:
    return merge_utility_quality_metrics(pipeline.tasks_dict)


def _run_harbor_eval(
    # === general ===
    model_args: str,
    tasks: str,
    harbor_command_template: Annotated[
        str | None,
        Option(
            help="Shell command template used to run Harbor for each sample. "
            "Available placeholders: {input_path}, {output_path}, {reward_path}, {task_name}, {doc_id}.",
        ),
    ] = None,
    harbor_agent_runner: Annotated[
        str | None,
        Option(
            help="Path to an external runner script. "
            "The script must accept --input-path and --output-path.",
        ),
    ] = None,
    harbor_working_dir: Annotated[str | None, Option(help="Working directory for Harbor command execution.")] = None,
    harbor_timeout_seconds: Annotated[
        float,
        Option(help="Per-sample timeout (seconds) for Harbor command execution."),
    ] = 1800.0,
    harbor_reward_output_path: Annotated[
        str | None,
        Option(
            help="Where to write Harbor reward JSON that combines canonical lighteval metrics and Harbor operational metrics. "
            "Defaults to <output_dir>/harbor/reward.json.",
        ),
    ] = None,
    harbor_answers_dir: Annotated[
        str | None,
        Option(help="Path to pre-generated Harbor answers. If set, no subprocess runner is invoked."),
    ] = None,
    # === Common parameters ===
    load_tasks_multilingual: load_tasks_multilingual.type = load_tasks_multilingual.default,
    dataset_loading_processes: dataset_loading_processes.type = dataset_loading_processes.default,
    custom_tasks: custom_tasks.type = custom_tasks.default,
    num_fewshot_seeds: num_fewshot_seeds.type = num_fewshot_seeds.default,
    load_responses_from_details_date_id: load_responses_from_details_date_id.type = load_responses_from_details_date_id.default,
    remove_reasoning_tags: remove_reasoning_tags.type = remove_reasoning_tags.default,
    reasoning_tags: reasoning_tags.type = reasoning_tags.default,
    # === saving ===
    output_dir: str = output_dir.default,
    results_path_template: results_path_template.type = results_path_template.default,
    push_to_hub: push_to_hub.type = push_to_hub.default,
    push_to_tensorboard: push_to_tensorboard.type = push_to_tensorboard.default,
    public_run: public_run.type = public_run.default,
    results_org: results_org.type = results_org.default,
    save_details: save_details.type = True,
    wandb: wandb.type = wandb.default,
    # === debug ===
    max_samples: max_samples.type = max_samples.default,
    job_id: job_id.type = job_id.default,
):
    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        results_path_template=results_path_template,
        save_details=save_details,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public=public_run,
        hub_results_org=results_org,
        use_wandb=wandb,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        job_id=job_id,
        load_tasks_multilingual=load_tasks_multilingual,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        load_responses_from_details_date_id=load_responses_from_details_date_id,
        remove_reasoning_tags=remove_reasoning_tags,
        reasoning_tags=reasoning_tags,
    )

    if model_args.endswith(".yaml"):
        model_config = HarborModelConfig.from_path(model_args)
    else:
        model_config = HarborModelConfig.from_args(model_args)

    if harbor_command_template is not None:
        model_config.cli_command_template = harbor_command_template
    if harbor_agent_runner is not None:
        model_config.cli_agent_runner = harbor_agent_runner
    if harbor_answers_dir is not None:
        model_config.answers_dir = harbor_answers_dir
    model_config.working_dir = harbor_working_dir
    model_config.timeout_seconds = harbor_timeout_seconds

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )
    _wire_harbor_task_specs(pipeline)

    pipeline.evaluate()
    results = pipeline.get_results()
    details = pipeline.get_details()

    output_path = (
        Path(harbor_reward_output_path)
        if harbor_reward_output_path is not None
        else Path(output_dir) / "harbor" / "reward.json"
    )
    write_harbor_reward_json(
        output_path=output_path,
        canonical_metrics=results,
        details=details,
        quality_metric_priority=_resolve_reward_quality_metrics(pipeline),
    )
    pipeline.show_results()
    pipeline.save_and_push_results()

    return results, details


def _build_harbor_pipeline(
    *,
    model_args: str,
    tasks: str,
    load_tasks_multilingual: bool,
    dataset_loading_processes: int,
    custom_tasks: str | None,
    num_fewshot_seeds: int,
    load_responses_from_details_date_id: str | None,
    remove_reasoning_tags: bool,
    reasoning_tags: str,
    output_dir: str,
    results_path_template: str,
    push_to_hub: bool,
    push_to_tensorboard: bool,
    public_run: bool,
    results_org: str | None,
    save_details: bool,
    wandb: bool,
    max_samples: int | None,
    job_id: int,
) -> Pipeline:
    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        results_path_template=results_path_template,
        save_details=save_details,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public=public_run,
        hub_results_org=results_org,
        use_wandb=wandb,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        job_id=job_id,
        load_tasks_multilingual=load_tasks_multilingual,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        load_responses_from_details_date_id=load_responses_from_details_date_id,
        remove_reasoning_tags=remove_reasoning_tags,
        reasoning_tags=reasoning_tags,
    )

    if model_args.endswith(".yaml"):
        model_config = HarborModelConfig.from_path(model_args)
    else:
        model_config = HarborModelConfig.from_args(model_args)
    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )
    _wire_harbor_task_specs(pipeline)
    return pipeline


def _evaluate_harbor_pipeline(
    pipeline: Pipeline,
    *,
    harbor_answers_dir: str,
    harbor_reward_output_path: str | None,
    output_dir: str,
) -> tuple[dict, dict]:
    if not isinstance(pipeline.model.config, HarborModelConfig):
        raise TypeError("Harbor pipeline requires HarborModelConfig.")
    pipeline.model.config.answers_dir = harbor_answers_dir

    pipeline.evaluate()
    results = pipeline.get_results()
    details = pipeline.get_details()

    output_path = (
        Path(harbor_reward_output_path)
        if harbor_reward_output_path is not None
        else Path(output_dir) / "harbor" / "reward.json"
    )
    write_harbor_reward_json(
        output_path=output_path,
        canonical_metrics=results,
        details=details,
        quality_metric_priority=_resolve_reward_quality_metrics(pipeline),
    )
    pipeline.show_results()
    pipeline.save_and_push_results()
    return results, details


def _run_harbor_job_pipeline(
    *,
    pipeline: Pipeline | None,
    model_args: str,
    tasks: str,
    harbor_agent: str | None,
    harbor_agent_import: str | None,
    harbor_model: str | None,
    harbor_env: str | None,
    harbor_timeout_seconds: float,
    harbor_job_dir: Path,
    harbor_reward_output_path: str | None,
    output_dir: str,
    load_tasks_multilingual: bool,
    dataset_loading_processes: int,
    custom_tasks: str | None,
    num_fewshot_seeds: int,
    load_responses_from_details_date_id: str | None,
    remove_reasoning_tags: bool,
    reasoning_tags: str,
    results_path_template: str,
    push_to_hub: bool,
    push_to_tensorboard: bool,
    public_run: bool,
    results_org: str | None,
    save_details: bool,
    wandb: bool,
    max_samples: int | None,
    job_id: int,
) -> tuple[dict, dict]:
    if pipeline is None:
        pipeline = _build_harbor_pipeline(
            model_args=model_args,
            tasks=tasks,
            load_tasks_multilingual=load_tasks_multilingual,
            dataset_loading_processes=dataset_loading_processes,
            custom_tasks=custom_tasks,
            num_fewshot_seeds=num_fewshot_seeds,
            load_responses_from_details_date_id=load_responses_from_details_date_id,
            remove_reasoning_tags=remove_reasoning_tags,
            reasoning_tags=reasoning_tags,
            output_dir=output_dir,
            results_path_template=results_path_template,
            push_to_hub=push_to_hub,
            push_to_tensorboard=push_to_tensorboard,
            public_run=public_run,
            results_org=results_org,
            save_details=save_details,
            wandb=wandb,
            max_samples=max_samples,
            job_id=job_id,
        )

    answers_dir = run_harbor_jobs_for_docs(
        tasks_dict=pipeline.tasks_dict,
        documents_dict=pipeline.documents_dict,
        harbor_job_dir=harbor_job_dir,
        harbor_agent=harbor_agent,
        harbor_agent_import=harbor_agent_import,
        harbor_model=harbor_model,
        harbor_env=harbor_env,
        harbor_timeout_seconds=harbor_timeout_seconds,
    )
    return _evaluate_harbor_pipeline(
        pipeline,
        harbor_answers_dir=str(answers_dir),
        harbor_reward_output_path=harbor_reward_output_path,
        output_dir=output_dir,
    )


@app.command("run")
def run_harbor(
    # === general ===
    model_args: model_args.type,
    tasks: tasks.type,
    harbor_command_template: Annotated[
        str | None,
        Option(
            help="Shell command template used to run Harbor for each sample. "
            "Available placeholders: {input_path}, {output_path}, {reward_path}, {task_name}, {doc_id}.",
        ),
    ] = None,
    harbor_agent_runner: Annotated[
        str | None,
        Option(
            help="Path to an external runner script. "
            "The script must accept --input-path and --output-path.",
        ),
    ] = None,
    harbor_working_dir: Annotated[str | None, Option(help="Working directory for Harbor command execution.")] = None,
    harbor_timeout_seconds: Annotated[
        float,
        Option(help="Per-sample timeout (seconds) for Harbor command execution."),
    ] = 1800.0,
    harbor_reward_output_path: Annotated[
        str | None,
        Option(
            help="Where to write Harbor reward JSON that combines canonical lighteval metrics and Harbor operational metrics. "
            "Defaults to <output_dir>/harbor/reward.json.",
        ),
    ] = None,
    # === Common parameters ===
    load_tasks_multilingual: load_tasks_multilingual.type = load_tasks_multilingual.default,
    dataset_loading_processes: dataset_loading_processes.type = dataset_loading_processes.default,
    custom_tasks: custom_tasks.type = custom_tasks.default,
    num_fewshot_seeds: num_fewshot_seeds.type = num_fewshot_seeds.default,
    load_responses_from_details_date_id: load_responses_from_details_date_id.type = load_responses_from_details_date_id.default,
    remove_reasoning_tags: remove_reasoning_tags.type = remove_reasoning_tags.default,
    reasoning_tags: reasoning_tags.type = reasoning_tags.default,
    # === saving ===
    output_dir: output_dir.type = output_dir.default,
    results_path_template: results_path_template.type = results_path_template.default,
    push_to_hub: push_to_hub.type = push_to_hub.default,
    push_to_tensorboard: push_to_tensorboard.type = push_to_tensorboard.default,
    public_run: public_run.type = public_run.default,
    results_org: results_org.type = results_org.default,
    save_details: save_details.type = True,
    wandb: wandb.type = wandb.default,
    # === debug ===
    max_samples: max_samples.type = max_samples.default,
    job_id: job_id.type = job_id.default,
):
    """Run lighteval with Harbor-backed agent execution and native task metrics."""
    return _run_harbor_eval(
        model_args=model_args,
        tasks=tasks,
        harbor_command_template=harbor_command_template,
        harbor_agent_runner=harbor_agent_runner,
        harbor_working_dir=harbor_working_dir,
        harbor_timeout_seconds=harbor_timeout_seconds,
        harbor_reward_output_path=harbor_reward_output_path,
        harbor_answers_dir=None,
        load_tasks_multilingual=load_tasks_multilingual,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks=custom_tasks,
        num_fewshot_seeds=num_fewshot_seeds,
        load_responses_from_details_date_id=load_responses_from_details_date_id,
        remove_reasoning_tags=remove_reasoning_tags,
        reasoning_tags=reasoning_tags,
        output_dir=output_dir,
        results_path_template=results_path_template,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public_run=public_run,
        results_org=results_org,
        save_details=save_details,
        wandb=wandb,
        max_samples=max_samples,
        job_id=job_id,
    )


@app.command("job")
def job_harbor(
    model_args: model_args.type,
    tasks: tasks.type,
    harbor_agent: Annotated[
        str | None,
        Option(
            help="Harbor built-in agent name (e.g. 'terminus-2') passed to `harbor run -a`. "
            "When omitted, uses the task's harbor_agent_import. "
            "When set to the task's default_harbor_agent alias, resolves to harbor_agent_import."
        ),
    ] = None,
    harbor_agent_import: Annotated[
        str | None,
        Option(help="Optional Harbor agent import override (pkg.module:Class). If omitted, task-local harbor_agent_import is used."),
    ] = None,
    harbor_model: Annotated[str | None, Option(help="Harbor model to pass to `harbor run --model`.")] = None,
    harbor_env: Annotated[str | None, Option(help="Harbor execution environment passed as `--env`.")] = None,
    harbor_timeout_seconds: Annotated[
        float,
        Option(help="Per-task timeout (seconds) for Harbor CLI invocations."),
    ] = 1800.0,
    harbor_job_dir: Annotated[
        str | None,
        Option(help="Directory used for Harbor task scaffolding, logs, and ingested answers. Defaults to <output_dir>/harbor/job."),
    ] = None,
    harbor_reward_output_path: Annotated[
        str | None,
        Option(help="Reward JSON path. Defaults to <output_dir>/harbor/reward.json."),
    ] = None,
    load_tasks_multilingual: load_tasks_multilingual.type = load_tasks_multilingual.default,
    dataset_loading_processes: dataset_loading_processes.type = dataset_loading_processes.default,
    custom_tasks: custom_tasks.type = custom_tasks.default,
    num_fewshot_seeds: num_fewshot_seeds.type = num_fewshot_seeds.default,
    load_responses_from_details_date_id: load_responses_from_details_date_id.type = load_responses_from_details_date_id.default,
    remove_reasoning_tags: remove_reasoning_tags.type = remove_reasoning_tags.default,
    reasoning_tags: reasoning_tags.type = reasoning_tags.default,
    output_dir: output_dir.type = output_dir.default,
    results_path_template: results_path_template.type = results_path_template.default,
    push_to_hub: push_to_hub.type = push_to_hub.default,
    push_to_tensorboard: push_to_tensorboard.type = push_to_tensorboard.default,
    public_run: public_run.type = public_run.default,
    results_org: results_org.type = results_org.default,
    save_details: save_details.type = True,
    wandb: wandb.type = wandb.default,
    max_samples: max_samples.type = max_samples.default,
    job_id: job_id.type = job_id.default,
):
    """Run Harbor sandbox jobs, ingest answers, then score with lighteval metrics."""
    resolved_job_dir = Path(harbor_job_dir) if harbor_job_dir else Path(output_dir) / "harbor" / "job"
    return _run_harbor_job_pipeline(
        pipeline=None,
        model_args=model_args,
        tasks=tasks,
        harbor_agent=harbor_agent,
        harbor_agent_import=harbor_agent_import,
        harbor_model=harbor_model,
        harbor_env=harbor_env,
        harbor_timeout_seconds=harbor_timeout_seconds,
        harbor_job_dir=resolved_job_dir,
        harbor_reward_output_path=harbor_reward_output_path,
        output_dir=output_dir,
        load_tasks_multilingual=load_tasks_multilingual,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks=custom_tasks,
        num_fewshot_seeds=num_fewshot_seeds,
        load_responses_from_details_date_id=load_responses_from_details_date_id,
        remove_reasoning_tags=remove_reasoning_tags,
        reasoning_tags=reasoning_tags,
        results_path_template=results_path_template,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public_run=public_run,
        results_org=results_org,
        save_details=save_details,
        wandb=wandb,
        max_samples=max_samples,
        job_id=job_id,
    )


@app.command("score")
def score_harbor(
    model_args: model_args.type,
    tasks: tasks.type,
    harbor_answers_dir: Annotated[
        str | None,
        Option(help="Pre-generated answers directory. Defaults to <harbor-job-dir>/answers."),
    ] = None,
    harbor_job_dir: Annotated[
        str | None,
        Option(help="Job directory created by `lighteval harbor job`. Used to infer answers dir if not provided."),
    ] = None,
    harbor_timeout_seconds: Annotated[float, Option(help="Unused in score mode, kept for API symmetry.")] = 1800.0,
    harbor_reward_output_path: Annotated[
        str | None,
        Option(help="Reward JSON path. Defaults to <output_dir>/harbor/reward.json."),
    ] = None,
    load_tasks_multilingual: load_tasks_multilingual.type = load_tasks_multilingual.default,
    dataset_loading_processes: dataset_loading_processes.type = dataset_loading_processes.default,
    custom_tasks: custom_tasks.type = custom_tasks.default,
    num_fewshot_seeds: num_fewshot_seeds.type = num_fewshot_seeds.default,
    load_responses_from_details_date_id: load_responses_from_details_date_id.type = load_responses_from_details_date_id.default,
    remove_reasoning_tags: remove_reasoning_tags.type = remove_reasoning_tags.default,
    reasoning_tags: reasoning_tags.type = reasoning_tags.default,
    output_dir: output_dir.type = output_dir.default,
    results_path_template: results_path_template.type = results_path_template.default,
    push_to_hub: push_to_hub.type = push_to_hub.default,
    push_to_tensorboard: push_to_tensorboard.type = push_to_tensorboard.default,
    public_run: public_run.type = public_run.default,
    results_org: results_org.type = results_org.default,
    save_details: save_details.type = True,
    wandb: wandb.type = wandb.default,
    max_samples: max_samples.type = max_samples.default,
    job_id: job_id.type = job_id.default,
):
    """Score existing Harbor outputs with lighteval metrics only."""
    resolved_answers_dir = harbor_answers_dir
    if resolved_answers_dir is None and harbor_job_dir is not None:
        resolved_answers_dir = str(Path(harbor_job_dir) / "answers")
    if resolved_answers_dir is None:
        raise typer.BadParameter("Provide --harbor-answers-dir or --harbor-job-dir.")

    return _run_harbor_eval(
        model_args=model_args,
        tasks=tasks,
        harbor_command_template=None,
        harbor_agent_runner=None,
        harbor_working_dir=None,
        harbor_timeout_seconds=harbor_timeout_seconds,
        harbor_reward_output_path=harbor_reward_output_path,
        harbor_answers_dir=resolved_answers_dir,
        load_tasks_multilingual=load_tasks_multilingual,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks=custom_tasks,
        num_fewshot_seeds=num_fewshot_seeds,
        load_responses_from_details_date_id=load_responses_from_details_date_id,
        remove_reasoning_tags=remove_reasoning_tags,
        reasoning_tags=reasoning_tags,
        output_dir=output_dir,
        results_path_template=results_path_template,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public_run=public_run,
        results_org=results_org,
        save_details=save_details,
        wandb=wandb,
        max_samples=max_samples,
        job_id=job_id,
    )


@app.command("parity-check")
def parity_check(
    baseline_results_path: Annotated[str, Argument(help="Path to canonical lighteval results JSON.")],
    harbor_results_path: Annotated[str, Argument(help="Path to Harbor-backed lighteval results JSON.")],
    abs_tol: Annotated[float, Option(help="Absolute tolerance used for drift checking.")] = 1e-6,
):
    """Compare canonical and Harbor-backed lighteval metrics and fail on drift."""

    baseline = json.loads(Path(baseline_results_path).read_text())
    harbor = json.loads(Path(harbor_results_path).read_text())
    drift = compare_canonical_metric_drift(baseline_results=baseline, harbor_results=harbor, abs_tol=abs_tol)
    print(json.dumps(drift, indent=2, sort_keys=True))
    if not drift["within_tolerance"]:
        raise typer.Exit(code=1)
