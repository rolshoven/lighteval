import json
import logging
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

from pydantic import Field
from transformers.models.auto.tokenization_auto import AutoTokenizer

from lighteval.harbor.agent_spec import HarborAgentSpec
from lighteval.harbor.integration import HarborSampleResult, build_harbor_payload
from lighteval.harbor.runner import resolve_command_template
from lighteval.harbor.scaffold import safe_harbor_name
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached


logger = logging.getLogger(__name__)


class HarborModelConfig(ModelConfig):
    model_name: str = "harbor-agent"
    command_template: str | None = None
    cli_command_template: str | None = None
    agent_runner: str | None = None
    cli_agent_runner: str | None = None
    reference_model: str | None = None
    working_dir: str | None = None
    timeout_seconds: float = 1800.0
    output_text_key: str = "text"
    output_reasoning_key: str = "reasoning"
    output_metrics_key: str = "metrics"
    answers_dir: str | None = None
    fail_on_command_error: bool = True
    environment: dict[str, str] = Field(default_factory=dict)


class HarborModel(LightevalModel):
    def __init__(self, config: HarborModelConfig):
        self.config = config
        self._tokenizer = None
        self._cache = SampleCache(config)
        self._task_harbor_specs: dict[str, HarborAgentSpec] = {}
        self._task_command_cache: dict[str, str] = {}

    def set_task_harbor_specs(self, task_harbor_specs: dict[str, HarborAgentSpec | None]):
        self._task_harbor_specs = {
            task_name: spec for task_name, spec in task_harbor_specs.items() if spec is not None
        }
        self._task_command_cache = {}

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return self._tokenizer

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        return 32768

    def _parse_harbor_output(self, output_path: Path) -> HarborSampleResult:
        if not output_path.exists():
            return HarborSampleResult(text="", reasonings=None, operational_metrics={})

        payload = json.loads(output_path.read_text())
        text = payload.get(self.config.output_text_key, "")
        reasoning_value = payload.get(self.config.output_reasoning_key)
        if reasoning_value is None:
            reasonings = None
        elif isinstance(reasoning_value, list):
            reasonings = [str(item) for item in reasoning_value]
        else:
            reasonings = [str(reasoning_value)]

        metrics = payload.get(self.config.output_metrics_key, {})
        if not isinstance(metrics, dict):
            metrics = {}
        parsed_metrics = {}
        for metric_name, metric_value in metrics.items():
            try:
                parsed_metrics[metric_name] = float(metric_value)
            except (TypeError, ValueError):
                continue
        return HarborSampleResult(text=str(text), reasonings=reasonings, operational_metrics=parsed_metrics)

    def _resolve_command_template(self, task_name: str) -> str:
        if task_name in self._task_command_cache:
            return self._task_command_cache[task_name]

        base_task_name = task_name.split("|", 1)[0]
        task_spec = self._task_harbor_specs.get(task_name) or self._task_harbor_specs.get(base_task_name)
        command = resolve_command_template(
            cli_template=self.config.cli_command_template,
            cli_agent_runner=self.config.cli_agent_runner,
            model_agent_runner=self.config.agent_runner,
            model_command_template=self.config.command_template,
            task_spec=task_spec,
        )
        self._task_command_cache[task_name] = command
        return command

    def _read_cached_answer(self, doc: Doc) -> HarborSampleResult:
        if self.config.answers_dir is None:
            raise ValueError("answers_dir is not configured.")

        task_candidates = [doc.task_name]
        if "|" in doc.task_name:
            task_candidates.append(doc.task_name.split("|", 1)[0])

        for task_name in task_candidates:
            answer_path = Path(self.config.answers_dir) / safe_harbor_name(task_name) / f"{doc.id}.json"
            if answer_path.exists():
                return self._parse_harbor_output(answer_path)

        candidates = ", ".join(
            str(Path(self.config.answers_dir) / safe_harbor_name(task_name) / f"{doc.id}.json")
            for task_name in task_candidates
        )
        raise FileNotFoundError(
            f"No cached Harbor answer found for task={doc.task_name} doc_id={doc.id}. "
            f"Tried: {candidates}"
        )

    def _invoke_harbor(self, doc: Doc) -> ModelResponse:
        if self.config.answers_dir is not None:
            sample_result = self._read_cached_answer(doc)
            if doc.specific is None:
                doc.specific = {}
            task_spec = self._task_harbor_specs.get(doc.task_name) or self._task_harbor_specs.get(
                doc.task_name.split("|", 1)[0]
            )
            doc.specific["harbor"] = {
                "operational_metrics": sample_result.operational_metrics,
                "runner_stdout": "",
                "runner_stderr": "",
                "command": "<cached-answers>",
                "task_runner_module": task_spec.runner_module if task_spec else None,
            }
            return ModelResponse(text=[sample_result.text], reasonings=sample_result.reasonings or [])

        payload = build_harbor_payload(doc)
        if doc.specific is None:
            doc.specific = {}

        with tempfile.TemporaryDirectory(prefix="lighteval_harbor_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "sample_input.json"
            output_path = tmpdir_path / "sample_output.json"
            reward_path = tmpdir_path / "sample_reward.json"

            input_path.write_text(json.dumps(payload.__dict__, indent=2, sort_keys=True))

            command_template = self._resolve_command_template(doc.task_name)
            command = command_template.format(
                input_path=shlex.quote(str(input_path)),
                output_path=shlex.quote(str(output_path)),
                reward_path=shlex.quote(str(reward_path)),
                task_name=doc.task_name,
                doc_id=doc.id,
            )
            env = os.environ.copy()
            env.update(self.config.environment)
            if self.config.reference_model:
                env["HARBOR_REFERENCE_MODEL"] = self.config.reference_model
                env["REFERENCE_MODEL"] = self.config.reference_model
            start = time.perf_counter()

            process = subprocess.run(
                command,
                shell=True,
                cwd=self.config.working_dir,
                timeout=self.config.timeout_seconds,
                capture_output=True,
                text=True,
                env=env,
            )
            wall_time_sec = time.perf_counter() - start

            if process.returncode != 0 and self.config.fail_on_command_error:
                raise RuntimeError(
                    "Harbor command failed.\n"
                    f"Command: {command}\n"
                    f"Return code: {process.returncode}\n"
                    f"Stdout: {process.stdout}\n"
                    f"Stderr: {process.stderr}"
                )

            sample_result = self._parse_harbor_output(output_path)
            sample_result.operational_metrics.setdefault("wall_time_sec", wall_time_sec)
            if reward_path.exists():
                reward_json = json.loads(reward_path.read_text())
                if isinstance(reward_json, dict):
                    for metric_name, metric_value in reward_json.items():
                        try:
                            sample_result.operational_metrics[metric_name] = float(metric_value)
                        except (TypeError, ValueError):
                            continue

            task_spec = self._task_harbor_specs.get(doc.task_name) or self._task_harbor_specs.get(
                doc.task_name.split("|", 1)[0]
            )
            doc.specific["harbor"] = {
                "operational_metrics": sample_result.operational_metrics,
                "runner_stdout": process.stdout,
                "runner_stderr": process.stderr,
                "command": command,
                "task_runner_module": task_spec.runner_module if task_spec else None,
            }

            return ModelResponse(text=[sample_result.text], reasonings=sample_result.reasonings or [])

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
        return [self._invoke_harbor(doc) for doc in docs]

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        raise NotImplementedError(
            "HarborModel supports GENERATIVE sampling only. "
            "Use a generative Harbor-backed task or another model backend for LOGPROBS."
        )

    @cached(SamplingMethod.PERPLEXITY)
    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        raise NotImplementedError(
            "HarborModel supports GENERATIVE sampling only. "
            "Use a generative Harbor-backed task or another model backend for PERPLEXITY."
        )
