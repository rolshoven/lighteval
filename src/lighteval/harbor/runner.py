import shlex
import sys

from lighteval.harbor.agent_spec import HarborAgentSpec


def _external_runner_command_template(runner_path: str) -> str:
    quoted_python = shlex.quote(sys.executable)
    quoted_runner = shlex.quote(runner_path)
    return (
        f"{quoted_python} {quoted_runner} "
        "--input-path {input_path} --output-path {output_path}"
    )


def command_template_for_task_agent(spec: HarborAgentSpec) -> str:
    quoted_python = shlex.quote(sys.executable)
    quoted_module = shlex.quote(spec.runner_module)
    return (
        f"{quoted_python} -m {quoted_module} "
        "--input-path {input_path} --output-path {output_path}"
    )


def resolve_command_template(
    *,
    cli_template: str | None,
    cli_agent_runner: str | None,
    model_agent_runner: str | None,
    model_command_template: str | None,
    task_spec: HarborAgentSpec | None,
) -> str:
    if cli_template:
        return cli_template

    if cli_agent_runner:
        return _external_runner_command_template(cli_agent_runner)

    if model_agent_runner:
        return _external_runner_command_template(model_agent_runner)

    if model_command_template:
        return model_command_template

    if task_spec is not None:
        return command_template_for_task_agent(task_spec)

    raise ValueError(
        "No Harbor runner found for this task. "
        "Either set a task-level harbor_agent on its task config, or pass "
        "--harbor-command-template / --harbor-agent-runner."
    )
