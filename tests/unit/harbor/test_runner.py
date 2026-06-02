from lighteval.harbor.agent_spec import HarborAgentSpec
from lighteval.harbor.runner import resolve_command_template


def test_resolve_command_template_priority_order():
    spec = HarborAgentSpec(
        runner_module="lighteval.tasks.multilingual.tasks.swiss_legal.lexam_subprocess_runner",
    )
    command = resolve_command_template(
        cli_template="echo from-cli",
        cli_agent_runner="/tmp/cli_runner.py",
        model_agent_runner="/tmp/model_runner.py",
        model_command_template="echo from-model-template",
        task_spec=spec,
    )
    assert command == "echo from-cli"

    command = resolve_command_template(
        cli_template=None,
        cli_agent_runner="/tmp/cli_runner.py",
        model_agent_runner="/tmp/model_runner.py",
        model_command_template="echo from-model-template",
        task_spec=spec,
    )
    assert "/tmp/cli_runner.py" in command

    command = resolve_command_template(
        cli_template=None,
        cli_agent_runner=None,
        model_agent_runner="/tmp/model_runner.py",
        model_command_template="echo from-model-template",
        task_spec=spec,
    )
    assert "/tmp/model_runner.py" in command

    command = resolve_command_template(
        cli_template=None,
        cli_agent_runner=None,
        model_agent_runner=None,
        model_command_template="echo from-model-template",
        task_spec=spec,
    )
    assert command == "echo from-model-template"

    command = resolve_command_template(
        cli_template=None,
        cli_agent_runner=None,
        model_agent_runner=None,
        model_command_template=None,
        task_spec=spec,
    )
    assert "-m lighteval.tasks.multilingual.tasks.swiss_legal.lexam_subprocess_runner" in command
    assert "--harness" not in command


def test_resolve_command_template_without_task_spec_raises():
    try:
        resolve_command_template(
            cli_template=None,
            cli_agent_runner=None,
            model_agent_runner=None,
            model_command_template=None,
            task_spec=None,
        )
    except ValueError as exc:
        assert "No Harbor runner found for this task" in str(exc)
        return
    raise AssertionError("Expected ValueError when no command template and no task spec are provided.")
