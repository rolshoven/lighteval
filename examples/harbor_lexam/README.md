# Harbor LEXam examples

Optional helpers for LEXam with Harbor. The reference Harbor agent for `lighteval harbor job` is
[`lexam_harbor_agent.py`](../../src/lighteval/tasks/multilingual/tasks/swiss_legal/lexam_harbor_agent.py).

## Commands

**Subprocess bridge** (`harbor run`) — resolves runner in order: `--harbor-command-template`, `--harbor-agent-runner`, model args, then task `harbor_agent.runner_module`.

**Sandbox job** (`harbor job`):

- Default for LEXam: task `harbor_agent_import` → `LexamHarborAgent`
- Built-in Harbor agent: `--harbor-agent terminus-2`
- Override: `--harbor-agent-import pkg.module:Class`

**Score only:** `lighteval harbor score … --harbor-job-dir ./path/to/job`

> Upstream Harbor also has `harbor run`; in lighteval, `harbor job` is the sandbox wrapper and `harbor run` is the subprocess bridge.

## Skills and tools by agent type

| Agent | Who provides skills/tools |
|-------|---------------------------|
| `lexam` (`LexamHarborAgent`) | lighteval: `HarborAgentSpec.skills_dir`, [`agent_skills`](../../src/lighteval/harbor/agent_skills.py), OpenCaseLaw tools, optional MCP via `mcp_config_path` |
| `terminus-2` (and other `-a` builtins) | Harbor upstream only — `harbor run -p … -a terminus-2 -m …` |
| `harbor run` | Subprocess `runner_module` only; no sandbox skills |

Domain skills live under [`harbor_skills/`](../../src/lighteval/tasks/multilingual/tasks/swiss_legal/harbor_skills/) (e.g. `opencaselaw/SKILL.md`).

## Scaffold Harbor task folders

```bash
python examples/harbor_lexam/scaffold_tasks.py ./harbor_lexam_tasks --max-samples 10
```

## External runner template

[`run_custom_agent.py`](./run_custom_agent.py) — starting point for a custom subprocess runner while keeping lighteval scoring.
