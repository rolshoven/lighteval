# Harbor skills for LEXam

Optional [Agent Skills](https://agentskills.io/specification) live here. LEXam exam conventions (structured reasoning, `Final Answer: ###X###`) are baked into `LexamHarborAgent`'s system prompt, not loaded as a skill.

```
harbor_skills/
└── opencaselaw/SKILL.md   # lazy-loaded via activate_skill when case-law lookup is needed
```

- `name` in YAML frontmatter must match the parent directory name.
- `description` is shown in the agent skills catalog at run start.
- `allowed-tools` is an optional hint (tools are registered in Python, not in frontmatter).

`LexamHarborAgent` discovers skills via `lighteval.harbor.agent_skills` and loads full bodies on demand with `activate_skill`.
