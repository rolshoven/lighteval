"""agentskills.io discovery, catalog, and lazy activation helpers for Harbor agents."""

from __future__ import annotations

import logging
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ACTIVATE_SKILL_TOOL = "activate_skill"
SKILL_RESOURCE_SUBDIRS = ("scripts", "references", "assets")


@dataclass(frozen=True)
class SkillRecord:
    name: str
    description: str
    location: str
    skill_dir: str
    body: str
    allowed_tools: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


def _parse_yaml_scalar(block: str, key: str) -> str | None:
    pattern = re.compile(
        rf"^{re.escape(key)}:\s*(.*)$",
        re.MULTILINE,
    )
    match = pattern.search(block)
    if not match:
        return None
    raw = match.group(1).strip()
    if not raw or raw in {"|", "|-", ">", ">-", "|+"}:
        return _parse_yaml_block_scalar(block, key)
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw


def _is_yaml_block_indicator(value: str) -> bool:
    if not value:
        return True
    if value in {"|", "|-", "|+", ">", ">-", ">+"}:
        return True
    return value[0] in {"|", ">"}


def _parse_yaml_block_scalar(block: str, key: str) -> str | None:
    lines = block.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith(f"{key}:"):
            continue
        remainder = line.split(":", 1)[1].strip()
        if not _is_yaml_block_indicator(remainder):
            return remainder.strip("\"'") if remainder else None
        collected: list[str] = []
        for follow in lines[index + 1 :]:
            if follow and not follow[0].isspace():
                break
            collected.append(follow.strip())
        return "\n".join(collected).strip() or None
    return None


def split_skill_md(text: str) -> tuple[dict[str, str], str]:
    stripped = text.lstrip("\ufeff")
    if not stripped.startswith("---"):
        return {}, stripped
    parts = stripped.split("---", 2)
    if len(parts) < 3:
        return {}, stripped
    frontmatter_block = parts[1]
    body = parts[2].lstrip("\n")
    frontmatter: dict[str, str] = {}
    for key in ("name", "description", "license", "compatibility"):
        value = _parse_yaml_scalar(frontmatter_block, key)
        if value is not None:
            frontmatter[key] = value
    allowed = _parse_yaml_scalar(frontmatter_block, "allowed-tools")
    if allowed is not None:
        frontmatter["allowed-tools"] = allowed
    return frontmatter, body


def parse_skill_md(path: Path) -> SkillRecord:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = split_skill_md(text)
    skill_dir = path.parent
    name = frontmatter.get("name") or skill_dir.name
    description = frontmatter.get("description") or ""
    allowed_raw = frontmatter.get("allowed-tools", "")
    allowed_tools = [part for part in allowed_raw.split() if part]
    metadata = {
        key: value
        for key, value in frontmatter.items()
        if key not in {"name", "description", "allowed-tools"}
    }
    return SkillRecord(
        name=name,
        description=description,
        location=str(path.resolve()),
        skill_dir=str(skill_dir.resolve()),
        body=body.strip(),
        allowed_tools=allowed_tools,
        metadata=metadata,
    )


def validate_skill(record: SkillRecord) -> bool:
    if not record.description.strip():
        logger.warning("Skipping skill %s: empty description", record.name)
        return False
    dir_name = Path(record.skill_dir).name
    if record.name != dir_name:
        logger.warning(
            "Skill name %r does not match directory %r; loading anyway",
            record.name,
            dir_name,
        )
    if len(record.name) > 64:
        logger.warning("Skill name %r exceeds 64 characters; loading anyway", record.name)
    return True


def discover_skills(skills_root: Path) -> list[SkillRecord]:
    if not skills_root.is_dir():
        return []
    records: list[SkillRecord] = []
    for child in sorted(skills_root.iterdir()):
        if not child.is_dir():
            continue
        skill_md = child / "SKILL.md"
        if not skill_md.is_file():
            continue
        record = parse_skill_md(skill_md)
        if validate_skill(record):
            records.append(record)
    return records


def build_skills_catalog(skills: list[SkillRecord]) -> str:
    if not skills:
        return ""
    lines = ["<available_skills>"]
    for skill in skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{skill.name}</name>")
        lines.append(f"    <description>{skill.description}</description>")
        lines.append(f"    <location>{skill.location}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def skills_catalog_instructions() -> str:
    return (
        "The following skills provide specialized instructions for specific tasks. "
        "When a task matches a skill's description, call the activate_skill tool "
        "with that skill's name to load its full instructions before proceeding."
    )


def _list_skill_resources(skill_dir: Path) -> list[str]:
    resources: list[str] = []
    for subdir_name in SKILL_RESOURCE_SUBDIRS:
        subdir = skill_dir / subdir_name
        if not subdir.is_dir():
            continue
        for path in sorted(subdir.rglob("*")):
            if path.is_file():
                resources.append(str(path.relative_to(skill_dir)))
    return resources


def format_activated_skill(record: SkillRecord) -> str:
    resources = _list_skill_resources(Path(record.skill_dir))
    lines = [
        f'<skill_content name="{record.name}">',
        record.body,
        "",
        f"Skill directory: {record.skill_dir}",
        "Relative paths in this skill are relative to the skill directory.",
    ]
    if resources:
        lines.append("")
        lines.append("<skill_resources>")
        for resource in resources:
            lines.append(f"  <file>{resource}</file>")
        lines.append("</skill_resources>")
    lines.append("</skill_content>")
    return "\n".join(lines)


def build_activate_skill_tool_schema(skill_names: list[str]) -> dict[str, Any] | None:
    if not skill_names:
        return None
    return {
        "type": "function",
        "function": {
            "name": ACTIVATE_SKILL_TOOL,
            "description": (
                "Load the full instructions for an agent skill by name. "
                "Use when the current task matches a skill in the available_skills catalog."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name from the catalog.",
                        "enum": skill_names,
                    },
                },
                "required": ["name"],
            },
        },
    }


def activate_skill_in_messages(
    *,
    skill_name: str,
    skills_by_name: dict[str, SkillRecord],
    activated: set[str],
    messages: list[dict[str, Any]],
) -> str:
    if skill_name in activated:
        return f"Skill '{skill_name}' is already active in this session."
    record = skills_by_name.get(skill_name)
    if record is None:
        known = ", ".join(sorted(skills_by_name)) or "(none)"
        return f"Unknown skill: {skill_name}. Known skills: {known}"
    activated.add(skill_name)
    content = format_activated_skill(record)
    messages.append({"role": "user", "content": content})
    return f"Activated skill '{skill_name}'."


def resolve_skills_dir(*candidates: str | Path | None) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_dir():
            return path
    for env_key in ("LIGHTEVAL_HARBOR_SKILLS_DIR", "LIGHEVAL_HARBOR_SKILLS_DIR"):
        env_value = os.getenv(env_key)
        if env_value:
            path = Path(env_value)
            if path.is_dir():
                return path
    return None


def sandbox_skills_copy_shell(host_dir: str) -> str:
    quoted = shlex.quote(host_dir)
    return (
        f"mkdir -p /workspace/skills && cp -R {quoted}/* /workspace/skills/ 2>/dev/null || true"
    )
