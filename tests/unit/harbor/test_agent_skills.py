from pathlib import Path

from lighteval.harbor import agent_skills


FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "skills"


def test_parse_skill_md_extracts_frontmatter_and_body():
    record = agent_skills.parse_skill_md(FIXTURES_ROOT / "lexam" / "SKILL.md")
    assert record.name == "lexam"
    assert "structured reasoning" in record.description
    assert "LEXam fixture skill" in record.body
    assert record.allowed_tools == ["execute_bash"]


def test_discover_skills_finds_fixture_skills():
    skills = agent_skills.discover_skills(FIXTURES_ROOT)
    names = {skill.name for skill in skills}
    assert names == {"lexam", "opencaselaw"}


def test_build_skills_catalog_includes_name_and_description():
    skills = agent_skills.discover_skills(FIXTURES_ROOT)
    catalog = agent_skills.build_skills_catalog(skills)
    assert "<available_skills>" in catalog
    assert "<name>lexam</name>" in catalog
    assert "structured reasoning" in catalog


def test_format_activated_skill_omits_yaml_frontmatter():
    record = agent_skills.parse_skill_md(FIXTURES_ROOT / "lexam" / "SKILL.md")
    formatted = agent_skills.format_activated_skill(record)
    assert "---" not in formatted.split("<skill_content")[0]
    assert "LEXam fixture skill" in formatted
    assert '<skill_content name="lexam">' in formatted


def test_activate_skill_in_messages_dedupes_second_activation():
    skills = agent_skills.discover_skills(FIXTURES_ROOT)
    skills_by_name = {skill.name: skill for skill in skills}
    messages: list[dict] = []
    activated: set[str] = set()

    first = agent_skills.activate_skill_in_messages(
        skill_name="lexam",
        skills_by_name=skills_by_name,
        activated=activated,
        messages=messages,
    )
    second = agent_skills.activate_skill_in_messages(
        skill_name="lexam",
        skills_by_name=skills_by_name,
        activated=activated,
        messages=messages,
    )

    assert "Activated skill" in first
    assert "already active" in second
    assert len(messages) == 1


def test_build_activate_skill_tool_schema_uses_enum():
    schema = agent_skills.build_activate_skill_tool_schema(["lexam", "opencaselaw"])
    assert schema is not None
    enum_values = schema["function"]["parameters"]["properties"]["name"]["enum"]
    assert enum_values == ["lexam", "opencaselaw"]


def test_build_activate_skill_tool_schema_none_when_empty():
    assert agent_skills.build_activate_skill_tool_schema([]) is None
