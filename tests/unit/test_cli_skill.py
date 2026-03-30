"""Tests for the `datachain skill` CLI command."""

from pathlib import Path
from unittest.mock import patch

import pytest

from datachain.cli.parser import get_parser

# ---------------------------------------------------------------------------
# Argument parsing tests
# ---------------------------------------------------------------------------


def test_skill_install_defaults():
    parser = get_parser()
    args = parser.parse_args(["skill", "install"])
    assert args.command == "skill"
    assert args.skill_cmd == "install"
    assert args.skills is None
    assert args.target == "claude"
    assert args.local is False


def test_skill_install_one_skill():
    parser = get_parser()
    args = parser.parse_args(["skill", "install", "graph"])
    assert args.skills == "graph"


def test_skill_install_multiple_skills():
    parser = get_parser()
    args = parser.parse_args(["skill", "install", "graph,core"])
    assert args.skills == "graph,core"


def test_skill_install_target_cursor():
    parser = get_parser()
    args = parser.parse_args(["skill", "install", "--target", "cursor"])
    assert args.target == "cursor"


def test_skill_install_target_codex():
    parser = get_parser()
    args = parser.parse_args(["skill", "install", "--target", "codex"])
    assert args.target == "codex"


def test_skill_install_local_flag():
    parser = get_parser()
    args = parser.parse_args(["skill", "install", "--local"])
    assert args.local is True


def test_skill_install_invalid_target(capsys):
    parser = get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["skill", "install", "--target", "vscode"])


def test_skill_list_no_args():
    parser = get_parser()
    args = parser.parse_args(["skill", "list"])
    assert args.command == "skill"
    assert args.skill_cmd == "list"


# ---------------------------------------------------------------------------
# install_skills() functional tests
# ---------------------------------------------------------------------------


def test_install_invalid_skill_raises(tmp_path, fake_skills_src, fake_home):
    from datachain.cli.commands.skill import install_skills

    with (
        patch("datachain.cli.commands.skill._skills_src", return_value=fake_skills_src),
        patch("pathlib.Path.home", return_value=fake_home),
        pytest.raises(ValueError, match=r"Unknown skill.*nope"),
    ):
        install_skills(skills="nope", target="claude", local=False)


ALL_SKILLS = ("core", "graph", "jobs")


def _make_fake_skills_src(tmp_path: Path) -> Path:
    """Create a minimal skills source tree for testing."""
    skills_src = tmp_path / "skills_src"
    for skill_name in ALL_SKILLS:
        skill_dir = skills_src / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: datachain-{skill_name}\n"
            f"description: Test skill {skill_name}\n"
            f"---\n# {skill_name}\n"
            "```bash\npython3 scripts/plan.py\n```\n"
        )
        scripts = skill_dir / "scripts"
        scripts.mkdir()
        (scripts / "plan.py").write_text("# stub\n")
        # Add __pycache__ junk to verify it gets filtered
        pycache = scripts / "__pycache__"
        pycache.mkdir()
        (pycache / "plan.cpython-312.pyc").write_bytes(b"\x00")
    return skills_src


@pytest.fixture()
def fake_skills_src(tmp_path):
    return _make_fake_skills_src(tmp_path)


@pytest.fixture()
def fake_home(tmp_path):
    return tmp_path / "home"


def _run_install(
    fake_skills_src,
    fake_home,
    skills,
    target,
    local,
    monkeypatch=None,
    project_dir=None,
):
    from datachain.cli.commands.skill import install_skills

    with (
        patch("datachain.cli.commands.skill._skills_src", return_value=fake_skills_src),
        patch("pathlib.Path.home", return_value=fake_home),
    ):
        if local and project_dir:
            monkeypatch.chdir(project_dir)
        install_skills(skills=skills, target=target, local=local)


# --- claude, global ---


def test_install_all_claude_global(tmp_path, fake_skills_src, fake_home):
    _run_install(fake_skills_src, fake_home, skills=None, target="claude", local=False)

    skills_base = fake_home / ".claude" / "skills"

    for skill in ALL_SKILLS:
        assert (skills_base / skill / "SKILL.md").exists()

    # graph should have its scripts directory too
    assert (skills_base / "graph" / "scripts" / "plan.py").exists()

    # Claude global installs should NOT create commands
    # (~/.claude/commands/ is not a real Claude Code path)
    assert not (fake_home / ".claude" / "commands").exists()

    # {skill_dir} should be resolved to absolute path in installed SKILL.md
    content = (skills_base / "graph" / "SKILL.md").read_text()
    assert "{skill_dir}" not in content
    assert "scripts/plan.py" in content


def test_install_only_core_claude_global(tmp_path, fake_skills_src, fake_home):
    _run_install(
        fake_skills_src, fake_home, skills="core", target="claude", local=False
    )

    skills_base = fake_home / ".claude" / "skills"
    assert (skills_base / "core" / "SKILL.md").exists()
    assert not (skills_base / "graph").exists()


def test_install_only_graph_claude_global(tmp_path, fake_skills_src, fake_home):
    _run_install(
        fake_skills_src, fake_home, skills="graph", target="claude", local=False
    )

    skills_base = fake_home / ".claude" / "skills"
    assert (skills_base / "graph" / "SKILL.md").exists()
    assert not (skills_base / "core").exists()


# --- cursor, global ---


def test_install_all_cursor_global(tmp_path, fake_skills_src, fake_home):
    _run_install(fake_skills_src, fake_home, skills=None, target="cursor", local=False)

    skills_base = fake_home / ".cursor" / "skills"
    rules_base = fake_home / ".cursor" / "rules"

    for skill in ALL_SKILLS:
        assert (skills_base / skill / "SKILL.md").exists()
        mdc_file = rules_base / f"datachain-{skill}.mdc"
        assert mdc_file.exists()
        # Verify Cursor .mdc has correct frontmatter format
        content = mdc_file.read_text()
        assert "alwaysApply: true" in content
        assert "description: Test skill" in content
        # Original SKILL.md frontmatter fields should NOT appear
        assert "triggers:" not in content


# --- codex, global ---


def test_install_all_codex_global(tmp_path, fake_skills_src, fake_home):
    _run_install(fake_skills_src, fake_home, skills=None, target="codex", local=False)

    skills_base = fake_home / ".codex" / "skills"
    for skill in ALL_SKILLS:
        assert (skills_base / skill / "SKILL.md").exists()

    # codex has no commands dir
    assert not (fake_home / ".codex" / "commands").exists()


# --- local install ---


def test_install_claude_local(tmp_path, fake_skills_src, fake_home, monkeypatch):
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    from datachain.cli.commands.skill import install_skills

    with patch(
        "datachain.cli.commands.skill._skills_src", return_value=fake_skills_src
    ):
        monkeypatch.chdir(project_dir)
        install_skills(skills=None, target="claude", local=True)

    skills_base = project_dir / ".claude" / "skills"
    commands_base = project_dir / ".claude" / "commands"

    for skill in ALL_SKILLS:
        assert (skills_base / skill / "SKILL.md").exists()
        # Local Claude installs SHOULD create commands
        assert (commands_base / f"datachain-{skill}.md").exists()

    # Nothing written to home
    assert not (fake_home / ".claude").exists()


# --- __pycache__ filtering ---


def test_pycache_not_copied(tmp_path, fake_skills_src, fake_home):
    """Verify __pycache__ and .pyc files are not copied to the destination."""
    _run_install(fake_skills_src, fake_home, skills=None, target="claude", local=False)

    skills_base = fake_home / ".claude" / "skills"
    # Scripts should exist
    assert (skills_base / "graph" / "scripts" / "plan.py").exists()
    # But __pycache__ should NOT
    assert not (skills_base / "graph" / "scripts" / "__pycache__").exists()


# ---------------------------------------------------------------------------
# list_skills() smoke test
# ---------------------------------------------------------------------------


def test_list_skills_output(capsys):
    from datachain.cli.commands.skill import list_skills

    list_skills()
    out = capsys.readouterr().out
    assert "core" in out
    assert "graph" in out
    assert "jobs" in out
    assert "claude" in out
    assert "cursor" in out
    assert "codex" in out


# ---------------------------------------------------------------------------
# install edge cases
# ---------------------------------------------------------------------------


def test_install_missing_source_returns_nonzero(tmp_path, fake_home):
    """If a skill source dir is missing, install returns 1."""
    from datachain.cli.commands.skill import install_skills

    # Create a skills_src with only "core" — graph and jobs missing
    skills_src = tmp_path / "partial_src"
    core = skills_src / "core"
    core.mkdir(parents=True)
    (core / "SKILL.md").write_text("---\nname: core\n---\n# core\n")

    with (
        patch(
            "datachain.cli.commands.skill._skills_src",
            return_value=skills_src,
        ),
        patch("pathlib.Path.home", return_value=fake_home),
    ):
        result = install_skills(skills=None, target="claude", local=False)

    # core installed, but graph+jobs missing → non-zero
    assert result == 1
    assert (fake_home / ".claude" / "skills" / "core" / "SKILL.md").exists()


def test_transform_cursor_mdc_no_frontmatter():
    """SKILL.md without frontmatter markers produces valid .mdc."""
    from datachain.cli.commands.skill import _transform_cursor_mdc

    p = Path(__file__).parent / "_test_no_fm.md"
    try:
        p.write_text("# No frontmatter here\nJust body.\n")
        result = _transform_cursor_mdc(p)
        assert "alwaysApply: true" in result
        assert "# No frontmatter here" in result
        # description should be empty
        assert "description: \n" in result
    finally:
        p.unlink(missing_ok=True)


def test_transform_cursor_mdc_missing_description():
    """SKILL.md with frontmatter but no description field."""
    from datachain.cli.commands.skill import _transform_cursor_mdc

    p = Path(__file__).parent / "_test_no_desc.md"
    try:
        p.write_text("---\nname: test-skill\n---\n# Body\n")
        result = _transform_cursor_mdc(p)
        assert "alwaysApply: true" in result
        assert "description: \n" in result
    finally:
        p.unlink(missing_ok=True)


def test_skills_src_resolves_real_package():
    """importlib.resources finds the bundled SKILL.md files."""
    from datachain.cli.commands.skill import SKILLS, _skills_src

    src = _skills_src()
    assert src.is_dir(), f"_skills_src() returned non-directory: {src}"
    for skill_name in SKILLS:
        skill_md = src / skill_name / "SKILL.md"
        assert skill_md.exists(), f"Missing bundled {skill_name}/SKILL.md"
