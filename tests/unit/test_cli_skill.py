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
    args = parser.parse_args(["skill", "install", "dc-graph"])
    assert args.skills == "dc-graph"


def test_skill_install_multiple_skills():
    parser = get_parser()
    args = parser.parse_args(["skill", "install", "dc-graph,dc-core"])
    assert args.skills == "dc-graph,dc-core"


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


def _make_fake_skills_src(tmp_path: Path) -> Path:
    """Create a minimal skills source tree for testing."""
    skills_src = tmp_path / "skills_src"
    for skill_name in ("dc-core", "dc-graph"):
        skill_dir = skills_src / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill_name}\n---\n# {skill_name}\n"
        )
        scripts = skill_dir / "scripts"
        scripts.mkdir()
        (scripts / "graph.py").write_text("# stub\n")
    return skills_src


@pytest.fixture()
def fake_skills_src(tmp_path):
    return _make_fake_skills_src(tmp_path)


@pytest.fixture()
def fake_home(tmp_path):
    return tmp_path / "home"


def _run_install(fake_skills_src, fake_home, skills, target, local, project_dir=None):
    from datachain.cli.commands.skill import install_skills

    with (
        patch("datachain.cli.commands.skill._skills_src", return_value=fake_skills_src),
        patch("pathlib.Path.home", return_value=fake_home),
    ):
        if local and project_dir:
            orig_cwd = Path.cwd()
            import os

            os.chdir(project_dir)
            try:
                install_skills(skills=skills, target=target, local=local)
            finally:
                os.chdir(orig_cwd)
        else:
            install_skills(skills=skills, target=target, local=local)


# --- claude, global ---


def test_install_all_claude_global(tmp_path, fake_skills_src, fake_home, capsys):
    _run_install(fake_skills_src, fake_home, skills=None, target="claude", local=False)

    skills_base = fake_home / ".claude" / "skills"
    commands_base = fake_home / ".claude" / "commands"

    for skill in ("dc-core", "dc-graph"):
        assert (skills_base / skill / "SKILL.md").exists()
        assert (commands_base / f"{skill}.md").exists()

    # dc-graph should have its scripts directory too
    assert (skills_base / "dc-graph" / "scripts" / "graph.py").exists()


def test_install_only_core_claude_global(tmp_path, fake_skills_src, fake_home):
    _run_install(fake_skills_src, fake_home, skills="dc-core", target="claude", local=False)

    skills_base = fake_home / ".claude" / "skills"
    assert (skills_base / "dc-core" / "SKILL.md").exists()
    assert not (skills_base / "dc-graph").exists()


def test_install_only_graph_claude_global(tmp_path, fake_skills_src, fake_home):
    _run_install(fake_skills_src, fake_home, skills="dc-graph", target="claude", local=False)

    skills_base = fake_home / ".claude" / "skills"
    assert (skills_base / "dc-graph" / "SKILL.md").exists()
    assert not (skills_base / "dc-core").exists()


# --- cursor, global ---


def test_install_all_cursor_global(tmp_path, fake_skills_src, fake_home):
    _run_install(fake_skills_src, fake_home, skills=None, target="cursor", local=False)

    skills_base = fake_home / ".cursor" / "skills"
    rules_base = fake_home / ".cursor" / "rules"

    for skill in ("dc-core", "dc-graph"):
        assert (skills_base / skill / "SKILL.md").exists()
        assert (rules_base / f"{skill}.mdc").exists()


# --- codex, global ---


def test_install_all_codex_global(tmp_path, fake_skills_src, fake_home):
    _run_install(fake_skills_src, fake_home, skills=None, target="codex", local=False)

    skills_base = fake_home / ".codex" / "skills"
    for skill in ("dc-core", "dc-graph"):
        assert (skills_base / skill / "SKILL.md").exists()

    # codex has no commands dir
    assert not (fake_home / ".codex" / "commands").exists()


# --- local install ---


def test_install_claude_local(tmp_path, fake_skills_src, fake_home):
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    from datachain.cli.commands.skill import install_skills

    with patch(
        "datachain.cli.commands.skill._skills_src", return_value=fake_skills_src
    ):
        import os

        orig = os.getcwd()
        os.chdir(project_dir)
        try:
            install_skills(skills=None, target="claude", local=True)
        finally:
            os.chdir(orig)

    skills_base = project_dir / ".claude" / "skills"
    commands_base = project_dir / ".claude" / "commands"

    for skill in ("dc-core", "dc-graph"):
        assert (skills_base / skill / "SKILL.md").exists()
        assert (commands_base / f"{skill}.md").exists()

    # Nothing written to home
    assert not (fake_home / ".claude").exists()


# ---------------------------------------------------------------------------
# list_skills() smoke test
# ---------------------------------------------------------------------------


def test_list_skills_output(capsys):
    from datachain.cli.commands.skill import list_skills

    list_skills()
    out = capsys.readouterr().out
    assert "core" in out
    assert "graph" in out
    assert "claude" in out
    assert "cursor" in out
    assert "codex" in out
