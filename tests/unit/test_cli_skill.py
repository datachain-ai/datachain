"""Tests for the `datachain skill` CLI command."""

from pathlib import Path
from unittest.mock import patch

import pytest

from datachain.cli.commands.skill import DEFAULT_SKILLS, SKILLS
from datachain.cli.parser import get_parser


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
    args = parser.parse_args(["skill", "install", "knowledge"])
    assert args.skills == "knowledge"


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


def test_skill_install_target_pi():
    parser = get_parser()
    args = parser.parse_args(["skill", "install", "--target", "pi"])
    assert args.target == "pi"


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


def test_install_invalid_skill_raises(tmp_path, fake_skills_src, fake_home):
    from datachain.cli.commands.skill import install_skills

    with (
        patch("datachain.cli.commands.skill._skills_src", return_value=fake_skills_src),
        patch("pathlib.Path.home", return_value=fake_home),
        pytest.raises(ValueError, match=r"Unknown skill.*nope"),
    ):
        install_skills(skills="nope", target="claude", local=False)


def _make_fake_skills_src(tmp_path: Path) -> Path:
    """Create a minimal skills source tree for testing."""
    skills_src = tmp_path / "skills_src"
    for skill_name in SKILLS:
        skill_dir = skills_src / skill_name
        skill_dir.mkdir(parents=True)
        name = "datachain" if skill_name == "datachain" else f"datachain-{skill_name}"
        resource_dir = "skill_dir" if skill_name == "datachain" else "core_skill_dir"
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\n"
            f"description: Test skill {skill_name}\n"
            f"---\n# {skill_name}\n"
            "```bash\npython3 scripts/plan.py\n```\n"
            f"Read `{{{resource_dir}}}/SDK.md`.\n"
        )
        if skill_name != "datachain":
            (skill_dir / "SDK.md").write_text(f"# {skill_name} rules\n")
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


def test_install_defaults_claude_global(tmp_path, fake_skills_src, fake_home):
    _run_install(fake_skills_src, fake_home, skills=None, target="claude", local=False)

    skills_base = fake_home / ".claude" / "skills"

    for skill in DEFAULT_SKILLS:
        assert (skills_base / skill / "SKILL.md").exists()

    assert {p.name for p in skills_base.iterdir()} == set(DEFAULT_SKILLS)
    assert (skills_base / "datachain" / "scripts" / "plan.py").exists()

    # Claude global installs should NOT create commands
    # (~/.claude/commands/ is not a real Claude Code path)
    assert not (fake_home / ".claude" / "commands").exists()

    # {skill_dir} should be resolved to absolute path in installed SKILL.md
    content = (skills_base / "datachain" / "SKILL.md").read_text(encoding="utf-8")
    assert "{skill_dir}" not in content
    assert "scripts/plan.py" in content
    assert "{core_skill_dir}" not in content
    assert f"{(skills_base / 'datachain').resolve().as_posix()}/SDK.md" in content


def test_install_only_core_claude_global(tmp_path, fake_skills_src, fake_home):
    _run_install(
        fake_skills_src, fake_home, skills="core", target="claude", local=False
    )

    skills_base = fake_home / ".claude" / "skills"
    assert (skills_base / "core" / "SKILL.md").exists()
    assert not (skills_base / "knowledge").exists()


def test_install_only_graph_claude_global(tmp_path, fake_skills_src, fake_home):
    _run_install(
        fake_skills_src, fake_home, skills="knowledge", target="claude", local=False
    )

    skills_base = fake_home / ".claude" / "skills"
    assert (skills_base / "knowledge" / "SKILL.md").exists()
    assert (skills_base / "core" / "SDK.md").exists()


def test_uninstall_core_refused_while_knowledge_installed(
    tmp_path, fake_skills_src, fake_home
):
    from datachain.cli.commands.skill import uninstall_skills

    _run_install(
        fake_skills_src, fake_home, skills="knowledge", target="claude", local=False
    )
    skills_base = fake_home / ".claude" / "skills"

    with (
        patch("pathlib.Path.home", return_value=fake_home),
        pytest.raises(ValueError, match=r"Cannot uninstall core: knowledge"),
    ):
        uninstall_skills(skills="core", target="claude", local=False)
    assert (skills_base / "core" / "SDK.md").exists()

    with patch("pathlib.Path.home", return_value=fake_home):
        uninstall_skills(skills="core,knowledge", target="claude", local=False)
    assert not (skills_base / "core").exists()
    assert not (skills_base / "knowledge").exists()


def test_install_knowledge_from_package_ships_core_sdk(tmp_path, fake_home):
    from datachain.cli.commands.skill import install_skills

    with patch("pathlib.Path.home", return_value=fake_home):
        install_skills(skills="knowledge", target="claude", local=False)

    skills_base = fake_home / ".claude" / "skills"
    assert (skills_base / "core" / "SDK.md").exists()
    content = (skills_base / "knowledge" / "SKILL.md").read_text(encoding="utf-8")
    assert f"{(skills_base / 'core').resolve().as_posix()}/SDK.md" in content


def test_install_defaults_cursor_global(tmp_path, fake_skills_src, fake_home):
    _run_install(fake_skills_src, fake_home, skills=None, target="cursor", local=False)

    skills_base = fake_home / ".cursor" / "skills"
    rules_base = fake_home / ".cursor" / "rules"

    for skill in DEFAULT_SKILLS:
        assert (skills_base / skill / "SKILL.md").exists()
        mdc_file = rules_base / f"datachain-{skill}.mdc"
        assert mdc_file.exists()
        # Verify Cursor .mdc has correct frontmatter format
        content = mdc_file.read_text(encoding="utf-8")
        assert "alwaysApply: true" in content
        assert "description: Test skill" in content
        # Original SKILL.md frontmatter fields should NOT appear
        assert "triggers:" not in content
        assert (skills_base / skill / "SDK.md").resolve().as_posix() in content


def test_install_defaults_codex_global(tmp_path, fake_skills_src, fake_home):
    _run_install(fake_skills_src, fake_home, skills=None, target="codex", local=False)

    skills_base = fake_home / ".codex" / "skills"
    for skill in DEFAULT_SKILLS:
        assert (skills_base / skill / "SKILL.md").exists()

    # codex has no commands dir
    assert not (fake_home / ".codex" / "commands").exists()


def test_install_defaults_pi_global(tmp_path, fake_skills_src, fake_home):
    _run_install(fake_skills_src, fake_home, skills=None, target="pi", local=False)

    skills_base = fake_home / ".pi" / "agent" / "skills"
    prompts_base = fake_home / ".pi" / "agent" / "prompts"

    for skill in DEFAULT_SKILLS:
        assert (skills_base / skill / "SKILL.md").exists()
        prompt_file = prompts_base / f"datachain-{skill}.md"
        assert prompt_file.exists()
        # No transform — Pi accepts SKILL.md frontmatter as-is.
        content = prompt_file.read_text(encoding="utf-8")
        assert "name: datachain\n" in content
        assert f"description: Test skill {skill}" in content

    # User-level layout MUST be ~/.pi/agent/..., not ~/.pi/...
    assert not (fake_home / ".pi" / "skills").exists()
    assert not (fake_home / ".pi" / "prompts").exists()


def test_install_pi_local(tmp_path, fake_skills_src, fake_home, monkeypatch):
    """--local mode for pi writes to .pi/skills/ + .pi/prompts/ (no `agent/`)."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    from datachain.cli.commands.skill import install_skills

    with patch(
        "datachain.cli.commands.skill._skills_src", return_value=fake_skills_src
    ):
        monkeypatch.chdir(project_dir)
        install_skills(skills=None, target="pi", local=True)

    skills_base = project_dir / ".pi" / "skills"
    prompts_base = project_dir / ".pi" / "prompts"

    for skill in DEFAULT_SKILLS:
        assert (skills_base / skill / "SKILL.md").exists()
        assert (prompts_base / f"datachain-{skill}.md").exists()

    # Repo-local --local mode MUST NOT use the user-level agent/ layout.
    assert not (project_dir / ".pi" / "agent").exists()
    # And the user home is untouched.
    assert not (fake_home / ".pi").exists()


def test_install_defaults_copilot_global(tmp_path, fake_skills_src, fake_home):
    _run_install(fake_skills_src, fake_home, skills=None, target="copilot", local=False)

    skills_base = fake_home / ".copilot" / "skills"
    instructions_base = fake_home / ".copilot" / "instructions"

    for skill in DEFAULT_SKILLS:
        # Full vendor copy of the skill dir (scripts, SKILL.md, etc.)
        assert (skills_base / skill / "SKILL.md").exists()
        # Per-skill .instructions.md file with applyTo frontmatter
        instr_file = instructions_base / f"datachain-{skill}.instructions.md"
        assert instr_file.exists()
        content = instr_file.read_text(encoding="utf-8")
        assert "applyTo: '**/*.py'" in content
        # Original SKILL.md frontmatter fields should be stripped
        assert "name: datachain-" not in content
        assert "description: Test skill" not in content
        assert (skills_base / skill / "SDK.md").resolve().as_posix() in content


def test_install_copilot_local_uses_github_path(
    tmp_path, fake_skills_src, fake_home, monkeypatch
):
    """--local mode for copilot writes to .github/instructions/ + .datachain/skills/
    (the canonical GitHub Copilot paths), not the ~/.copilot/ user-level layout."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    from datachain.cli.commands.skill import install_skills

    with patch(
        "datachain.cli.commands.skill._skills_src", return_value=fake_skills_src
    ):
        monkeypatch.chdir(project_dir)
        install_skills(skills=None, target="copilot", local=True)

    # Repo-local Copilot uses the canonical .github/instructions/ path,
    # NOT .copilot/instructions/.
    instructions_base = project_dir / ".github" / "instructions"
    vendor_base = project_dir / ".datachain" / "skills"

    for skill in DEFAULT_SKILLS:
        assert (vendor_base / skill / "SKILL.md").exists()
        instr_file = instructions_base / f"datachain-{skill}.instructions.md"
        assert instr_file.exists()
        assert "applyTo: '**/*.py'" in instr_file.read_text(encoding="utf-8")

    # The user-level layout MUST NOT be touched in --local mode.
    assert not (project_dir / ".copilot").exists()
    assert not (fake_home / ".copilot").exists()


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

    for skill in DEFAULT_SKILLS:
        assert (skills_base / skill / "SKILL.md").exists()
        # Local Claude installs SHOULD create commands
        assert (commands_base / f"datachain-{skill}.md").exists()

    # Nothing written to home
    assert not (fake_home / ".claude").exists()


def test_pycache_not_copied(tmp_path, fake_skills_src, fake_home):
    """Verify __pycache__ and .pyc files are not copied to the destination."""
    _run_install(fake_skills_src, fake_home, skills=None, target="claude", local=False)

    skills_base = fake_home / ".claude" / "skills"
    assert (skills_base / "datachain" / "scripts" / "plan.py").exists()
    assert not (skills_base / "datachain" / "scripts" / "__pycache__").exists()


def test_list_skills_output(capsys):
    from datachain.cli.commands.skill import list_skills

    list_skills()
    out = capsys.readouterr().out
    assert "datachain" in out
    assert "core" in out
    assert "knowledge" in out
    assert "claude" in out
    assert "cursor" in out
    assert "codex" in out
    assert "pi" in out


def test_install_missing_source_returns_nonzero(tmp_path, fake_home):
    """If a skill source dir is missing, install returns 1."""
    from datachain.cli.commands.skill import install_skills

    # The default skill is absent even though a legacy skill is available.
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

    assert result == 1
    assert not (fake_home / ".claude" / "skills" / "core").exists()


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


def test_transform_copilot_instructions_strips_frontmatter():
    """SKILL.md frontmatter is replaced with Copilot's applyTo block."""
    from datachain.cli.commands.skill import _transform_copilot_instructions

    p = Path(__file__).parent / "_test_copilot_fm.md"
    try:
        p.write_text(
            "---\n"
            "name: datachain-core\n"
            "description: SDK skill\n"
            "triggers: ['py']\n"
            "---\n"
            "# Body\n"
        )
        result = _transform_copilot_instructions(p)
        assert result.startswith("---\napplyTo: '**/*.py'\n---\n")
        assert "# Body" in result
        # All original frontmatter fields stripped
        assert "name: datachain-core" not in result
        assert "description: SDK skill" not in result
        assert "triggers:" not in result
    finally:
        p.unlink(missing_ok=True)


def test_transform_copilot_instructions_no_frontmatter():
    """SKILL.md without frontmatter still gets the Copilot applyTo block prepended."""
    from datachain.cli.commands.skill import _transform_copilot_instructions

    p = Path(__file__).parent / "_test_copilot_no_fm.md"
    try:
        p.write_text("# Direct heading\nbody text\n")
        result = _transform_copilot_instructions(p)
        assert result == "---\napplyTo: '**/*.py'\n---\n# Direct heading\nbody text\n"
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


def _seed_retired_install(fake_home: Path, target: str = "cursor") -> tuple[Path, Path]:
    """A jobs skill left over from a release that still shipped one."""
    from datachain.cli.commands.skill import TARGET_LAYOUT

    layout = TARGET_LAYOUT[target]
    commands_dir, command_ext = layout["commands_dir"], layout["command_ext"]
    assert commands_dir and command_ext, f"{target} writes a command file"

    skill_dir = fake_home / layout["skills_dir"] / "jobs"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: datachain-jobs\n---\n# jobs\n")

    command = fake_home / commands_dir / f"datachain-jobs{command_ext}"
    command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text("# datachain-jobs\n")
    return skill_dir, command


def test_install_clears_a_retired_skill(tmp_path, fake_skills_src, fake_home):
    """Upgrading over an install with `jobs` must not leave it instructing agents."""
    skill_dir, command = _seed_retired_install(fake_home)

    _run_install(fake_skills_src, fake_home, skills=None, target="cursor", local=False)

    assert not skill_dir.exists()
    assert not command.exists()
    assert (fake_home / ".cursor" / "skills" / "datachain" / "SKILL.md").exists()


def _seed_someone_elses_jobs_skill(fake_home: Path, target: str = "cursor") -> Path:
    """A `jobs` skill datachain did not install. `skills/` is a shared namespace."""
    from datachain.cli.commands.skill import TARGET_LAYOUT

    skill_dir = fake_home / TARGET_LAYOUT[target]["skills_dir"] / "jobs"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: my-own-jobs\n---\n# mine\n")
    (skill_dir / "scripts" / "run.py").write_text("print('mine')\n")
    return skill_dir


def test_install_leaves_a_jobs_skill_we_did_not_install(
    tmp_path, fake_skills_src, fake_home
):
    """The sweep takes the retired datachain skill, not whatever shares its directory
    name. Deleting someone else's `jobs` would take their work with it."""
    skill_dir = _seed_someone_elses_jobs_skill(fake_home)

    _run_install(fake_skills_src, fake_home, skills=None, target="cursor", local=False)

    assert "my-own-jobs" in (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    assert (skill_dir / "scripts" / "run.py").exists()


def test_install_leaves_a_jobs_skill_that_only_writes_about_ours(
    tmp_path, fake_skills_src, fake_home
):
    """Ownership is the frontmatter, not any `name:` in the document. A skill that
    documents datachain's own would otherwise declare itself to be datachain's."""
    from datachain.cli.commands.skill import TARGET_LAYOUT

    skill_dir = fake_home / TARGET_LAYOUT["cursor"]["skills_dir"] / "jobs"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-own-jobs\n---\n"
        "# notes\n\nThe retired skill declared itself:\n\n"
        "```yaml\nname: datachain-jobs\n```\n"
    )
    (skill_dir / "scripts" / "run.py").write_text("print('mine')\n")

    _run_install(fake_skills_src, fake_home, skills=None, target="cursor", local=False)

    assert "my-own-jobs" in (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    assert (skill_dir / "scripts" / "run.py").exists()


def test_uninstall_leaves_a_jobs_skill_we_did_not_install(
    tmp_path, fake_skills_src, fake_home
):
    skill_dir = _seed_someone_elses_jobs_skill(fake_home)
    from datachain.cli.commands.skill import uninstall_skills

    with patch("pathlib.Path.home", return_value=fake_home):
        assert uninstall_skills(skills="jobs", target="cursor", local=False) == 0

    assert (skill_dir / "scripts" / "run.py").exists()


def test_uninstall_all_clears_a_retired_skill(tmp_path, fake_skills_src, fake_home):
    skill_dir, command = _seed_retired_install(fake_home)
    from datachain.cli.commands.skill import uninstall_skills

    with patch("pathlib.Path.home", return_value=fake_home):
        assert uninstall_skills(skills=None, target="cursor", local=False) == 0

    assert not skill_dir.exists()
    assert not command.exists()


def test_uninstall_names_a_retired_skill(tmp_path, fake_skills_src, fake_home):
    """`uninstall jobs` has to keep working for anyone who installed it."""
    skill_dir, command = _seed_retired_install(fake_home)
    from datachain.cli.commands.skill import uninstall_skills

    with patch("pathlib.Path.home", return_value=fake_home):
        assert uninstall_skills(skills="jobs", target="cursor", local=False) == 0

    assert not skill_dir.exists()
    assert not command.exists()


def test_install_does_not_offer_a_retired_skill(tmp_path, fake_skills_src, fake_home):
    from datachain.cli.commands.skill import install_skills

    with (
        patch("datachain.cli.commands.skill._skills_src", return_value=fake_skills_src),
        patch("pathlib.Path.home", return_value=fake_home),
        pytest.raises(ValueError, match=r"Unknown skill.*jobs"),
    ):
        install_skills(skills="jobs", target="cursor", local=False)


def test_uninstall_error_names_the_retired_skill(tmp_path, fake_home):
    """Anyone who installed `jobs` must be able to see it is still removable."""
    from datachain.cli.commands.skill import uninstall_skills

    with (
        patch("pathlib.Path.home", return_value=fake_home),
        pytest.raises(ValueError, match=r"Valid skills:.*jobs"),
    ):
        uninstall_skills(skills="nope", target="cursor", local=False)


@pytest.mark.parametrize(
    "target,local,skills_path,companion_path",
    [
        ("claude", False, ".claude/skills", None),
        ("claude", True, ".claude/skills", ".claude/commands/datachain-datachain.md"),
        ("cursor", False, ".cursor/skills", ".cursor/rules/datachain-datachain.mdc"),
        ("cursor", True, ".cursor/skills", ".cursor/rules/datachain-datachain.mdc"),
        ("codex", False, ".codex/skills", None),
        ("codex", True, ".codex/skills", None),
        ("pi", False, ".pi/agent/skills", ".pi/agent/prompts/datachain-datachain.md"),
        ("pi", True, ".pi/skills", ".pi/prompts/datachain-datachain.md"),
        (
            "copilot",
            False,
            ".copilot/skills",
            ".copilot/instructions/datachain-datachain.instructions.md",
        ),
        (
            "copilot",
            True,
            ".datachain/skills",
            ".github/instructions/datachain-datachain.instructions.md",
        ),
    ],
)
def test_default_package_round_trip_and_reference_reachability(
    tmp_path, fake_home, monkeypatch, target, local, skills_path, companion_path
):
    from datachain.cli.commands.skill import (
        _skills_src,
        install_skills,
        uninstall_skills,
    )

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    base = project if local else fake_home
    skills_dir = base / skills_path
    with patch("pathlib.Path.home", return_value=fake_home):
        assert install_skills(skills=None, target=target, local=local) == 0
        assert {p.name for p in skills_dir.iterdir()} == {"datachain"}
        dest = skills_dir / "datachain"
        sdk = dest / "SDK.md"
        assert sdk.read_text(encoding="utf-8") == (
            _skills_src() / "core" / "SDK.md"
        ).read_text(encoding="utf-8")
        assert not (dest / "reference.md").exists()
        content = (dest / "SKILL.md").read_text(encoding="utf-8")
        assert "name: datachain\n" in content
        reference = sdk.relative_to(project) if local else sdk
        assert reference.as_posix() in content
        if local:
            assert project.as_posix() not in content
        if companion_path:
            companion = base / companion_path
            assert reference.as_posix() in companion.read_text(encoding="utf-8")
            if local:
                assert project.as_posix() not in companion.read_text(encoding="utf-8")
            assert "# DataChain SDK Rules" not in companion.read_text(encoding="utf-8")
        assert uninstall_skills(skills=None, target=target, local=local) == 0

    assert not dest.exists()
    if companion_path:
        assert not (base / companion_path).exists()


def test_default_upgrade_preserves_optional_skills_and_rules(
    tmp_path, fake_home, monkeypatch
):
    from datachain.cli.commands.skill import install_skills

    monkeypatch.chdir(tmp_path)
    with patch("pathlib.Path.home", return_value=fake_home):
        assert install_skills("knowledge", "cursor", True) == 0
        assert install_skills(None, "cursor", True) == 0

    skills = tmp_path / ".cursor" / "skills"
    rules = tmp_path / ".cursor" / "rules"
    assert {p.name for p in skills.iterdir()} == {"datachain", "core", "knowledge"}
    assert {p.name for p in rules.iterdir()} == {
        "datachain-datachain.mdc",
        "datachain-core.mdc",
        "datachain-knowledge.mdc",
    }


def test_failed_default_upgrade_keeps_legacy_skills(tmp_path, fake_home, monkeypatch):
    from datachain.cli.commands.skill import install_skills

    monkeypatch.chdir(tmp_path)
    with patch("pathlib.Path.home", return_value=fake_home):
        assert install_skills("knowledge", "cursor", True) == 0
        skills = tmp_path / ".cursor" / "skills"
        for name in ("core", "knowledge"):
            (skills / name / ".datachain-installed").unlink()
        before = {
            name: (skills / name / "SKILL.md").read_bytes()
            for name in ("core", "knowledge")
        }
        empty_source = tmp_path / "empty-source"
        empty_source.mkdir()
        monkeypatch.setattr(
            "datachain.cli.commands.skill._skills_src", lambda: empty_source
        )
        assert install_skills(None, "cursor", True) == 1
    assert {
        name: (skills / name / "SKILL.md").read_bytes() for name in before
    } == before


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252"])
def test_install_refuses_unrelated_datachain_skill(
    tmp_path, fake_home, monkeypatch, capsys, encoding
):
    from datachain.cli.commands.skill import install_skills

    monkeypatch.chdir(tmp_path)
    dest = tmp_path / ".cursor" / "skills" / "datachain"
    dest.mkdir(parents=True)
    content = "---\nname: datachain\ndescription: Café personnel\n---\n"
    (dest / "SKILL.md").write_text(content, encoding=encoding)
    (dest / "notes.txt").write_text("mine")
    with patch("pathlib.Path.home", return_value=fake_home):
        assert install_skills(None, "cursor", True) == 1
    assert (dest / "SKILL.md").read_text(encoding=encoding) == content
    assert (dest / "notes.txt").read_text(encoding="utf-8") == "mine"
    assert not (tmp_path / ".cursor" / "rules").exists()
    assert "Cannot overwrite unrelated skill" in capsys.readouterr().err
    from datachain.cli.commands.skill import uninstall_skills

    with patch("pathlib.Path.home", return_value=fake_home):
        assert uninstall_skills(None, "cursor", True) == 0
    assert (dest / "SKILL.md").read_text(encoding=encoding) == content
    assert (dest / "notes.txt").read_text(encoding="utf-8") == "mine"


@pytest.mark.parametrize("target", ["claude", "cursor", "codex", "pi", "copilot"])
def test_install_all_skills_by_name(fake_skills_src, fake_home, target):
    from datachain.cli.commands.skill import TARGET_LAYOUT

    _run_install(
        fake_skills_src,
        fake_home,
        skills=",".join(SKILLS),
        target=target,
        local=False,
    )
    skills_base = fake_home / TARGET_LAYOUT[target]["skills_dir"]
    assert {p.name for p in skills_base.iterdir()} == set(SKILLS)
    for skill in SKILLS:
        assert (skills_base / skill / "SKILL.md").exists()


def test_default_upgrade_removes_previous_release_skills(
    tmp_path, fake_home, monkeypatch
):
    from datachain.cli.commands.skill import install_skills

    monkeypatch.chdir(tmp_path)
    with patch("pathlib.Path.home", return_value=fake_home):
        assert install_skills("knowledge", "cursor", True) == 0
        skills = tmp_path / ".cursor" / "skills"
        for name in ("core", "knowledge"):
            (skills / name / ".datachain-installed").unlink()
        assert install_skills(None, "cursor", True) == 0

    assert {p.name for p in skills.iterdir()} == {"datachain"}
    assert {p.name for p in (tmp_path / ".cursor" / "rules").iterdir()} == {
        "datachain-datachain.mdc"
    }
