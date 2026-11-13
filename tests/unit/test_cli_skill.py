"""Tests for the `datachain skill` CLI command."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from datachain.cli.commands.skill import SKILLS
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
    args = parser.parse_args(["skill", "install", "datachain"])
    assert args.skills == "datachain"


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
    skills_src = tmp_path / "skills_src"
    skill_source = skills_src / "datachain"
    skill_source.mkdir(parents=True)
    (skill_source / "SKILL.md").write_text(
        "---\nname: datachain\ndescription: Test skill datachain\n"
        "---\n# datachain\nRead `{skill_dir}/SDK.md`.\n"
    )
    (skills_src / "core").mkdir()
    (skills_src / "core" / "SDK.md").write_text("# DataChain SDK Rules\n")
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


def test_install_explicit_datachain(tmp_path, fake_skills_src, fake_home):
    _run_install(
        fake_skills_src, fake_home, skills="datachain", target="claude", local=False
    )
    skills = fake_home / ".claude" / "skills"
    assert {p.name for p in skills.iterdir()} == {"datachain"}
    assert (skills / "datachain" / "SDK.md").read_text() == "# DataChain SDK Rules\n"


def test_install_writes_only_skill_and_reference(tmp_path, fake_skills_src, fake_home):
    _run_install(fake_skills_src, fake_home, skills=None, target="claude", local=False)

    installed = fake_home / ".claude" / "skills" / "datachain"
    assert {p.name for p in installed.iterdir()} == {
        "SKILL.md",
        "SDK.md",
        ".datachain-installed",
    }


def test_list_skills_output(capsys):
    from datachain.cli.commands.skill import list_skills

    list_skills()
    out = capsys.readouterr().out
    assert "datachain" in out
    assert "core" not in out
    assert "knowledge" not in out
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
    from datachain.cli.commands.skill import _skills_src

    src = _skills_src()
    assert src.is_dir(), f"_skills_src() returned non-directory: {src}"
    for skill_name in SKILLS:
        skill_md = src / skill_name / "SKILL.md"
        assert skill_md.exists(), f"Missing bundled {skill_name}/SKILL.md"


def _seed_someone_elses_jobs_skill(fake_home: Path, target: str = "cursor") -> Path:
    """A `jobs` skill datachain did not install. `skills/` is a shared namespace."""
    from datachain.cli.commands.skill import TARGET_LAYOUT

    skill_dir = fake_home / TARGET_LAYOUT[target][False][0] / "jobs"
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

    skill_dir = fake_home / TARGET_LAYOUT["cursor"][False][0] / "jobs"
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
    skills = _seed_legacy_skills(fake_home, "cursor", False)
    from datachain.cli.commands.skill import uninstall_skills

    with patch("pathlib.Path.home", return_value=fake_home):
        assert uninstall_skills(skills=None, target="cursor", local=False) == 0

    assert not any(skills.iterdir())
    assert not any((fake_home / ".cursor" / "rules").iterdir())


@pytest.mark.parametrize("name", ["jobs", "core", "knowledge"])
def test_uninstall_names_a_retired_skill(tmp_path, fake_skills_src, fake_home, name):
    skills = _seed_legacy_skills(fake_home, "cursor", False)
    skill_dir = skills / name
    command = fake_home / ".cursor" / "rules" / f"datachain-{name}.mdc"
    from datachain.cli.commands.skill import uninstall_skills

    with patch("pathlib.Path.home", return_value=fake_home):
        assert uninstall_skills(skills=name, target="cursor", local=False) == 0

    assert not skill_dir.exists()
    assert not command.exists()


@pytest.mark.parametrize("name", ["jobs", "core", "knowledge"])
def test_install_does_not_offer_a_retired_skill(
    tmp_path, fake_skills_src, fake_home, name
):
    from datachain.cli.commands.skill import install_skills

    with (
        patch("datachain.cli.commands.skill._skills_src", return_value=fake_skills_src),
        patch("pathlib.Path.home", return_value=fake_home),
        pytest.raises(
            ValueError, match=rf"Unknown skill.*{name}.*Valid skills: datachain$"
        ),
    ):
        install_skills(skills=name, target="cursor", local=False)


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
    dest = skills_dir / "datachain"
    expected = {
        (dest / name).relative_to(tmp_path).as_posix()
        for name in ("SKILL.md", "SDK.md", ".datachain-installed")
    }
    if companion_path:
        expected.add((base / companion_path).relative_to(tmp_path).as_posix())

    def installed():
        return {
            p.relative_to(tmp_path).as_posix()
            for p in tmp_path.rglob("*")
            if p.is_file()
        }

    with patch("pathlib.Path.home", return_value=fake_home):
        assert install_skills(skills=None, target=target, local=local) == 0
        assert installed() == expected
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
            companion_text = companion.read_text(encoding="utf-8")
            assert reference.as_posix() in companion_text
            header = yaml.safe_load(companion_text.split("---", 2)[1])
            skill_header = yaml.safe_load(content.split("---", 2)[1])
            if target == "cursor":
                assert header == {
                    "description": skill_header["description"],
                    "globs": None,
                    "alwaysApply": True,
                }
            elif target == "copilot":
                assert header == {"applyTo": "**/*.py"}
            else:
                assert header == skill_header
            if local:
                assert project.as_posix() not in companion.read_text(encoding="utf-8")
            assert "# DataChain SDK Rules" not in companion.read_text(encoding="utf-8")
        assert uninstall_skills(skills=None, target=target, local=local) == 0

    assert installed() == set()
    assert not dest.exists()
    if companion_path:
        assert not (base / companion_path).exists()


def _seed_legacy_skills(base, target, local, marked=False):
    from datachain.cli.commands.skill import RETIRED_SKILLS, TARGET_LAYOUT

    skills_path, commands_path, suffix = TARGET_LAYOUT[target][local]
    skills = base / skills_path
    for name in RETIRED_SKILLS:
        dest = skills / name
        dest.mkdir(parents=True)
        (dest / "SKILL.md").write_text(f"---\nname: datachain-{name}\n---\n")
        if marked:
            (dest / ".datachain-installed").write_text(name)
        if commands_path:
            command = base / commands_path / f"datachain-{name}{suffix}"
            command.parent.mkdir(parents=True, exist_ok=True)
            command.write_text(f"# datachain-{name}\n")
    return skills


@pytest.mark.parametrize("target", ["claude", "cursor", "codex", "pi", "copilot"])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("marked", [False, True])
def test_upgrade_removes_legacy_skills_and_preserves_data(
    tmp_path, fake_home, monkeypatch, target, local, marked
):
    from datachain.cli.commands.skill import TARGET_LAYOUT, install_skills

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    base = project if local else fake_home
    skills = _seed_legacy_skills(base, target, local, marked)
    untouched = [project / ".datachain" / "db", project / "dc-knowledge" / "index.md"]
    other = project / ".other-agent" / "skills" / "knowledge" / "SKILL.md"
    untouched.append(other)
    for path in untouched:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"user data")
    with patch("pathlib.Path.home", return_value=fake_home):
        assert install_skills(None, target, local) == 0
    assert {p.name for p in skills.iterdir()} == {"datachain"}
    _, commands_path, suffix = TARGET_LAYOUT[target][local]
    if commands_path:
        assert {p.name for p in (base / commands_path).iterdir()} == {
            f"datachain-datachain{suffix}"
        }
    assert all(path.read_bytes() == b"user data" for path in untouched)


@pytest.mark.parametrize("failure", ["missing_source", "missing_skill", "write_error"])
def test_failed_default_upgrade_keeps_legacy_skills(
    tmp_path, fake_home, monkeypatch, fake_skills_src, failure
):
    from datachain.cli.commands.skill import install_skills

    monkeypatch.chdir(tmp_path)
    with patch("pathlib.Path.home", return_value=fake_home):
        _seed_legacy_skills(tmp_path, "cursor", True, marked=True)
        before = {
            path: path.read_bytes()
            for path in (tmp_path / ".cursor").rglob("*")
            if path.is_file()
        }
        monkeypatch.setattr(
            "datachain.cli.commands.skill._skills_src", lambda: fake_skills_src
        )
        if failure == "write_error":
            with (
                patch(
                    "datachain.cli.commands.skill._write_text_atomic",
                    side_effect=OSError("write failed"),
                ),
                pytest.raises(OSError, match="write failed"),
            ):
                install_skills(None, "cursor", True)
        else:
            (fake_skills_src / "datachain" / "SKILL.md").unlink()
            if failure == "missing_source":
                import shutil

                shutil.rmtree(fake_skills_src / "datachain")
            assert install_skills(None, "cursor", True) == 1
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("name", ["core", "knowledge"])
def test_upgrade_preserves_unrelated_legacy_named_skill(
    tmp_path, fake_home, monkeypatch, name
):
    from datachain.cli.commands.skill import install_skills, uninstall_skills

    monkeypatch.chdir(tmp_path)
    dest = tmp_path / ".cursor" / "skills" / name
    dest.mkdir(parents=True)
    content = f"---\nname: my-{name}\n---\nUser instructions\n"
    (dest / "SKILL.md").write_text(content)
    companion = tmp_path / ".cursor" / "rules" / f"datachain-{name}.mdc"
    companion.parent.mkdir(parents=True)
    companion.write_text("User-written companion instructions")
    with patch("pathlib.Path.home", return_value=fake_home):
        assert install_skills(None, "cursor", True) == 0
        assert (dest / "SKILL.md").read_text() == content
        assert uninstall_skills(None, "cursor", True) == 0
    assert (dest / "SKILL.md").read_text() == content
    assert companion.read_text() == "User-written companion instructions"


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


@pytest.mark.parametrize(
    "encoding,profile",
    [("utf-8", "José"), ("cp1252", "José"), ("cp1251", "Дмитрий")],
)
def test_default_upgrade_removes_previous_release_skills(
    tmp_path, fake_home, monkeypatch, encoding, profile
):
    from datachain.cli.commands.skill import install_skills

    monkeypatch.chdir(tmp_path)
    with patch("pathlib.Path.home", return_value=fake_home):
        skills = _seed_legacy_skills(tmp_path, "cursor", True)
        for name in ("core", "knowledge"):
            text = f"---\nname: datachain-{name}\n---\nProfile: C:\\Users\\{profile}\n"
            (skills / name / "SKILL.md").write_bytes(text.encode(encoding))
        assert install_skills(None, "cursor", True) == 0

    assert {p.name for p in skills.iterdir()} == {"datachain"}
    assert {p.name for p in (tmp_path / ".cursor" / "rules").iterdir()} == {
        "datachain-datachain.mdc"
    }


def test_uninstall_reports_only_requested_missing_skills(
    tmp_path, fake_home, monkeypatch, capsys
):
    from datachain.cli.commands.skill import install_skills, uninstall_skills

    monkeypatch.chdir(tmp_path)
    with patch("pathlib.Path.home", return_value=fake_home):
        assert uninstall_skills(None, "cursor", True) == 0
        assert capsys.readouterr().out == "No DataChain skills found.\n"
        assert install_skills(None, "cursor", True) == 0
        capsys.readouterr()
        assert uninstall_skills(None, "cursor", True) == 0
        assert "Not found" not in capsys.readouterr().out
        assert uninstall_skills("core", "cursor", True) == 0
        assert capsys.readouterr().out == "Not found (already uninstalled): core\n"


@pytest.mark.parametrize("write_first", [False, True])
def test_interrupted_install_can_be_retried(
    tmp_path, fake_home, monkeypatch, write_first
):
    from datachain.cli.commands import skill

    monkeypatch.chdir(tmp_path)
    original = skill._write_text_atomic

    def interrupted(path, content):
        if path.name == "SDK.md":
            if write_first:
                original(path, content)
            raise OSError("disk full")
        return original(path, content)

    with patch("pathlib.Path.home", return_value=fake_home):
        with patch.object(skill, "_write_text_atomic", interrupted):
            with pytest.raises(OSError, match="disk full"):
                skill.install_skills(None, "claude", True)
        assert not (tmp_path / ".claude/skills/datachain").exists()
        assert skill.install_skills(None, "claude", True) == 0
        assert skill.uninstall_skills(None, "claude", True) == 0


@pytest.mark.parametrize("upgrade", [False, True])
def test_interrupted_skill_rewrite_preserves_retryable_ownership(
    tmp_path, fake_home, monkeypatch, upgrade
):
    from datachain.cli.commands import skill

    monkeypatch.chdir(tmp_path)
    original = Path.write_text

    def interrupted(path, *args, **kwargs):
        if path.name == "SKILL.md":
            original(path, "", encoding="utf-8")
            raise OSError("disk full")
        return original(path, *args, **kwargs)

    with patch("pathlib.Path.home", return_value=fake_home):
        if upgrade:
            assert skill.install_skills(None, "claude", True) == 0
        with patch.object(Path, "write_text", interrupted):
            with pytest.raises(OSError, match="disk full"):
                skill.install_skills(None, "claude", True)
        dest = tmp_path / ".claude/skills/datachain"
        if upgrade:
            assert skill._installed_by_datachain(dest, "datachain")
        else:
            assert not dest.exists()
        assert skill.install_skills(None, "claude", True) == 0
        assert skill.uninstall_skills(None, "claude", True) == 0


@pytest.mark.parametrize("action", ["install", "uninstall"])
def test_marker_only_install_can_be_retried_or_removed(
    tmp_path, fake_home, fake_skills_src, monkeypatch, action
):
    from datachain.cli.commands import skill

    monkeypatch.chdir(tmp_path)
    dest = tmp_path / ".claude/skills/datachain"
    dest.mkdir(parents=True)
    (dest / ".datachain-installed").write_text("datachain")

    with (
        patch("pathlib.Path.home", return_value=fake_home),
        patch.object(skill, "_skills_src", return_value=fake_skills_src),
    ):
        if action == "install":
            assert skill.install_skills(None, "claude", True) == 0
            assert (dest / "SKILL.md").is_file()
            assert (dest / "SDK.md").is_file()
        else:
            assert skill.uninstall_skills(None, "claude", True) == 0
            assert not dest.exists()


def test_failed_sdk_upgrade_preserves_installed_reference(
    tmp_path, fake_home, fake_skills_src, monkeypatch
):
    from datachain.cli.commands import skill

    monkeypatch.chdir(tmp_path)
    original = Path.write_text
    dest = tmp_path / ".claude/skills/datachain"

    def interrupted(path, *args, **kwargs):
        if path.name == "SDK.md":
            original(path, "partial", encoding="utf-8")
            raise OSError("disk full")
        return original(path, *args, **kwargs)

    with (
        patch("pathlib.Path.home", return_value=fake_home),
        patch.object(skill, "_skills_src", return_value=fake_skills_src),
    ):
        assert skill.install_skills(None, "claude", True) == 0
        sdk = (dest / "SDK.md").read_text()
        (fake_skills_src / "core/SDK.md").write_text("new reference")
        with patch.object(Path, "write_text", interrupted):
            with pytest.raises(OSError, match="disk full"):
                skill.install_skills(None, "claude", True)
        assert (dest / "SDK.md").read_text() == sdk
        assert skill.install_skills(None, "claude", True) == 0
        assert (dest / "SDK.md").read_text() == "new reference"
