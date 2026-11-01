import re
import shutil
import sys
from importlib.resources import files
from pathlib import Path
from typing import TypedDict

SKILLS = ("datachain", "core", "knowledge")
DEFAULT_SKILLS = ("datachain",)
_INSTALL_MARKER = ".datachain-installed"
# Installable once, retired since. An upgrade must still be able to remove these:
# a stale skill directory keeps instructing the agent that reads it.
RETIRED_SKILLS = ("jobs",)
SKILL_DEPENDENCIES = {"knowledge": ("core",)}


class _TargetLayout(TypedDict):
    commands_dir: str | None
    skills_dir: str
    # Optional per-mode overrides. None = use the main `commands_dir`/`skills_dir`.
    # GitHub Copilot uses these to write to the standard `.github/instructions/`
    # path in --local mode while keeping the user-level vendor under `~/.copilot/`.
    commands_dir_local: str | None
    skills_dir_local: str | None
    command_ext: str | None
    commands_local_only: bool


# For each target: dirs relative to base (home or project root), and command extension.
# commands_dir=None means no command file is copied (skills only).
# commands_local_only=True means commands are only written for --local installs.
TARGET_LAYOUT: dict[str, _TargetLayout] = {
    "claude": {
        "commands_dir": ".claude/commands",
        "skills_dir": ".claude/skills",
        "commands_dir_local": None,
        "skills_dir_local": None,
        "command_ext": ".md",
        "commands_local_only": True,
    },
    "cursor": {
        "commands_dir": ".cursor/rules",
        "skills_dir": ".cursor/skills",
        "commands_dir_local": None,
        "skills_dir_local": None,
        "command_ext": ".mdc",
        "commands_local_only": False,
    },
    "codex": {
        "commands_dir": None,
        "skills_dir": ".codex/skills",
        "commands_dir_local": None,
        "skills_dir_local": None,
        "command_ext": None,
        "commands_local_only": False,
    },
    "pi": {
        # User-level: Pi scans ~/.pi/agent/{skills,prompts}/.
        # Repo-local: Pi scans .pi/{skills,prompts}/ (no `agent/` segment).
        "commands_dir": ".pi/agent/prompts",
        "skills_dir": ".pi/agent/skills",
        "commands_dir_local": ".pi/prompts",
        "skills_dir_local": ".pi/skills",
        "command_ext": ".md",
        "commands_local_only": False,
    },
    "copilot": {
        # User-level: write to ~/.copilot/ (datachain convention; VS Code can be
        # pointed here via the chat.instructionsFilesLocations setting).
        "commands_dir": ".copilot/instructions",
        "skills_dir": ".copilot/skills",
        # Repo-local: write to the canonical GitHub Copilot paths.
        "commands_dir_local": ".github/instructions",
        "skills_dir_local": ".datachain/skills",
        "command_ext": ".instructions.md",
        "commands_local_only": False,
    },
}

_COPYTREE_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".datachain")


def _skills_src() -> Path:
    """Return the path to the bundled skills source directory."""
    return Path(str(files("datachain.skill")))


def _transform_cursor_mdc(skill_md_path: Path) -> str:
    """Read SKILL.md and transform its frontmatter to Cursor .mdc format."""
    text = skill_md_path.read_text(encoding="utf-8")

    # Extract description from existing frontmatter
    description = ""
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if fm_match:
        for line in fm_match.group(1).splitlines():
            if line.startswith("description:"):
                description = line.split(":", 1)[1].strip()
                break
        body = text[fm_match.end() :]
    else:
        body = text

    return f"---\ndescription: {description}\nglobs:\nalwaysApply: true\n---\n{body}"


def _transform_copilot_instructions(skill_md_path: Path) -> str:
    """Transform SKILL.md into GitHub Copilot .instructions.md format.

    Strips any existing frontmatter (Claude/Cursor-specific keys like
    `triggers:`, `globs:`, `description:`) and replaces it with a Copilot
    `applyTo` glob. Copilot reads instruction files that match a glob against
    the active file path and prepends them to the prompt.
    """
    text = skill_md_path.read_text(encoding="utf-8")
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    body = text[fm_match.end() :] if fm_match else text
    return f"---\napplyTo: '**/*.py'\n---\n{body}"


def _frontmatter(text: str) -> str:
    """The leading `---` block, empty if the document does not open with one.

    Only the block counts: a `name:` further down is prose, and a skill that
    documents ours would otherwise declare itself to be ours.
    """
    match = re.match(r"---\r?\n(.*?)\r?\n---\s*?(\r?\n|\Z)", text, re.DOTALL)
    return match.group(1) if match else ""


def _frontmatter_name(name: str) -> str:
    return name if name == "datachain" else f"datachain-{name}"


def _installed_by_datachain(skill_dest: Path, name: str) -> bool:
    """Whether this directory holds the skill datachain installed under that name.

    `skills/<name>` is a shared namespace - anything may keep a `jobs` skill there -
    so ownership uses frontmatter and the generic default also needs our marker.
    Installing resolves placeholders and preserves the frontmatter name.
    """
    try:
        text = (skill_dest / "SKILL.md").read_text(encoding="utf-8")
    except OSError:
        return False
    frontmatter = _frontmatter(text)
    if not re.search(
        rf"^name:\s*{re.escape(_frontmatter_name(name))}\s*$", frontmatter, re.MULTILINE
    ):
        return False
    if name != "datachain":
        return True
    try:
        return (skill_dest / _INSTALL_MARKER).read_text(encoding="utf-8") == name
    except OSError:
        # Older default installs have no ownership marker.
        return (
            frontmatter.startswith(
                "name: datachain\ndescription: Use for any work on files "
                "in object storage or local folders ("
            )
            and "DataChain does the parallel I/O, keeps results as queryable datasets"
            in frontmatter
        )


def _remove_skill(
    skills_dir: Path, commands_dir: Path | None, command_ext: str | None, name: str
) -> bool:
    """Delete a skill's directory and its command file. True if either existed.

    The directory goes only if we installed it; the command file is named
    `datachain-<skill>`, so it can never be someone else's.
    """
    found = False
    skill_dest = skills_dir / name
    if skill_dest.exists() and _installed_by_datachain(skill_dest, name):
        shutil.rmtree(skill_dest)
        found = True
    if commands_dir and command_ext:
        cmd_dest = commands_dir / f"datachain-{name}{command_ext}"
        if cmd_dest.exists():
            cmd_dest.unlink()
            found = True
    return found


def _sweep_retired(
    skills_dir: Path,
    commands_dir: Path | None,
    command_ext: str | None,
    names: tuple[str, ...] = RETIRED_SKILLS,
) -> None:
    """Remove old installed skills and their companion instructions."""
    removed = [
        name
        for name in names
        if _remove_skill(skills_dir, commands_dir, command_ext, name)
    ]
    if removed:
        print(f"Removed old skills: {', '.join(removed)}")


def _write_companion(
    src: Path, dest: Path, skills_dir: Path, commands_dir: Path, command_ext: str
) -> None:
    skill_md = src / "SKILL.md"
    if not skill_md.exists():
        return
    commands_dir.mkdir(parents=True, exist_ok=True)
    if command_ext == ".mdc":
        content = _transform_cursor_mdc(skill_md)
    elif command_ext == ".instructions.md":
        content = _transform_copilot_instructions(skill_md)
    else:
        content = skill_md.read_text(encoding="utf-8")
    sdk_md = src / "SDK.md"
    if sdk_md.exists():
        content = f"{content.rstrip()}\n\n{sdk_md.read_text(encoding='utf-8')}"
    command = commands_dir / f"datachain-{src.name}{command_ext}"
    command.write_text(
        _resolve_placeholders(content, skills_dir, dest), encoding="utf-8"
    )


def _check_install_destinations(skills_dir: Path, names: list[str]) -> bool:
    for name in names:
        dest = skills_dir / name
        if dest.exists() and not _installed_by_datachain(dest, name):
            print(f"Cannot overwrite unrelated skill: {dest}", file=sys.stderr)
            return False
    return True


def install_skills(skills: str | None, target: str, local: bool) -> int:
    layout = TARGET_LAYOUT[target]
    base = Path.cwd() if local else Path.home()

    if skills:
        requested = [s.strip() for s in skills.split(",")]
        invalid = [s for s in requested if s not in SKILLS]
        if invalid:
            valid = ", ".join(SKILLS)
            raise ValueError(
                f"Unknown skill(s): {', '.join(invalid)}. Valid skills: {valid}"
            )
        skills_to_install = requested + [
            dep
            for skill in requested
            for dep in SKILL_DEPENDENCIES.get(skill, ())
            if dep not in requested
        ]
    else:
        skills_to_install = list(DEFAULT_SKILLS)

    # Per-mode overrides — when installing --local some targets (e.g. copilot)
    # write to a different directory layout than the user-level default.
    skills_dir_rel = (
        layout["skills_dir_local"]
        if (local and layout["skills_dir_local"] is not None)
        else layout["skills_dir"]
    )
    commands_dir_rel = (
        layout["commands_dir_local"]
        if (local and layout["commands_dir_local"] is not None)
        else layout["commands_dir"]
    )
    skills_dir = base / skills_dir_rel

    # Determine whether to write command/rule files
    write_commands = (
        commands_dir_rel is not None
        and layout["command_ext"] is not None
        and (local or not layout["commands_local_only"])
    )
    commands_dir = (
        base / commands_dir_rel if write_commands and commands_dir_rel else None
    )
    command_ext = layout["command_ext"]

    if not _check_install_destinations(skills_dir, skills_to_install):
        return 1

    installed = []
    missing = []
    for skill_name in skills_to_install:
        src = _skills_src() / skill_name
        if not src.exists():
            print(f"Warning: skill source not found: {src}", file=sys.stderr)
            missing.append(skill_name)
            continue

        dest = skills_dir / skill_name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest, dirs_exist_ok=True, ignore=_COPYTREE_IGNORE)
        (dest / _INSTALL_MARKER).write_text(skill_name, encoding="utf-8")
        if skill_name == "datachain":
            shutil.copyfile(_skills_src() / "core" / "SDK.md", dest / "SDK.md")

        # Resolve {skill_dir} and {<name>_skill_dir} placeholders in installed
        # SKILL.md so the agent doesn't have to probe the filesystem to find its
        # own scripts or another skill's files.
        installed_skill_md = dest / "SKILL.md"
        if installed_skill_md.exists():
            installed_skill_md.write_text(
                _resolve_placeholders(
                    installed_skill_md.read_text(encoding="utf-8"), skills_dir, dest
                ),
                encoding="utf-8",
            )

        if commands_dir is not None and command_ext is not None:
            _write_companion(src, dest, skills_dir, commands_dir, command_ext)

        installed.append(f"  {skill_name} → {dest}")

    old_skills: tuple[str, ...] = RETIRED_SKILLS
    if skills_to_install == list(DEFAULT_SKILLS) and not missing:
        old_skills += tuple(name for name in SKILLS if name not in DEFAULT_SKILLS)
    _sweep_retired(skills_dir, commands_dir, command_ext, old_skills)

    if installed:
        scope = "local" if local else "global"
        print(f"Installed skills ({scope}, target={target}):")
        for line in installed:
            print(line)
    else:
        print("No skills installed.")

    if missing:
        return 1
    return 0


def uninstall_skills(skills: str | None, target: str, local: bool) -> int:
    layout = TARGET_LAYOUT[target]
    base = Path.cwd() if local else Path.home()

    if skills:
        requested = [s.strip() for s in skills.split(",")]
        invalid = [s for s in requested if s not in SKILLS + RETIRED_SKILLS]
        if invalid:
            valid = ", ".join(SKILLS + RETIRED_SKILLS)
            raise ValueError(
                f"Unknown skill(s): {', '.join(invalid)}. Valid skills: {valid}"
            )
        skills_to_uninstall = requested
    else:
        skills_to_uninstall = list(SKILLS) + list(RETIRED_SKILLS)

    skills_dir_rel = (
        layout["skills_dir_local"]
        if (local and layout["skills_dir_local"] is not None)
        else layout["skills_dir"]
    )
    commands_dir_rel = (
        layout["commands_dir_local"]
        if (local and layout["commands_dir_local"] is not None)
        else layout["commands_dir"]
    )
    skills_dir = base / skills_dir_rel

    for skill_name in skills_to_uninstall:
        dependents = [
            dependent
            for dependent, deps in SKILL_DEPENDENCIES.items()
            if skill_name in deps
            and dependent not in skills_to_uninstall
            and (skills_dir / dependent).exists()
        ]
        if dependents:
            raise ValueError(
                f"Cannot uninstall {skill_name}: {', '.join(dependents)} depends "
                "on it. Uninstall them together or run without --skills."
            )

    write_commands = (
        commands_dir_rel is not None
        and layout["command_ext"] is not None
        and (local or not layout["commands_local_only"])
    )
    commands_dir = (
        base / commands_dir_rel if write_commands and commands_dir_rel else None
    )
    command_ext = layout["command_ext"]

    removed = []
    not_found = []
    for skill_name in skills_to_uninstall:
        if _remove_skill(skills_dir, commands_dir, command_ext, skill_name):
            removed.append(f"  {skill_name}")
        elif skill_name not in RETIRED_SKILLS or skills:
            # A retired skill nobody installed is not news unless it was asked for.
            not_found.append(skill_name)

    if removed:
        scope = "local" if local else "global"
        print(f"Uninstalled skills ({scope}, target={target}):")
        for line in removed:
            print(line)
    if not_found:
        print(f"Not found (already uninstalled): {', '.join(not_found)}")

    return 0


def list_skills() -> int:
    targets = ", ".join(TARGET_LAYOUT.keys())
    header = f"{'Skill':<12}  Targets"
    print(header)
    print("-" * len(header))
    for name in SKILLS:
        print(f"{name:<12}  {targets}")
    return 0


def _resolve_placeholders(text: str, skills_dir: Path, dest: Path) -> str:
    text = text.replace("{skill_dir}", dest.resolve().as_posix())
    for name in SKILLS:
        text = text.replace(
            f"{{{name}_skill_dir}}", (skills_dir / name).resolve().as_posix()
        )
    return text
