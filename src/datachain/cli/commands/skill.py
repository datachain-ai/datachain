import re
import shutil
import sys
from importlib.resources import files
from pathlib import Path
from typing import TypedDict

SKILLS = ("core", "knowledge", "jobs")


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


def _extract_description(frontmatter: str) -> str:
    for line in frontmatter.splitlines():
        if line.startswith("description:"):
            return line.split(":", 1)[1].strip()
    return ""


def _parse_frontmatter(text: str) -> tuple[str, str]:
    description = ""
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if fm_match:
        description = _extract_description(fm_match.group(1))
        body = text[fm_match.end() :]
    else:
        body = text
    return description, body


def _transform_cursor_mdc(skill_md_path: Path) -> str:
    text = skill_md_path.read_text()
    description, body = _parse_frontmatter(text)
    return f"---\ndescription: {description}\nglobs:\nalwaysApply: true\n---\n{body}"


def _transform_copilot_instructions(skill_md_path: Path) -> str:
    """Transform SKILL.md into GitHub Copilot .instructions.md format.

    Strips any existing frontmatter (Claude/Cursor-specific keys like
    `triggers:`, `globs:`, `description:`) and replaces it with a Copilot
    `applyTo` glob. Copilot reads instruction files that match a glob against
    the active file path and prepends them to the prompt.
    """
    text = skill_md_path.read_text()
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    body = text[fm_match.end() :] if fm_match else text
    return f"---\napplyTo: '**/*.py'\n---\n{body}"


class _ResolvedLayout(TypedDict):
    skills_dir: Path
    commands_dir: Path | None
    command_ext: str | None
    write_commands: bool


def _parse_skills(skills: str | None) -> list[str]:
    if not skills:
        return list(SKILLS)
    requested = [s.strip() for s in skills.split(",")]
    invalid = [s for s in requested if s not in SKILLS]
    if invalid:
        raise ValueError(
            f"Unknown skill(s): {', '.join(invalid)}. Valid skills: {', '.join(SKILLS)}"
        )
    return requested


def _resolve_layout(layout: _TargetLayout, local: bool, base: Path) -> _ResolvedLayout:
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
    write_commands = (
        commands_dir_rel is not None
        and layout["command_ext"] is not None
        and (local or not layout["commands_local_only"])
    )
    commands_dir = (
        base / commands_dir_rel if write_commands and commands_dir_rel else None
    )
    return {
        "skills_dir": base / skills_dir_rel,
        "commands_dir": commands_dir,
        "command_ext": layout["command_ext"],
        "write_commands": write_commands,
    }


def _get_skill_content(skill_md: Path, command_ext: str) -> str:
    if command_ext == ".mdc":
        return _transform_cursor_mdc(skill_md)
    if command_ext == ".instructions.md":
        return _transform_copilot_instructions(skill_md)
    return skill_md.read_text()


def _install_command_file(
    src: Path,
    skill_dest: Path,
    commands_dir: Path,
    command_ext: str,
    skill_name: str,
) -> None:
    commands_dir.mkdir(parents=True, exist_ok=True)
    skill_md = src / "SKILL.md"
    if not skill_md.exists():
        return
    content = _get_skill_content(skill_md, command_ext)
    cmd_dest = commands_dir / f"datachain-{skill_name}{command_ext}"
    cmd_dest.write_text(content.replace("{skill_dir}", str(skill_dest.resolve())))


def _remove_command_file(
    commands_dir: Path | None,
    command_ext: str | None,
    skill_name: str,
) -> bool:
    if not (commands_dir and command_ext):
        return False
    cmd_dest = commands_dir / f"datachain-{skill_name}{command_ext}"
    if not cmd_dest.exists():
        return False
    cmd_dest.unlink()
    return True


def _remove_skill(
    skill_name: str,
    skills_dir: Path,
    commands_dir: Path | None,
    command_ext: str | None,
) -> bool:
    skill_dest = skills_dir / skill_name
    found = False
    if skill_dest.exists():
        shutil.rmtree(skill_dest)
        found = True
    found = _remove_command_file(commands_dir, command_ext, skill_name) or found
    return found


def _check_skill_source(skill_name: str, missing: list[str]) -> Path | None:
    src = _skills_src() / skill_name
    if not src.exists():
        print(f"Warning: skill source not found: {src}", file=sys.stderr)
        missing.append(skill_name)
        return None
    return src


def _install_skill_files(
    src: Path,
    dest: Path,
    commands_dir: Path | None,
    command_ext: str | None,
    skill_name: str,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True, ignore=_COPYTREE_IGNORE)
    skill_md_dest = dest / "SKILL.md"
    if skill_md_dest.exists():
        resolved = skill_md_dest.read_text().replace("{skill_dir}", str(dest.resolve()))
        skill_md_dest.write_text(resolved)
    commands_dir and command_ext and _install_command_file(
        src, dest, commands_dir, command_ext, skill_name
    )


def _install_or_record(
    skill_name: str,
    resolved: _ResolvedLayout,
    installed: list[str],
    missing: list[str],
) -> None:
    src = _check_skill_source(skill_name, missing)
    if not src:
        return
    dest = resolved["skills_dir"] / skill_name
    _install_skill_files(
        src, dest, resolved["commands_dir"], resolved["command_ext"], skill_name
    )
    installed.append(f"  {skill_name} → {dest}")


def _install_all(
    skills_to_install: list[str], resolved: _ResolvedLayout
) -> tuple[list[str], list[str]]:
    installed: list[str] = []
    missing: list[str] = []
    for skill_name in skills_to_install:
        _install_or_record(skill_name, resolved, installed, missing)
    return installed, missing


def _print_install_report(
    installed: list[str],
    missing: list[str],
    local: bool,
    target: str,
) -> None:
    if not installed:
        print("No skills installed.")
        return
    scope = "local" if local else "global"
    print(f"Installed skills ({scope}, target={target}):")
    for line in installed:
        print(line)


def install_skills(skills: str | None, target: str, local: bool) -> int:
    layout = TARGET_LAYOUT[target]
    base = Path.cwd() if local else Path.home()
    resolved = _resolve_layout(layout, local, base)
    skills_to_install = _parse_skills(skills)
    installed, missing = _install_all(skills_to_install, resolved)
    _print_install_report(installed, missing, local, target)
    return 1 if missing else 0


def _uninstall_or_record(
    skill_name: str,
    resolved: _ResolvedLayout,
    removed: list[str],
    not_found: list[str],
) -> None:
    skills_dir = resolved["skills_dir"]
    commands_dir = resolved["commands_dir"]
    command_ext = resolved["command_ext"]
    if _remove_skill(skill_name, skills_dir, commands_dir, command_ext):
        removed.append(f"  {skill_name}")
    else:
        not_found.append(skill_name)


def _uninstall_all(
    skills_to_uninstall: list[str], resolved: _ResolvedLayout
) -> tuple[list[str], list[str]]:
    removed: list[str] = []
    not_found: list[str] = []
    for skill_name in skills_to_uninstall:
        _uninstall_or_record(skill_name, resolved, removed, not_found)
    return removed, not_found


def _print_uninstall_report(
    removed: list[str],
    not_found: list[str],
    local: bool,
    target: str,
) -> None:
    if removed:
        scope = "local" if local else "global"
        print(f"Uninstalled skills ({scope}, target={target}):")
        for line in removed:
            print(line)
    if not_found:
        print(f"Not found (already uninstalled): {', '.join(not_found)}")


def uninstall_skills(skills: str | None, target: str, local: bool) -> int:
    layout = TARGET_LAYOUT[target]
    base = Path.cwd() if local else Path.home()
    skills_to_uninstall = _parse_skills(skills)
    resolved = _resolve_layout(layout, local, base)
    removed, not_found = _uninstall_all(skills_to_uninstall, resolved)
    _print_uninstall_report(removed, not_found, local, target)
    return 0


def list_skills() -> int:
    targets = ", ".join(TARGET_LAYOUT.keys())
    header = f"{'Skill':<12}  Targets"
    print(header)
    print("-" * len(header))
    for name in SKILLS:
        print(f"{name:<12}  {targets}")
    return 0
