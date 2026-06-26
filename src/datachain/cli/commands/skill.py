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


def _transform_cursor_mdc(skill_md_path: Path) -> str:
    """Read SKILL.md and transform its frontmatter to Cursor .mdc format."""
    text = skill_md_path.read_text()

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
    text = skill_md_path.read_text()
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    body = text[fm_match.end() :] if fm_match else text
    return f"---\napplyTo: '**/*.py'\n---\n{body}"


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
        skills_to_install = requested
    else:
        skills_to_install = list(SKILLS)

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

        # Resolve {skill_dir} placeholder in installed SKILL.md so the agent
        # doesn't have to probe the filesystem to find its own scripts.
        installed_skill_md = dest / "SKILL.md"
        if installed_skill_md.exists():
            resolved = installed_skill_md.read_text().replace(
                "{skill_dir}", str(dest.resolve())
            )
            installed_skill_md.write_text(resolved)

        if commands_dir is not None and command_ext is not None:
            commands_dir.mkdir(parents=True, exist_ok=True)
            skill_md = src / "SKILL.md"
            if skill_md.exists():
                cmd_dest = commands_dir / f"datachain-{skill_name}{command_ext}"
                skill_dir_resolved = str(dest.resolve())
                if command_ext == ".mdc":
                    content = _transform_cursor_mdc(skill_md)
                elif command_ext == ".instructions.md":
                    content = _transform_copilot_instructions(skill_md)
                else:
                    content = skill_md.read_text()
                cmd_dest.write_text(content.replace("{skill_dir}", skill_dir_resolved))

        installed.append(f"  {skill_name} → {dest}")

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
        invalid = [s for s in requested if s not in SKILLS]
        if invalid:
            valid = ", ".join(SKILLS)
            raise ValueError(
                f"Unknown skill(s): {', '.join(invalid)}. Valid skills: {valid}"
            )
        skills_to_uninstall = requested
    else:
        skills_to_uninstall = list(SKILLS)

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
        skill_dest = skills_dir / skill_name
        cmd_dest = (
            commands_dir / f"datachain-{skill_name}{command_ext}"
            if commands_dir and command_ext
            else None
        )

        found = False
        if skill_dest.exists():
            shutil.rmtree(skill_dest)
            found = True
        if cmd_dest and cmd_dest.exists():
            cmd_dest.unlink()
            found = True

        if found:
            removed.append(f"  {skill_name}")
        else:
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
