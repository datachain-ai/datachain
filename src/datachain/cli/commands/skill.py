import re
import shutil
import sys
from importlib.resources import files
from pathlib import Path
from typing import TypedDict

SKILLS = ("core", "graph", "jobs")


class _TargetLayout(TypedDict):
    commands_dir: str | None
    skills_dir: str
    command_ext: str | None
    commands_local_only: bool


# For each target: dirs relative to base (home or project root), and command extension.
# commands_dir=None means no command file is copied (skills only).
# commands_local_only=True means commands are only written for --local installs.
TARGET_LAYOUT: dict[str, _TargetLayout] = {
    "claude": {
        "commands_dir": ".claude/commands",
        "skills_dir": ".claude/skills",
        "command_ext": ".md",
        "commands_local_only": True,
    },
    "cursor": {
        "commands_dir": ".cursor/rules",
        "skills_dir": ".cursor/skills",
        "command_ext": ".mdc",
        "commands_local_only": False,
    },
    "codex": {
        "commands_dir": None,
        "skills_dir": ".codex/skills",
        "command_ext": None,
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

    skills_dir = base / layout["skills_dir"]

    # Determine whether to write command/rule files
    write_commands = (
        layout["commands_dir"] is not None
        and layout["command_ext"] is not None
        and (local or not layout["commands_local_only"])
    )
    commands_dir = (
        base / layout["commands_dir"]
        if write_commands and layout["commands_dir"]
        else None
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


def list_skills() -> int:
    targets = ", ".join(TARGET_LAYOUT.keys())
    header = f"{'Skill':<12}  Targets"
    print(header)
    print("-" * len(header))
    for name in SKILLS:
        print(f"{name:<12}  {targets}")
    return 0
