import shutil
from pathlib import Path

SKILLS = {
    "core": "dc-core",
    "graph": "dc-graph",
}

# For each target: dirs relative to base (home or project root), and command extension.
# command_ext=None means no command file is copied (skills only).
TARGET_LAYOUT = {
    "claude": {
        "commands_dir": ".claude/commands",
        "skills_dir": ".claude/skills",
        "command_ext": ".md",
    },
    "cursor": {
        "commands_dir": ".cursor/rules",
        "skills_dir": ".cursor/skills",
        "command_ext": ".mdc",
    },
    "codex": {
        "commands_dir": None,
        "skills_dir": ".codex/skills",
        "command_ext": None,
    },
}

# Path to the bundled skills directory inside this package installation.
_SKILLS_SRC = Path(__file__).parent.parent.parent.parent.parent / "skills"


def _skills_src() -> Path:
    """Return the path to the bundled skills source directory."""
    return _SKILLS_SRC


def install_skills(skills: str | None, target: str, local: bool) -> None:
    layout = TARGET_LAYOUT[target]
    base = Path(".") if local else Path.home()

    if skills:
        requested = [s.strip() for s in skills.split(",")]
        invalid = [s for s in requested if s not in SKILLS.values()]
        if invalid:
            valid = ", ".join(SKILLS.values())
            raise ValueError(
                f"Unknown skill(s): {', '.join(invalid)}. Valid skills: {valid}"
            )
        skills_to_install = {k: v for k, v in SKILLS.items() if v in requested}
    else:
        skills_to_install = dict(SKILLS)

    skills_dir = base / layout["skills_dir"]
    commands_dir = base / layout["commands_dir"] if layout["commands_dir"] else None
    command_ext = layout["command_ext"]

    installed = []
    for skill_key, skill_name in skills_to_install.items():
        src = _skills_src() / skill_name
        if not src.exists():
            print(f"Warning: skill source not found: {src}")
            continue

        dest = skills_dir / skill_name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest, dirs_exist_ok=True)

        if commands_dir is not None and command_ext is not None:
            commands_dir.mkdir(parents=True, exist_ok=True)
            skill_md = src / "SKILL.md"
            if skill_md.exists():
                shutil.copy2(skill_md, commands_dir / f"{skill_name}{command_ext}")

        location = str(dest)
        installed.append(f"  {skill_name} → {location}")

    if installed:
        scope = "local" if local else "global"
        print(f"Installed skills ({scope}, target={target}):")
        for line in installed:
            print(line)
    else:
        print("No skills installed.")


def list_skills() -> None:
    targets = ", ".join(TARGET_LAYOUT.keys())
    header = f"{'Skill':<12}  {'Directory':<20}  Targets"
    print(header)
    print("-" * len(header))
    for key, name in SKILLS.items():
        print(f"{key:<12}  {name:<20}  {targets}")
