import re
import shutil
import sys
from importlib.resources import files
from pathlib import Path

SKILLS = ("datachain", "core", "knowledge")
DEFAULT_SKILLS = ("datachain",)
_INSTALL_MARKER = ".datachain-installed"
# Installable once, retired since. An upgrade must still be able to remove these:
# a stale skill directory keeps instructing the agent that reads it.
RETIRED_SKILLS = ("jobs",)
SKILL_DEPENDENCIES = {"knowledge": ("core",)}


TARGET_LAYOUT: dict[str, dict[bool, tuple[str, str | None, str]]] = {
    "claude": {
        False: (".claude/skills", None, ".md"),
        True: (".claude/skills", ".claude/commands", ".md"),
    },
    "cursor": dict.fromkeys((False, True), (".cursor/skills", ".cursor/rules", ".mdc")),
    "codex": dict.fromkeys((False, True), (".codex/skills", None, "")),
    "pi": {
        False: (".pi/agent/skills", ".pi/agent/prompts", ".md"),
        True: (".pi/skills", ".pi/prompts", ".md"),
    },
    "copilot": {
        False: (".copilot/skills", ".copilot/instructions", ".instructions.md"),
        True: (".datachain/skills", ".github/instructions", ".instructions.md"),
    },
}


def _layout(target: str, local: bool) -> tuple[Path, Path | None, str]:
    skills_dir, commands_dir, command_ext = TARGET_LAYOUT[target][local]
    base = Path() if local else Path.home()
    return base / skills_dir, base / commands_dir if commands_dir else None, command_ext


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
        text = (skill_dest / "SKILL.md").read_text(encoding="utf-8", errors="replace")
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
    except (OSError, UnicodeError):
        return False


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
    names: tuple[str, ...],
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
    src: Path,
    dest: Path,
    skills_dir: Path,
    commands_dir: Path,
    command_ext: str,
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
            print(
                f"Cannot overwrite unrelated skill: {dest}. Remove it to continue.",
                file=sys.stderr,
            )
            return False
    return True


def install_skills(skills: str | None, target: str, local: bool) -> int:
    skills_dir, commands_dir, command_ext = _layout(target, local)

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
        fresh = not dest.exists()
        dest.mkdir(parents=True, exist_ok=True)
        try:
            (dest / _INSTALL_MARKER).write_text(skill_name, encoding="utf-8")
            shutil.copytree(src, dest, dirs_exist_ok=True, ignore=_COPYTREE_IGNORE)
        except BaseException:
            if fresh:
                shutil.rmtree(dest)
            raise
        if skill_name == "datachain":
            shutil.copyfile(_skills_src() / "core" / "SDK.md", dest / "SDK.md")

        # Resolve {skill_dir} and {<name>_skill_dir} placeholders in installed
        # SKILL.md so the agent doesn't have to probe the filesystem to find its
        # own scripts or another skill's files.
        installed_skill_md = dest / "SKILL.md"
        if installed_skill_md.exists():
            installed_skill_md.write_text(
                _resolve_placeholders(
                    installed_skill_md.read_text(encoding="utf-8"),
                    skills_dir,
                    dest,
                ),
                encoding="utf-8",
            )

        if commands_dir is not None:
            _write_companion(src, dest, skills_dir, commands_dir, command_ext)

        installed.append(f"  {skill_name} -> {dest}")

    old_skills: tuple[str, ...] = RETIRED_SKILLS
    if "datachain" in skills_to_install and not missing:
        old_skills += tuple(
            name
            for name in ("core", "knowledge")
            if not (skills_dir / name / _INSTALL_MARKER).exists()
        )
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
    skills_dir, commands_dir, command_ext = _layout(target, local)

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

    removed = []
    not_found = []
    for skill_name in skills_to_uninstall:
        if _remove_skill(skills_dir, commands_dir, command_ext, skill_name):
            removed.append(f"  {skill_name}")
        elif skills:
            not_found.append(skill_name)

    if removed:
        scope = "local" if local else "global"
        print(f"Uninstalled skills ({scope}, target={target}):")
        for line in removed:
            print(line)
    if not_found:
        print(f"Not found (already uninstalled): {', '.join(not_found)}")

    if not removed and not not_found:
        print("No DataChain skills found.")

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
    text = text.replace("{skill_dir}", dest.as_posix())
    for name in SKILLS:
        text = text.replace(f"{{{name}_skill_dir}}", (skills_dir / name).as_posix())
    return text
