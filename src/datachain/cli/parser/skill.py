from datachain.cli.parser.utils import CustomHelpFormatter


def add_skill_parser(subparsers, parent_parser) -> None:
    skill_help = "Manage and install DataChain AI skills"
    skill_description = "Commands for managing DataChain skills for AI coding tools."
    skill_parser = subparsers.add_parser(
        "skill",
        parents=[parent_parser],
        description=skill_description,
        help=skill_help,
        formatter_class=CustomHelpFormatter,
    )
    skill_subparser = skill_parser.add_subparsers(
        dest="skill_cmd",
        help="Use `datachain skill CMD --help` to display command-specific help",
    )

    install_help = "Install DataChain skills into an AI coding tool"
    install_description = (
        "Install DataChain skills (dc-core, dc-graph) into an AI coding tool "
        "such as Claude Code, Cursor, or Codex."
    )
    install_parser = skill_subparser.add_parser(
        "install",
        parents=[parent_parser],
        description=install_description,
        help=install_help,
        formatter_class=CustomHelpFormatter,
    )
    install_parser.add_argument(
        "--only",
        choices=["core", "graph"],
        default=None,
        help="Install only the specified skill (default: install all)",
    )
    install_parser.add_argument(
        "--target",
        choices=["claude", "codex", "cursor"],
        default="claude",
        help="Target AI coding tool to install skills into (default: claude)",
    )
    install_parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help=(
            "Install into the current project directory instead of the user home directory"
        ),
    )

    list_help = "List available DataChain skills"
    list_description = "List all available DataChain skills and supported targets."
    skill_subparser.add_parser(
        "list",
        parents=[parent_parser],
        description=list_description,
        help=list_help,
        formatter_class=CustomHelpFormatter,
    )
