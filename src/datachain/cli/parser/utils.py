from argparse import Action, ArgumentParser, ArgumentTypeError, HelpFormatter

from datachain.cli.utils import CommaSeparatedArgs

FIND_COLUMNS = ["du", "name", "path", "size", "type"]


class CustomHelpFormatter(HelpFormatter):
    def add_arguments(self, actions):
        # Sort arguments to move --help and --version to the end
        normal_actions = [
            a for a in actions if a.dest not in ("help", "verbose", "quiet")
        ]
        special_actions = [a for a in actions if a.dest in ("help", "verbose", "quiet")]
        super().add_arguments(normal_actions + special_actions)


class CustomArgumentParser(ArgumentParser):
    def error(self, message):
        internal_commands = ["internal-run-udf", "internal-run-udf-worker"]

        hidden_portion = "".join(f"'{cmd}', " for cmd in internal_commands)
        message = message.replace(hidden_portion, "")
        super().error(message)


def find_columns_type(
    columns_str: str,
    default_colums_str: str = "path",
) -> list[str]:
    if not columns_str:
        columns_str = default_colums_str

    return [parse_find_column(c) for c in columns_str.split(",")]


def parse_find_column(column: str) -> str:
    column_lower = column.strip().lower()
    if column_lower in FIND_COLUMNS:
        return column_lower
    raise ArgumentTypeError(
        f"Invalid column for find: '{column}' Options are: {','.join(FIND_COLUMNS)}"
    )


def add_sources_arg(parser: ArgumentParser, nargs: str | int = "+") -> Action:
    return parser.add_argument(
        "sources",
        type=str,
        nargs=nargs,
        help="Data sources - paths to source storage directories or files",
    )


def add_anon_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--anon",
        action="store_true",
        help="Use anonymous access to storage",
    )


def add_update_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "-u",
        "--update",
        action="count",
        default=0,
        help="Update cached list of files for the sources",
    )


def add_show_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--limit",
        action="store",
        default=10,
        type=int,
        help="Number of rows to show",
    )
    parser.add_argument(
        "--offset",
        action="store",
        default=0,
        type=int,
        help="Number of rows to offset",
    )
    parser.add_argument(
        "--columns",
        default=[],
        action=CommaSeparatedArgs,
        help="Columns to show",
    )
    parser.add_argument(
        "--no-collapse",
        action="store_true",
        default=False,
        help="Do not collapse the columns",
    )
    parser.add_argument(
        "--hidden",
        action="store_true",
        default=False,
        help="Show hidden fields",
    )
