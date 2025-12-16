import argparse
from importlib.metadata import PackageNotFoundError, version

import shtab

from datachain.cli.utils import BooleanOptionalAction

from .job import add_jobs_parser
from .pipeline import add_pipeline_parser
from .studio import add_auth_parser
from .utils import (
    FIND_COLUMNS,
    CustomHelpFormatter,
    add_anon_arg,
    add_show_args,
    add_sources_arg,
    add_update_arg,
    find_columns_type,
)
from .utils import CustomArgumentParser as ArgumentParser


def get_parser() -> ArgumentParser:  # noqa: PLR0915
    try:
        __version__ = version("datachain")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"

    parser = ArgumentParser(
        description="DataChain: Wrangle unstructured AI data at scale.",
        prog="datachain",
        formatter_class=CustomHelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version=__version__)

    parent_parser = ArgumentParser(add_help=False, formatter_class=CustomHelpFormatter)
    parent_parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Be verbose"
    )
    parent_parser.add_argument(
        "-q", "--quiet", action="count", default=0, help="Be quiet"
    )

    parent_parser.add_argument(
        "--debug-sql",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parent_parser.add_argument(
        "--pdb",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )

    subp = parser.add_subparsers(
        title="Available Commands",
        metavar="command",
        dest="command",
        help=f"Use `{parser.prog} command --help` for command-specific help",
    )
    parse_cp = subp.add_parser(
        "cp",
        parents=[parent_parser],
        description="Copy data files from the cloud.",
        formatter_class=CustomHelpFormatter,
    )
    add_sources_arg(parse_cp).complete = shtab.DIR  # type: ignore[attr-defined]
    parse_cp.add_argument(
        "output", type=str, help="Path to a directory or file to put data to"
    )
    parse_cp.add_argument(
        "-f",
        "--force",
        default=False,
        action="store_true",
        help="Force creating files even if they already exist",
    )
    parse_cp.add_argument(
        "-r",
        "-R",
        "--recursive",
        default=False,
        action="store_true",
        help="Copy directories recursively",
    )
    parse_cp.add_argument(
        "--no-glob",
        default=False,
        action="store_true",
        help="Do not expand globs (such as * or ?)",
    )
    add_anon_arg(parse_cp)
    add_update_arg(parse_cp)

    parse_clone = subp.add_parser(
        "clone",
        parents=[parent_parser],
        description="Copy data files from the cloud.",
        formatter_class=CustomHelpFormatter,
    )
    add_sources_arg(parse_clone).complete = shtab.DIR  # type: ignore[attr-defined]
    parse_clone.add_argument(
        "output", type=str, help="Path to a directory or file to put data to"
    )
    parse_clone.add_argument(
        "-f",
        "--force",
        default=False,
        action="store_true",
        help="Force creating outputs",
    )
    parse_clone.add_argument(
        "-r",
        "-R",
        "--recursive",
        default=False,
        action="store_true",
        help="Copy directories recursively",
    )
    parse_clone.add_argument(
        "--no-glob",
        default=False,
        action="store_true",
        help="Do not expand globs (such as * or ?)",
    )
    parse_clone.add_argument(
        "--no-cp",
        default=False,
        action="store_true",
        help="Do not copy files, just create a dataset",
    )
    add_anon_arg(parse_clone)
    add_update_arg(parse_clone)

    add_auth_parser(subp, parent_parser)
    add_jobs_parser(subp, parent_parser)
    add_pipeline_parser(subp, parent_parser)

    datasets_parser = subp.add_parser(
        "dataset",
        aliases=["ds"],
        parents=[parent_parser],
        description="Commands for managing datasets.",
        formatter_class=CustomHelpFormatter,
    )
    add_anon_arg(datasets_parser)
    datasets_subparser = datasets_parser.add_subparsers(
        dest="datasets_cmd",
        help="Use `datachain dataset CMD --help` to display command-specific help",
    )

    parse_pull = datasets_subparser.add_parser(
        "pull",
        parents=[parent_parser],
        description="Pull specific dataset version from Studio.",
        formatter_class=CustomHelpFormatter,
    )
    parse_pull.add_argument(
        "dataset",
        type=str,
        help="Name and version of remote dataset created in Studio",
    )
    parse_pull.add_argument("-o", "--output", type=str, help="Output")
    parse_pull.add_argument(
        "-f",
        "--force",
        default=False,
        action="store_true",
        help="Force creating outputs",
    )
    parse_pull.add_argument(
        "-r",
        "-R",
        "--recursive",
        default=False,
        action="store_true",
        help="Copy directories recursively",
    )
    parse_pull.add_argument(
        "--cp",
        default=False,
        action="store_true",
        help="Copy actual files after pulling remote dataset into local DB",
    )

    parse_pull.add_argument(
        "--local-name",
        action="store",
        default=None,
        help="Name of the local dataset",
    )
    parse_pull.add_argument(
        "--local-version",
        action="store",
        default=None,
        help="Version of the local dataset",
    )

    parse_edit_dataset = datasets_subparser.add_parser(
        "edit",
        parents=[parent_parser],
        description="Edit dataset metadata.",
        formatter_class=CustomHelpFormatter,
    )
    parse_edit_dataset.add_argument("name", type=str, help="Dataset name")
    parse_edit_dataset.add_argument(
        "--new-name",
        action="store",
        help="Dataset new name",
    )
    parse_edit_dataset.add_argument(
        "--description",
        action="store",
        help="Dataset description",
    )
    parse_edit_dataset.add_argument(
        "--attrs",
        nargs="+",
        help="Dataset attributes",
    )
    parse_edit_dataset.add_argument(
        "--team",
        action="store",
        default=None,
        help="The team to edit a dataset. By default, it will use team from config",
    )

    datasets_ls_parser = datasets_subparser.add_parser(
        "ls",
        parents=[parent_parser],
        description="List datasets.",
        formatter_class=CustomHelpFormatter,
    )
    datasets_ls_parser.add_argument(
        "name", action="store", help="Name of the dataset to list", nargs="?"
    )
    datasets_ls_parser.add_argument(
        "--versions",
        action="store_true",
        default=False,
        help="List all the versions of each dataset",
    )
    datasets_ls_parser.add_argument(
        "--studio",
        action="store_true",
        default=False,
        help="List the files in the Studio",
    )
    datasets_ls_parser.add_argument(
        "-L",
        "--local",
        action="store_true",
        default=False,
        help="List local files only",
    )
    datasets_ls_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        default=True,
        help="List all files including hidden files",
    )
    datasets_ls_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="The team to list datasets for. By default, it will use team from config",
    )

    rm_dataset_parser = datasets_subparser.add_parser(
        "rm",
        parents=[parent_parser],
        description="Remove dataset.",
        aliases=["remove"],
        formatter_class=CustomHelpFormatter,
    )
    rm_dataset_parser.add_argument("name", type=str, help="Dataset name")
    rm_dataset_parser.add_argument(
        "--version",
        action="store",
        default=None,
        type=str,
        help="Dataset version",
    )
    rm_dataset_parser.add_argument(
        "--force",
        default=False,
        action=BooleanOptionalAction,
        help="Force delete registered dataset with all of its versions",
    )
    rm_dataset_parser.add_argument(
        "--studio",
        action="store_true",
        default=False,
        help="Remove dataset from Studio only",
    )
    rm_dataset_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="The team to delete a dataset. By default, it will use team from config",
    )

    parse_ls = subp.add_parser(
        "ls",
        parents=[parent_parser],
        description="List storage contents.",
        formatter_class=CustomHelpFormatter,
    )
    add_anon_arg(parse_ls)
    add_update_arg(parse_ls)
    add_sources_arg(parse_ls, nargs="*")
    parse_ls.add_argument(
        "-l",
        "--long",
        action="count",
        default=0,
        help="List files in long format",
    )
    parse_ls.add_argument(
        "--studio",
        action="store_true",
        default=False,
        help="List the files in the Studio",
    )
    parse_ls.add_argument(
        "-L",
        "--local",
        action="store_true",
        default=False,
        help="List local files only",
    )
    parse_ls.add_argument(
        "-a",
        "--all",
        action="store_true",
        default=True,
        help="List all files including hidden files",
    )
    parse_ls.add_argument(
        "--team",
        action="store",
        default=None,
        help="The team to list datasets for. By default, it will use team from config",
    )

    parse_du = subp.add_parser(
        "du",
        parents=[parent_parser],
        description="Display space usage.",
        formatter_class=CustomHelpFormatter,
    )
    add_sources_arg(parse_du)
    add_anon_arg(parse_du)
    add_update_arg(parse_du)
    parse_du.add_argument(
        "-b",
        "--bytes",
        default=False,
        action="store_true",
        help="Display sizes in bytes instead of human-readable sizes",
    )
    parse_du.add_argument(
        "-d",
        "--depth",
        "--max-depth",
        default=0,
        type=int,
        metavar="N",
        help=(
            "Display sizes up to N directory levels deep "
            "(default: 0, summarize provided directory only)"
        ),
    )
    parse_du.add_argument(
        "--si",
        default=False,
        action="store_true",
        help="Display sizes using powers of 1000 not 1024",
    )

    parse_find = subp.add_parser(
        "find",
        parents=[parent_parser],
        description="Search in a directory hierarchy.",
        formatter_class=CustomHelpFormatter,
    )
    add_anon_arg(parse_find)
    add_update_arg(parse_find)
    add_sources_arg(parse_find)
    parse_find.add_argument(
        "--name",
        type=str,
        action="append",
        help="Match filename pattern",
    )
    parse_find.add_argument(
        "--iname",
        type=str,
        action="append",
        help="Match filename pattern (case insensitive)",
    )
    parse_find.add_argument(
        "--path",
        type=str,
        action="append",
        help="Path to match pattern",
    )
    parse_find.add_argument(
        "--ipath",
        type=str,
        action="append",
        help="Like -path but case insensitive",
    )
    parse_find.add_argument(
        "--size",
        type=str,
        help=(
            "Filter by size (+ is greater or equal, - is less or equal). "
            "Specified size is in bytes, or use a suffix like K, M, G for "
            "kilobytes, megabytes, gigabytes, etc"
        ),
    )
    parse_find.add_argument(
        "--type",
        type=str,
        help='File type: "f" - regular, "d" - directory',
    )
    parse_find.add_argument(
        "-c",
        "--columns",
        type=find_columns_type,
        default=None,
        help=(
            "A comma-separated list of columns to print for each result. "
            f"Options are: {','.join(FIND_COLUMNS)} (Default: path)"
        ),
    )

    parse_index = subp.add_parser(
        "index",
        parents=[parent_parser],
        description="Index storage location.",
        formatter_class=CustomHelpFormatter,
    )
    add_anon_arg(parse_index)
    add_update_arg(parse_index)
    add_sources_arg(parse_index)

    show_parser = subp.add_parser(
        "show",
        parents=[parent_parser],
        description="Create a new dataset with a query script.",
        formatter_class=CustomHelpFormatter,
    )
    show_parser.add_argument("name", type=str, help="Dataset name")
    show_parser.add_argument(
        "--version",
        action="store",
        default=None,
        type=str,
        help="Dataset version",
    )
    show_parser.add_argument("--schema", action="store_true", help="Show schema")
    add_show_args(show_parser)

    parse_clear_cache = subp.add_parser(
        "clear-cache",
        parents=[parent_parser],
        description="Clear the local file cache.",
        formatter_class=CustomHelpFormatter,
    )
    add_anon_arg(parse_clear_cache)

    parse_gc = subp.add_parser(
        "gc",
        parents=[parent_parser],
        description="Garbage collect temporary tables and failed dataset versions.",
        formatter_class=CustomHelpFormatter,
    )
    add_anon_arg(parse_gc)

    subp.add_parser("internal-run-udf", parents=[parent_parser])
    subp.add_parser("internal-run-udf-worker", parents=[parent_parser])

    add_completion_parser(subp, [parent_parser])
    return parser


def add_completion_parser(subparsers, parents):
    parser = subparsers.add_parser(
        "completion",
        parents=parents,
        description="Output shell completion script.",
        formatter_class=CustomHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--shell",
        help="Shell syntax for completions",
        default="bash",
        choices=shtab.SUPPORTED_SHELLS,
    )
