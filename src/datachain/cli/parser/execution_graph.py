from datachain.cli.parser.utils import CustomHelpFormatter


def add_execution_graph_parser(subparsers, parent_parser) -> None:
    graph_helper = "Manage execution graphs in Studio"
    graph_description = "Commands to manage execution graphs in Studio."
    graph_parser = subparsers.add_parser(
        "graph",
        parents=[parent_parser],
        description=graph_description,
        help=graph_helper,
        formatter_class=CustomHelpFormatter,
    )
    graph_subparser = graph_parser.add_subparsers(
        dest="cmd",
        help="Use `datachain graph CMD --help` to display command-specific help",
    )

    graph_trigger_help = "Trigger an update for dataset dependency in Studio"
    graph_trigger_description = "Trigger an execution graph for a dataset dependency.\n"
    graph_trigger_description += (
        "Execution graph will figure out based on the dependency for the"
        " dataset in studio and triggered accordingly."
        " The dataset name, which can be a fully qualified name including the"
        " namespace and project. Alternatively, it can be a regular name, in which"
        " case the explicitly defined namespace and project will be used if they are"
        " set; otherwise, default values will be applied."
    )
    graph_trigger_parser = graph_subparser.add_parser(
        "trigger",
        parents=[parent_parser],
        description=graph_trigger_description,
        help=graph_trigger_help,
        formatter_class=CustomHelpFormatter,
    )
    graph_trigger_parser.add_argument(
        "dataset",
        type=str,
        action="store",
        help=(
            "Name of the dataset (can be a fully qualified name including the "
            "namespace and project or regular name)"
        ),
    )
    graph_trigger_parser.add_argument(
        "-V",
        "--version",
        type=str,
        action="store",
        default=None,
        help="Version of the dataset (default: latest)",
    )
    graph_trigger_parser.add_argument(
        "-r",
        "--review",
        action="store_true",
        help="Review the execution graph before triggering",
    )
    graph_trigger_parser.add_argument(
        "-n",
        "--namespace",
        action="store",
        default=None,
        help="Namespace of the dataset",
    )
    graph_trigger_parser.add_argument(
        "-p",
        "--project",
        action="store",
        default=None,
        help="Project of the dataset",
    )
    graph_trigger_parser.add_argument(
        "-t",
        "--team",
        action="store",
        default=None,
        help="Team of the dataset",
    )
