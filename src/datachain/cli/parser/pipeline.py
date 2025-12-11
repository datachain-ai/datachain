from datachain.cli.parser.utils import CustomHelpFormatter


def add_pipeline_parser(subparsers, parent_parser) -> None:
    pipeline_helper = "Manage pipelines in Studio"
    pipeline_description = "Commands to manage pipelines in Studio."
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        parents=[parent_parser],
        description=pipeline_description,
        help=pipeline_helper,
        formatter_class=CustomHelpFormatter,
    )
    pipeline_subparser = pipeline_parser.add_subparsers(
        dest="cmd",
        help="Use `datachain pipeline CMD --help` to display command-specific help",
    )

    pipeline_trigger_help = "Trigger an update for dataset dependency in Studio"
    pipeline_trigger_description = "Trigger an pipeline for a dataset dependency.\n"
    pipeline_trigger_description += (
        "The pipeline will be determined based on the dependency for the"
        " dataset in Studio and triggered accordingly."
        " The dataset name, which can be a fully qualified name including the"
        " namespace and project. Alternatively, it can be a regular name, in which"
        " case the explicitly defined namespace and project will be used if they are"
        " set; otherwise, default values will be applied."
    )
    pipeline_trigger_parser = pipeline_subparser.add_parser(
        "trigger",
        parents=[parent_parser],
        description=pipeline_trigger_description,
        help=pipeline_trigger_help,
        formatter_class=CustomHelpFormatter,
    )
    pipeline_trigger_parser.add_argument(
        "dataset",
        type=str,
        action="store",
        help=(
            "Name of the dataset (can be a fully qualified name including the "
            "namespace and project or regular name)"
        ),
    )
    pipeline_trigger_parser.add_argument(
        "-V",
        "--version",
        type=str,
        action="store",
        default=None,
        help="Version of the dataset (default: latest)",
    )
    pipeline_trigger_parser.add_argument(
        "-r",
        "--review",
        action="store_true",
        help="Review the pipeline before triggering",
    )
    pipeline_trigger_parser.add_argument(
        "-n",
        "--namespace",
        action="store",
        default=None,
        help="Namespace of the dataset",
    )
    pipeline_trigger_parser.add_argument(
        "-p",
        "--project",
        action="store",
        default=None,
        help="Project of the dataset",
    )
    pipeline_trigger_parser.add_argument(
        "-t",
        "--team",
        action="store",
        default=None,
        help="Team of the dataset",
    )
