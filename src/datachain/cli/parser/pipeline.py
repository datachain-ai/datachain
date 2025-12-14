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

    pipeline_create_help = "Create a pipeline to update a dataset in Studio"
    pipeline_create_description = (
        "This command creates a pipeline in Studio that will update the specified"
        " dataset. The pipeline automatically includes all necessary jobs to update"
        " the dataset based on its dependencies. "
        "If no version is specified, the latest version of the dataset is used.\n\n"
        "The pipeline is created in paused state. Use `datachain pipeline resume`"
        " to start pipeline execution.\n\n"
        "The dataset name can be provided in fully qualified format "
        "(e.g., @namespace.project.name) or as a short name. "
        "If using a short name, Studio uses the default project and namespace."
    )
    pipeline_create_parser = pipeline_subparser.add_parser(
        "create",
        parents=[parent_parser],
        description=pipeline_create_description,
        help=pipeline_create_help,
        formatter_class=CustomHelpFormatter,
    )
    pipeline_create_parser.add_argument(
        "dataset",
        type=str,
        action="store",
        help=(
            "Name of the dataset. Can be a fully qualified name "
            "(e.g., @namespace.project.name) or a short name"
        ),
    )
    pipeline_create_parser.add_argument(
        "-V",
        "--version",
        type=str,
        action="store",
        default=None,
        help="Dataset version to create the pipeline for (default: latest version)",
    )
    pipeline_create_parser.add_argument(
        "-t",
        "--team",
        action="store",
        default=None,
        help="Team to create the pipeline for (default: from config)",
    )
