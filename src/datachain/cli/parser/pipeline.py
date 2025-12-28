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

    pipeline_create_help = "Create a pipeline to update datasets in Studio"
    pipeline_create_description = (
        "Creates a pipeline in Studio to update the specified datasets. "
        "The pipeline automatically includes all necessary jobs to update "
        "the datasets based on their dependencies.\n\n"
        "Each dataset name can optionally include a version suffix "
        "(e.g., dataset@1.0.9). If no version is specified, the latest "
        "version is used.\n\n"
        "The pipeline is created in paused state for review. Use "
        "`datachain pipeline resume` to start execution.\n\n"
        "Dataset names can be provided in fully qualified format "
        "(e.g., @namespace.project.name) or as a short name. "
        "Short names use the default project and namespace from Studio."
    )
    pipeline_create_parser = pipeline_subparser.add_parser(
        "create",
        parents=[parent_parser],
        description=pipeline_create_description,
        help=pipeline_create_help,
        formatter_class=CustomHelpFormatter,
    )
    pipeline_create_parser.add_argument(
        "datasets",
        type=str,
        nargs="+",
        help=(
            "Dataset name(s). Can be fully qualified (e.g., @namespace.project.name) "
            "or short names. Optionally include version suffix: name@version"
        ),
    )
    pipeline_create_parser.add_argument(
        "-t",
        "--team",
        action="store",
        default=None,
        help="Team to create the pipeline for (default: from config)",
    )

    pipeline_status_help = "Get the status of a pipeline from Studio"
    pipeline_status_description = (
        "This command fetches the latest status of a pipeline along with "
        "the status of its jobs from Studio."
    )
    pipeline_status_parser = pipeline_subparser.add_parser(
        "status",
        parents=[parent_parser],
        description=pipeline_status_description,
        help=pipeline_status_help,
        formatter_class=CustomHelpFormatter,
    )
    pipeline_status_parser.add_argument(
        "name",
        type=str,
        action="store",
        help="Name of the pipeline",
    )
    pipeline_status_parser.add_argument(
        "-t",
        "--team",
        action="store",
        default=None,
        help="Team of the pipeline",
    )

    pipeline_list_help = "List pipelines"
    pipeline_list_description = "List pipelines in Studio"
    pipeline_list_parser = pipeline_subparser.add_parser(
        "list",
        parents=[parent_parser],
        description=pipeline_list_description,
        help=pipeline_list_help,
        formatter_class=CustomHelpFormatter,
    )
    pipeline_list_parser.add_argument(
        "-t",
        "--team",
        action="store",
        default=None,
        help="Team to list pipelines for.",
    )
    pipeline_list_parser.add_argument(
        "-s",
        "--status",
        action="store",
        default=None,
        help=(
            "Status of the pipelines to list. Possible values are: "
            "'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'PAUSED', 'CANCELED'"
        ),
    )
    pipeline_list_parser.add_argument(
        "-l",
        "--limit",
        action="store",
        type=int,
        default=20,
        help="Limit the number of pipelines to list",
    )
    pipeline_list_parser.add_argument(
        "-S",
        "--search",
        action="store",
        default=None,
        help="Search for pipelines by name or the dataset created from.",
    )

    pipeline_pause_help = "Pause a pipeline"
    pipeline_pause_description = "Pause a pipeline in Studio"
    pipeline_pause_parser = pipeline_subparser.add_parser(
        "pause",
        parents=[parent_parser],
        description=pipeline_pause_description,
        help=pipeline_pause_help,
        formatter_class=CustomHelpFormatter,
    )
    pipeline_pause_parser.add_argument(
        "name",
        type=str,
        action="store",
        help="Name of the pipeline",
    )
    pipeline_pause_parser.add_argument(
        "-t",
        "--team",
        action="store",
        default=None,
        help="Team of the pipeline",
    )

    pipeline_resume_help = "Resume a pipeline"
    pipeline_resume_description = "Resume a pipeline in Studio"
    pipeline_resume_parser = pipeline_subparser.add_parser(
        "resume",
        parents=[parent_parser],
        description=pipeline_resume_description,
        help=pipeline_resume_help,
        formatter_class=CustomHelpFormatter,
    )
    pipeline_resume_parser.add_argument(
        "name",
        type=str,
        action="store",
        help="Name of the pipeline",
    )
    pipeline_resume_parser.add_argument(
        "-t",
        "--team",
        action="store",
        default=None,
        help="Team of the pipeline",
    )

    pipeline_remove_job_help = "Remove a job from a pipeline"
    pipeline_remove_job_description = (
        "Remove a specific job from a pipeline before it runs in Studio"
    )
    pipeline_remove_job_parser = pipeline_subparser.add_parser(
        "remove-job",
        parents=[parent_parser],
        description=pipeline_remove_job_description,
        help=pipeline_remove_job_help,
        formatter_class=CustomHelpFormatter,
    )
    pipeline_remove_job_parser.add_argument(
        "name",
        type=str,
        action="store",
        help="Name of the pipeline",
    )
    pipeline_remove_job_parser.add_argument(
        "job_id",
        type=str,
        action="store",
        help="ID of the job to remove",
    )
    pipeline_remove_job_parser.add_argument(
        "-t",
        "--team",
        action="store",
        default=None,
        help="Team of the pipeline",
    )
