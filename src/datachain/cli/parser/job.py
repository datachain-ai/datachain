from datachain.cli.parser.utils import CustomHelpFormatter


def add_jobs_parser(subparsers, parent_parser) -> None:
    jobs_help = "Manage jobs in Studio"
    jobs_description = "Commands to manage job execution in Studio."
    jobs_parser = subparsers.add_parser(
        "job",
        parents=[parent_parser],
        description=jobs_description,
        help=jobs_help,
        formatter_class=CustomHelpFormatter,
    )
    jobs_subparser = jobs_parser.add_subparsers(
        dest="cmd",
        help="Use `datachain job CMD --help` to display command-specific help",
    )

    studio_run_help = "Run a job in Studio"
    studio_run_description = "Run a job in Studio. \n"
    studio_run_description += (
        "When using --start-time or --cron,"
        " the job is scheduled to run but won't start immediately"
        " (can be seen in the Tasks tab in UI)"
    )

    studio_run_parser = jobs_subparser.add_parser(
        "run",
        parents=[parent_parser],
        description=studio_run_description,
        help=studio_run_help,
        formatter_class=CustomHelpFormatter,
    )

    studio_run_parser.add_argument(
        "file",
        action="store",
        help="Query file to run",
    )

    studio_run_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="Team to run job for (default: from config)",
    )
    studio_run_parser.add_argument(
        "--env-file",
        action="store",
        help="File with environment variables for the job",
    )

    studio_run_parser.add_argument(
        "--env",
        nargs="+",
        action="append",
        help="Environment variables in KEY=VALUE format",
    )

    studio_run_parser.add_argument(
        "--cluster",
        type=str,
        action="store",
        help="Compute cluster to run the job on",
    )

    studio_run_parser.add_argument(
        "-c",
        "--credentials-name",
        action="store",
        help="Name of the credentials to use for the job",
    )

    studio_run_parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers for the job",
    )
    studio_run_parser.add_argument(
        "--files",
        nargs="+",
        help="Additional files to include in the job",
    )
    studio_run_parser.add_argument(
        "--python-version",
        action="store",
        help="Python version for the job (e.g., 3.10, 3.11, 3.12, 3.13)",
    )
    studio_run_parser.add_argument(
        "--repository",
        action="store",
        help="Repository URL to clone before running the job",
    )
    studio_run_parser.add_argument(
        "--req-file",
        action="store",
        help="Python requirements file",
    )

    studio_run_parser.add_argument(
        "--req",
        nargs="+",
        help="Python package requirements",
    )
    studio_run_parser.add_argument(
        "--priority",
        type=int,
        default=5,
        help="Priority for the job in range 0-5. "
        "Lower value is higher priority (default: 5)",
    )
    studio_run_parser.add_argument(
        "--start-time",
        action="store",
        help="Time to schedule a task in YYYY-MM-DDTHH:mm format or natural language.",
    )
    studio_run_parser.add_argument(
        "--cron", action="store", help="Cron expression for the cron task."
    )
    studio_run_parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for the job to finish",
    )
    studio_run_parser.add_argument(
        "--no-follow",
        action="store_true",
        help="Do not print the job logs to the console",
    )
    studio_run_parser.add_argument(
        "--ignore-checkpoints",
        action="store_true",
        help="Ignore existing checkpoints and run from scratch",
    )

    studio_ls_help = "List jobs in Studio"
    studio_ls_description = "List jobs in Studio."

    studio_ls_parser = jobs_subparser.add_parser(
        "ls",
        parents=[parent_parser],
        description=studio_ls_description,
        help=studio_ls_help,
        formatter_class=CustomHelpFormatter,
    )

    studio_ls_parser.add_argument(
        "--status",
        action="store",
        help="Status to filter jobs by",
    )

    studio_ls_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="Team to list jobs for (default: from config)",
    )
    studio_ls_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit the number of jobs returned (default: 20)",
    )

    studio_cancel_help = "Cancel a job in Studio"
    studio_cancel_description = "Cancel a running job in Studio."

    studio_cancel_parser = jobs_subparser.add_parser(
        "cancel",
        parents=[parent_parser],
        description=studio_cancel_description,
        help=studio_cancel_help,
        formatter_class=CustomHelpFormatter,
    )

    studio_cancel_parser.add_argument(
        "id",
        action="store",
        help="Job ID to cancel",
    )
    studio_cancel_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="Team to cancel job for (default: from config)",
    )

    studio_log_help = "Show job logs and status in Studio"
    studio_log_description = "Display logs and current status of jobs in Studio."

    studio_log_parser = jobs_subparser.add_parser(
        "logs",
        parents=[parent_parser],
        description=studio_log_description,
        help=studio_log_help,
        formatter_class=CustomHelpFormatter,
    )

    studio_log_parser.add_argument(
        "id",
        action="store",
        help="Job ID to show logs for",
    )
    studio_log_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="Team to check logs for (default: from config)",
    )

    studio_clusters_help = "List compute clusters in Studio"
    studio_clusters_description = "List compute clusters in Studio."

    studio_clusters_parser = jobs_subparser.add_parser(
        "clusters",
        parents=[parent_parser],
        description=studio_clusters_description,
        help=studio_clusters_help,
        formatter_class=CustomHelpFormatter,
    )

    studio_clusters_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="Team to list clusters for (default: from config)",
    )
