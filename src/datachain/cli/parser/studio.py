from datachain.cli.parser.utils import CustomHelpFormatter


def add_auth_parser(subparsers, parent_parser) -> None:
    from dvc_studio_client.auth import AVAILABLE_SCOPES

    auth_help = "Manage Studio authentication"
    auth_description = "Manage authentication and settings for Studio. "

    auth_parser = subparsers.add_parser(
        "auth",
        parents=[parent_parser],
        description=auth_description,
        help=auth_help,
        formatter_class=CustomHelpFormatter,
    )
    auth_subparser = auth_parser.add_subparsers(
        dest="cmd",
        help="Use `datachain auth CMD --help` to display command-specific help",
    )

    auth_login_help = "Authenticate with Studio"
    auth_login_description = (
        "Authenticate with Studio using default scopes. "
        "A random name will be assigned if the token name is not specified."
    )

    allowed_scopes = ", ".join(AVAILABLE_SCOPES)
    login_parser = auth_subparser.add_parser(
        "login",
        parents=[parent_parser],
        description=auth_login_description,
        help=auth_login_help,
        formatter_class=CustomHelpFormatter,
    )

    login_parser.add_argument(
        "-H",
        "--hostname",
        action="store",
        default=None,
        help="Hostname of the Studio instance",
    )
    login_parser.add_argument(
        "-s",
        "--scopes",
        action="store",
        default=None,
        help=f"Authentication token scopes. Allowed scopes: {allowed_scopes}",
    )

    login_parser.add_argument(
        "-n",
        "--name",
        action="store",
        default=None,
        help="Authentication token name (shown in Studio profile)",
    )

    login_parser.add_argument(
        "--no-open",
        action="store_true",
        default=False,
        help="Use code-based authentication without browser",
    )
    login_parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help=(
            "Save the token in the project's .datachain/credentials, "
            "which is used before the user's tokens"
        ),
    )

    login_parser.add_argument(
        "--team",
        action="append",
        help=("Create a token for this team, used only for it (can specify multiple)"),
    )

    # Expiration arguments
    login_parser.add_argument(
        "--expires-in",
        type=int,
        default=365,
        help="Token expiration in days (default: 365)",
    )

    auth_logout_help = "Log out from Studio"
    auth_logout_description = (
        "Revoke and remove the Studio token for all teams. With --team, the "
        "token for that team instead; with --all, every token for the Studio."
    )

    logout_parser = auth_subparser.add_parser(
        "logout",
        parents=[parent_parser],
        description=auth_logout_description,
        help=auth_logout_help,
        formatter_class=CustomHelpFormatter,
    )
    logout_parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Remove the token from the project's .datachain/credentials",
    )
    logout_tokens = logout_parser.add_mutually_exclusive_group()
    logout_tokens.add_argument(
        "--team",
        action="store",
        default=None,
        help="Remove the token for this team instead of the token for all teams",
    )
    logout_tokens.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Remove every token for the Studio: for all teams and for each team",
    )
    logout_parser.add_argument(
        "-H",
        "--hostname",
        action="store",
        default=None,
        help="Hostname of the Studio instance (default: the current one)",
    )

    auth_team_help = "Set or show default team for Studio operations"
    auth_team_description = (
        "Set or show the default team for Studio operations. "
        "This will be used globally by default. "
        "Use --local to set the team locally for the current project. "
        "If no team name is provided, the default team will be shown."
    )

    team_parser = auth_subparser.add_parser(
        "team",
        parents=[parent_parser],
        description=auth_team_description,
        help=auth_team_help,
        formatter_class=CustomHelpFormatter,
    )
    team_parser.add_argument(
        "team_name",
        action="store",
        default=None,
        nargs="?",
        help="Name of the team to set as default",
    )
    team_parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Set team locally for the current project",
    )

    auth_token_help = "View Studio authentication token"  # noqa: S105
    auth_token_description = (
        "Display the token DataChain uses for the current Studio and team."  # noqa: S105
    )

    token_parser = auth_subparser.add_parser(
        "token",
        parents=[parent_parser],
        description=auth_token_description,
        help=auth_token_help,
        formatter_class=CustomHelpFormatter,
    )
    token_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="Show the token used for this team instead of the default team",
    )

    auth_list_help = "List saved Studio tokens"
    auth_list_description = (
        "List the Studio tokens saved for this project and user, and mark the one "
        "used for the current Studio and team."
    )

    auth_subparser.add_parser(
        "list",
        parents=[parent_parser],
        description=auth_list_description,
        help=auth_list_help,
        formatter_class=CustomHelpFormatter,
    )
