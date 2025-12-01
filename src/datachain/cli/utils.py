import logging
from argparse import SUPPRESS, Action, Namespace, _AppendAction

from datachain.error import DataChainError


class BooleanOptionalAction(Action):
    """
    Creates --[no-]option style bool options.

    Defined here since it doesn't exist in argparse in Python 3.8.

    Copied from:
    https://github.com/python/cpython/blob/c33aaa9d559398bbf2b80e891bf3ae6a716e4b8c/Lib/argparse.py#L863-L901
    """

    def __init__(
        self,
        option_strings,
        dest,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None,
    ):
        _option_strings = []
        for option_string in option_strings:
            _option_strings.append(option_string)

            if option_string.startswith("--"):
                option_string = "--no-" + option_string[2:]
                _option_strings.append(option_string)

        if help is not None and default is not None and default is not SUPPRESS:
            help += " (default: %(default)s)"

        super().__init__(
            option_strings=_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            setattr(namespace, self.dest, not option_string.startswith("--no-"))

    def format_usage(self):
        return " | ".join(self.option_strings)


class CommaSeparatedArgs(_AppendAction):  # pylint: disable=protected-access
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest) or []
        items.extend(v for value in values.split(",") if (v := value.strip()))
        setattr(namespace, self.dest, list(dict.fromkeys(items)))


def get_logging_level(args: Namespace) -> int:
    if args.quiet:
        return logging.CRITICAL
    if args.verbose:
        return logging.DEBUG
    return logging.INFO


def determine_flavors(studio: bool, local: bool, all: bool, token: str | None):
    if studio and not token:
        raise DataChainError(
            "Not logged in to Studio. Log in with 'datachain auth login'."
        )

    if local or studio:
        all = False

    all = all and not (local or studio)

    return all, local, studio
