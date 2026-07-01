import logging
from argparse import ArgumentTypeError

import pytest

from datachain.cli import get_logging_level, get_parser
from datachain.cli.parser.utils import CustomArgumentParser as ArgumentParser
from datachain.cli.parser.utils import find_columns_type
from datachain.cli.utils import CommaSeparatedArgs, determine_flavors
from datachain.error import DataChainError


def test_find_columns_type():
    assert find_columns_type("") == ["path"]
    assert find_columns_type("du") == ["du"]
    assert find_columns_type("", default_colums_str="name") == ["name"]
    assert find_columns_type("du, name,PATH") == ["du", "name", "path"]

    with pytest.raises(ArgumentTypeError):
        find_columns_type("bogus")


def test_cli_parser():
    parser = get_parser()

    args = parser.parse_args(("ls", "s3://example-bucket/"))

    assert args.sources == ["s3://example-bucket/"]

    assert args.quiet == 0
    assert args.verbose == 0

    assert get_logging_level(args) == logging.INFO

    args = parser.parse_args(("ls", "s3://example-bucket/", "-vvv"))

    assert args.quiet == 0
    assert args.verbose == 3

    assert get_logging_level(args) == logging.DEBUG

    args = parser.parse_args(("ls", "s3://example-bucket/", "-q"))

    assert args.quiet == 1
    assert args.verbose == 0

    assert get_logging_level(args) == logging.CRITICAL


@pytest.mark.parametrize(
    "param,parsed",
    (
        ("p1", ["p1"]),
        ("p1,p2", ["p1", "p2"]),
    ),
)
def test_comma_separated_args(param, parsed):
    parser = ArgumentParser()
    parser.add_argument("--param", default=[], action=CommaSeparatedArgs)

    args = parser.parse_args(("--param", param))
    assert args.param == parsed


@pytest.mark.parametrize("param", (None, ""))
def test_comma_separated_args_error(param):
    parser = ArgumentParser()
    parser.add_argument("--param", default=[], action=CommaSeparatedArgs)

    cmd = ["--param"]
    if param:
        cmd.append(param)
    with pytest.raises(SystemExit):
        parser.parse_args(cmd)


def test_bucket_status_parser():
    parser = get_parser()
    args = parser.parse_args(("bucket", "status", "s3://my-bucket/"))
    assert args.command == "bucket"
    assert args.bucket_cmd == "status"
    assert args.uri == "s3://my-bucket/"


def test_bucket_status_account_name():
    parser = get_parser()
    args = parser.parse_args(
        ("bucket", "status", "--account-name", "myaccount", "az://my-container/")
    )
    assert args.account_name == "myaccount"
    assert args.uri == "az://my-container/"


def test_bucket_no_subcommand():
    parser = get_parser()
    args = parser.parse_args(("bucket",))
    assert args.command == "bucket"
    assert args.bucket_cmd is None


def test_ls_flavor_defaults():
    parser = get_parser()
    args = parser.parse_args(("ls", "s3://example/"))
    assert args.local is False
    assert args.studio is False
    assert args.all is False


def test_datasets_ls_flavor_defaults():
    parser = get_parser()
    args = parser.parse_args(("dataset", "ls"))
    assert args.local is False
    assert args.studio is False
    assert args.all is False


@pytest.mark.parametrize(
    "studio,local,all,token,expected",
    [
        # No flags defaults to local-only (regardless of token presence)
        (False, False, False, None, (False, True, False)),
        (False, False, False, "tok", (False, True, False)),
        # Explicit --all keeps both on with a token, downgrades to local without
        (False, False, True, "tok", (True, False, False)),
        (False, False, True, None, (False, True, False)),
        # Explicit single-source flags clear --all
        (True, False, False, "tok", (False, False, True)),
        (False, True, False, None, (False, True, False)),
        (True, True, False, "tok", (False, True, True)),
    ],
)
def test_determine_flavors(studio, local, all, token, expected):
    assert determine_flavors(studio, local, all, token) == expected


def test_determine_flavors_studio_without_token():
    with pytest.raises(DataChainError, match="Not logged in to Studio"):
        determine_flavors(studio=True, local=False, all=False, token=None)
