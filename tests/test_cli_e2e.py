import os
import os.path
import subprocess
from textwrap import dedent

import pytest
import tabulate

from tests.utils import skip_if_not_sqlite


def _tabulated_datasets(name, version):
    row = [
        {"Name": name, "Latest Version": f"v{version}"},
    ]
    return tabulate.tabulate(row, headers="keys")


MNT_FILE_TREE = {
    "01375.png": 324,
    "07510.png": 308,
    "08433.png": 224,
    "mnist-info.txt": 0,
    "readme.md": 1193,
    "train": {
        "1": {
            "00266.png": 186,
            "06810.png": 175,
            "08537.png": 206,
            "09396.png": 168,
            "09846.png": 208,
        },
        "2": {
            "00422.png": 257,
            "04813.png": 266,
            "05508.png": 287,
            "07576.png": 297,
            "08747.png": 319,
        },
        "3": {
            "00271.png": 303,
            "00577.png": 293,
            "01608.png": 243,
            "07194.png": 300,
            "08051.png": 304,
        },
        "train-info.md": 18,
    },
    "val": {
        "1": {
            "01837.png": 182,
            "05385.png": 182,
            "09416.png": 162,
        },
        "2": {
            "02201.png": 283,
            "02297.png": 274,
            "08967.png": 333,
        },
        "3": {
            "01805.png": 345,
            "06366.png": 302,
            "08394.png": 329,
        },
        "val-info.md": 20,
    },
}

# Note that these commands are tested in order.
E2E_STEPS = (
    {
        "command": (
            "datachain",
            "ls",
            "--anon",
            "s3://ldb-public/remote/datasets/dogs-and-cats/",
        ),
        "expected": dedent(
            """
            dogs-and-cats.tar.gz
            dogs-and-cats.zip
            license
            """
        ),
        "listing": True,
    },
    {
        "command": (
            "datachain",
            "du",
            "--anon",
            "s3://ldb-public/remote/datasets/dogs-and-cats/",
        ),
        "expected": "   9.2M s3://ldb-public/remote/datasets/dogs-and-cats/\n",
    },
    {
        "command": (
            "datachain",
            "find",
            "--iname",
            "*DOG*",
            "--anon",
            "s3://ldb-public/remote/datasets/",
        ),
        "expected": dedent(
            """
            s3://ldb-public/remote/datasets/Stanford-dog-breeds/
            s3://ldb-public/remote/datasets/dogs-and-cats/
            s3://ldb-public/remote/datasets/dogs-and-cats/dogs-and-cats.tar.gz
            s3://ldb-public/remote/datasets/dogs-and-cats/dogs-and-cats.zip
            """
        ),
        "listing": True,
    },
    {
        "command": ("datachain", "du", "--anon", "s3://ldb-public/remote/datasets/"),
        "expected": "  43.3G s3://ldb-public/remote/datasets/\n",
    },
    {
        "command": (
            "datachain",
            "cp",
            "-r",
            "--anon",
            "s3://ldb-public/remote/datasets/mnist-tiny/",
            "mnt-cp",
        ),
        "expected": "",
        "downloading": True,
        "instantiating": True,
        "files": {
            "mnt-cp": MNT_FILE_TREE,
        },
    },
    {
        "command": (
            "datachain",
            "clone",
            "-r",
            "--anon",
            "s3://ldb-public/remote/datasets/mnist-tiny/",
            "mnt",
        ),
        "expected": "",
        "downloading": True,
        "instantiating": True,
        "files": {
            "mnt": MNT_FILE_TREE,
        },
    },
    {
        "command": ("datachain", "dataset", "rm", "mnt", "--version", "1.0.0"),
        "expected": "",
    },
    {
        "command": ("datachain", "dataset", "ls"),
        "expected": "",
    },
    {
        "command": ("datachain", "gc"),
        "expected": ("No temporary tables to clean up.\n"),
    },
)


E2E_STEPS_LOCAL = (
    {
        "command": (
            "datachain",
            "clone",
            "-r",
            "--anon",
            "s3://ldb-public/remote/datasets/mnist-tiny/",
            "mnt",
        ),
        "expected": "",
        "downloading": True,
        "instantiating": True,
        "files": {
            "mnt": MNT_FILE_TREE,
        },
        "listing": True,
    },
    {
        "command": ("datachain", "dataset", "ls"),
        "expected": _tabulated_datasets("local.local.mnt", "1.0.0"),
    },
    {
        "command": ("datachain", "dataset", "ls"),
        "expected": _tabulated_datasets("local.local.mnt", "1.0.0"),
    },
    {
        "command": ("datachain", "dataset", "edit", "mnt", "--new-name", "mnt-new"),
        "expected": "",
    },
    {
        "command": ("datachain", "dataset", "ls"),
        "expected": _tabulated_datasets("local.local.mnt-new", "1.0.0"),
    },
)


def verify_files(files, base=""):
    """Recursively validate file and directory structure."""
    for name, value in files.items():
        full_name = os.path.join(base, name)
        if isinstance(value, dict):
            assert os.path.isdir(full_name)
            verify_files(value, full_name)
        else:
            assert os.path.isfile(full_name)
            assert os.path.getsize(full_name) == value


def run_step(step, catalog):
    """Run an end-to-end test step with a command and expected output."""
    result = subprocess.run(  # noqa: S603
        step["command"],
        shell=False,
        capture_output=True,
        check=True,
        encoding="utf-8",
        env={
            **os.environ,
            "DATACHAIN__METASTORE": catalog.metastore.serialize(),
            "DATACHAIN__WAREHOUSE": catalog.warehouse.serialize(),
        },
    )

    if step.get("sort_expected_lines"):
        assert sorted(result.stdout.split("\n")) == sorted(
            step["expected"].lstrip("\n").split("\n")
        )
    else:
        assert result.stdout.strip("\n") == step["expected"].strip("\n")
    if step.get("listing"):
        assert "Listing" in result.stderr
    else:
        assert "Listing" not in result.stderr
    if step.get("downloading"):
        assert "Downloading" in result.stderr
    else:
        assert "Downloading" not in result.stderr
    if step.get("instantiating"):
        assert "Instantiating" in result.stderr
    else:
        assert "Instantiating" not in result.stderr
    files = step.get("files")
    if files:
        verify_files(files)


@pytest.mark.e2e
@pytest.mark.xdist_group(name="tmpfile")
def test_cli_e2e(tmp_dir, catalog_tmpfile):
    """End-to-end CLI Test"""
    for step in E2E_STEPS:
        run_step(step, catalog_tmpfile)


@pytest.mark.e2e
@pytest.mark.xdist_group(name="tmpfile")
@skip_if_not_sqlite
def test_cli_e2e_local_only(tmp_dir, catalog_tmpfile):
    """End-to-end CLI Test"""
    for step in E2E_STEPS_LOCAL:
        run_step(step, catalog_tmpfile)
