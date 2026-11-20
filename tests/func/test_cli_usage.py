import os
import subprocess
import sys
from pathlib import Path

import pytest

import datachain


@pytest.fixture
def cli_env(tmp_path):
    return {
        **{
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "DATACHAIN__METASTORE",
                "DATACHAIN__WAREHOUSE",
                "DATACHAIN_METASTORE",
                "DATACHAIN_WAREHOUSE",
            }
            and not key.startswith(
                ("DATACHAIN_METASTORE_ARG_", "DATACHAIN_WAREHOUSE_ARG_")
            )
        },
        "PYTHONPATH": os.pathsep.join(
            (
                str(Path(datachain.__file__).parent.parent),
                os.environ.get("PYTHONPATH", ""),
            )
        ),
        "DATACHAIN_DIR": str(tmp_path / "state"),
        "DATACHAIN_GLOBAL_CONFIG_DIR": str(tmp_path / "global-config"),
        "DATACHAIN_SYSTEM_CONFIG_DIR": str(tmp_path / "system-config"),
        "DATACHAIN_NO_ANALYTICS": "true",
    }


@pytest.mark.parametrize("command", ["dataset", "ds"])
@pytest.mark.parametrize("extra_args", [[], ["--anon"], ["--verbose"]])
def test_dataset_missing_subcommand_shows_usage(tmp_path, cli_env, command, extra_args):
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "datachain", command, *extra_args],
        cwd=tmp_path,
        env=cli_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "usage: datachain dataset" in result.stderr
    assert "pull" in result.stderr
    assert "ls" in result.stderr


@pytest.mark.parametrize("command", ["dataset", "ds"])
def test_dataset_help_still_succeeds(tmp_path, cli_env, command):
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "datachain", command, "--help"],
        cwd=tmp_path,
        env=cli_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0
    assert "usage: datachain dataset" in result.stdout
    assert "pull" in result.stdout
    assert "ls" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("command", ["dataset", "ds"])
def test_dataset_subcommand_still_succeeds(tmp_path, cli_env, command):
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "datachain", command, "ls", "--local"],
        cwd=tmp_path,
        env=cli_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0
    assert result.stderr == ""
