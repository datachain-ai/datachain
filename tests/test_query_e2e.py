import os
import os.path
import signal
import subprocess
import sys
import textwrap
import threading
from contextlib import nullcontext
from textwrap import dedent
from threading import Thread
from typing import IO

import pytest

tests_dir = os.path.dirname(os.path.abspath(__file__))

python_exc = sys.executable or "python3"


E2E_STEP_TIMEOUT_SEC = 90


E2E_STEPS = (
    {
        "command": (
            "datachain",
            "find",
            "--anon",
            "--name",
            "cat.1.*",
            "gs://dvcx-datalakes/dogs-and-cats/",
        ),
        "expected": dedent(
            """
            gs://dvcx-datalakes/dogs-and-cats/cat.1.jpg
            gs://dvcx-datalakes/dogs-and-cats/cat.1.json
            """
        ),
        "listing": True,
    },
    {
        "command": (
            python_exc,
            os.path.join(tests_dir, "scripts", "feature_class_parallel.py"),
        ),
        "expected_in": dedent(
            """
            dogs-and-cats/cat.1.jpg
            dogs-and-cats/cat.10.jpg
            dogs-and-cats/cat.100.jpg
            dogs-and-cats/cat.1000.jpg
            dogs-and-cats/cat.1001.jpg
            """
        ),
    },
    {
        "command": (
            python_exc,
            os.path.join(tests_dir, "scripts", "feature_class_parallel_data_model.py"),
        ),
        "expected_in": dedent(
            """
            dogs-and-cats/cat.1.jpg
            dogs-and-cats/cat.10.jpg
            dogs-and-cats/cat.100.jpg
            dogs-and-cats/cat.1000.jpg
            dogs-and-cats/cat.1001.jpg
            """
        ),
    },
    {
        # This reads from stdin, to emulate using the python REPL shell.
        "command": (python_exc, "-"),
        "stdin_file": os.path.join(
            tests_dir, "scripts", "feature_class_parallel_data_model.py"
        ),
        "expected_in": dedent(
            """
            dogs-and-cats/cat.1.jpg
            dogs-and-cats/cat.10.jpg
            dogs-and-cats/cat.100.jpg
            dogs-and-cats/cat.1000.jpg
            dogs-and-cats/cat.1001.jpg
            """
        ),
    },
    {
        "command": (
            python_exc,
            os.path.join(tests_dir, "scripts", "feature_class.py"),
        ),
        "expected_rows": dedent(
            """
                               file.path  emd.value
            0     dogs-and-cats/cat.1.jpg       512.0
            1    dogs-and-cats/cat.10.jpg       512.0
            2   dogs-and-cats/cat.100.jpg       512.0
            3  dogs-and-cats/cat.1000.jpg       512.0
            4  dogs-and-cats/cat.1001.jpg       512.0

            [Limited by 5 rows]
            """
        ),
    },
    {
        "command": (
            python_exc,
            os.path.join(tests_dir, "scripts", "name_len_slow.py"),
        ),
        "interrupt_after": "UDF Processing Started",
        "expected_in_stderr": "KeyboardInterrupt",
        "expected_not_in_stderr": "Warning",
    },
    {
        "command": ("datachain", "gc"),
        "expected": "Nothing to clean up.\n",
    },
)


def communicate_and_interrupt_process(
    process: subprocess.Popen, interrupt_after: str
) -> tuple[str, str]:
    if sys.platform == "win32":
        # Windows has a different mechanism of sending a Ctrl-C event.
        interrupt_signal = signal.CTRL_C_EVENT
    else:
        interrupt_signal = signal.SIGINT

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    lock = threading.Lock()
    interrupted = False

    def interrupt_step(stream: IO[str], output_lines: list[str]) -> None:
        nonlocal interrupted
        for line in iter(stream.readline, ""):
            output_lines.append(line)
            if not interrupted and interrupt_after in line:
                with lock:
                    if not interrupted:
                        process.send_signal(interrupt_signal)
                        interrupted = True

    watch_threads = []
    for args in (process.stdout, stdout_lines), (process.stderr, stderr_lines):
        th = Thread(target=interrupt_step, name=f"E2E-Interrupt-{args[0]}", args=args)
        watch_threads.append(th)
        th.start()

    process.wait(timeout=E2E_STEP_TIMEOUT_SEC)

    for th in watch_threads:
        th.join()

    # Close the pipes to avoid ResourceWarnings
    if process.stdout:
        process.stdout.close()
    if process.stderr:
        process.stderr.close()

    return "\n".join(stdout_lines), "\n".join(stderr_lines)


def run_step(step, catalog):
    """Run an end-to-end query test step with a command and expected output."""
    command = step["command"]
    # Note that a process.returncode of -2 is the same as the shell returncode of 130
    # (canceled by KeyboardInterrupt). Also accept 130 explicitly as some Python
    # scripts may exit with that code directly.
    interrupt_exit_codes = (-2, 130)
    if sys.platform == "win32":
        # Windows has a different mechanism of creating a process group.
        popen_args = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        # This is STATUS_CONTROL_C_EXIT which is equivalent to 0xC000013A
        interrupt_exit_codes = (3221225786, 130)
    else:
        popen_args = {"start_new_session": True}
    stdin_path = step.get("stdin_file")
    with open(stdin_path) if stdin_path else nullcontext(None) as stdin_file:
        process = subprocess.Popen(  # noqa: S603
            command,
            shell=False,
            stdin=stdin_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            env={
                **os.environ,
                "DATACHAIN__METASTORE": catalog.metastore.serialize(),
                "DATACHAIN__WAREHOUSE": catalog.warehouse.serialize(),
            },
            **popen_args,
        )
        interrupt_after = step.get("interrupt_after")
        if interrupt_after:
            stdout, stderr = communicate_and_interrupt_process(process, interrupt_after)
        else:
            stdout, stderr = process.communicate(timeout=E2E_STEP_TIMEOUT_SEC)

    if interrupt_after:
        if process.returncode not in (*interrupt_exit_codes, 11):
            print(f"Process stdout: {stdout}")
            print(f"Process stderr: {stderr}")
            raise RuntimeError(
                "Query script failed to interrupt correctly: "
                f"{process.returncode} Command: {command}"
            )
    elif process.returncode != 0:
        print(f"Process stdout: {stdout}")
        print(f"Process stderr: {stderr}")
        raise RuntimeError(
            "Query script failed with exit code: "
            f"{process.returncode} Command: {command}"
        )

    if step.get("sort_expected_lines"):
        assert sorted(stdout.split("\n")) == sorted(
            step["expected"].lstrip("\n").split("\n")
        )
    elif step.get("expected_in_stderr"):
        assert step["expected_in_stderr"] in stderr
        if step.get("expected_not_in_stderr"):
            assert step["expected_not_in_stderr"] not in stderr
    elif step.get("expected_in"):
        assert sorted(stdout.split("\n")) == sorted(
            step["expected_in"].lstrip("\n").split("\n")
        )
    elif step.get("expected_rows"):
        assert _comparable_row(stdout) == _comparable_row(step["expected_rows"])
    else:
        assert stdout == step["expected"].lstrip("\n")

    if step.get("listing"):
        assert "Listing" in stderr
    else:
        assert "Listing" not in stderr


@pytest.mark.e2e
@pytest.mark.xdist_group(name="tmpfile")
def test_query_e2e(tmp_dir, catalog_tmpfile, monkeypatch):
    """End-to-end CLI Query Test"""
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)
    for step in E2E_STEPS:
        run_step(step, catalog_tmpfile)


def _comparable_row(output: str) -> str:
    return "\n".join(
        sorted(
            [_remove_serial_index(line) for line in output.lstrip("\n").splitlines()]
        )
    )


def _remove_serial_index(output: str) -> str:
    splits = textwrap.shorten(output, width=1000).strip().split(" ")
    if splits[0].isdigit():
        return " ".join(splits[1:])
    return " ".join(splits)
