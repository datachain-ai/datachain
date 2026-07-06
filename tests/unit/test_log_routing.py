import gc
import io
import json
import logging
import os
import sys

import pytest

from datachain import log_routing
from datachain.cli import main
from datachain.log_routing import (
    INTERNAL_LOG_FD_ENV,
    INTERNAL_LOG_LEVEL_ENV,
    internal_log_fds,
    setup_internal_log_routing,
)


@pytest.fixture
def restore_loggers(monkeypatch):
    monkeypatch.setattr(log_routing, "_routed", False)
    names = ("datachain", "datachain_saas", "compute")
    saved = {
        n: (
            list(logging.getLogger(n).handlers),
            logging.getLogger(n).level,
            logging.getLogger(n).propagate,
        )
        for n in names
    }
    yield
    for name, (handlers, level, propagate) in saved.items():
        lg = logging.getLogger(name)
        for handler in lg.handlers:
            if handler not in handlers:
                handler.close()
        lg.handlers[:] = handlers
        lg.setLevel(level)
        lg.propagate = propagate


@pytest.fixture
def internal_fd(tmp_path, monkeypatch):
    out_file = tmp_path / "internal.log"
    fd = os.open(out_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    monkeypatch.setenv(INTERNAL_LOG_FD_ENV, str(fd))
    return out_file


def test_missing_env_leaves_loggers_untouched(monkeypatch, restore_loggers):
    monkeypatch.delenv(INTERNAL_LOG_FD_ENV, raising=False)
    logging.getLogger("datachain").propagate = True
    assert setup_internal_log_routing() is False
    assert logging.getLogger("datachain").propagate is True


def test_invalid_fd_is_noop(monkeypatch, restore_loggers):
    monkeypatch.setenv(INTERNAL_LOG_FD_ENV, "not-an-int")
    logging.getLogger("datachain").propagate = True
    assert setup_internal_log_routing() is False
    assert logging.getLogger("datachain").propagate is True


def test_unopenable_fd_is_noop(tmp_path, monkeypatch, restore_loggers):
    fd = os.open(tmp_path / "gone.log", os.O_WRONLY | os.O_CREAT)
    os.close(fd)
    monkeypatch.setenv(INTERNAL_LOG_FD_ENV, str(fd))
    logging.getLogger("datachain").propagate = True
    assert setup_internal_log_routing() is False
    assert logging.getLogger("datachain").propagate is True


@pytest.mark.parametrize("name", ["datachain", "datachain_saas", "compute"])
def test_records_go_to_fd_as_json(monkeypatch, restore_loggers, internal_fd, name):
    studio = io.StringIO()
    monkeypatch.setattr(sys, "stderr", studio)

    assert setup_internal_log_routing() is True
    logging.getLogger(name).info("hello-info")

    entry = json.loads(internal_fd.read_text().splitlines()[0])
    assert entry["message"] == "hello-info"
    assert entry["logger"] == name
    assert entry["level"] == "info"
    assert entry["time"]
    assert "hello-info" not in studio.getvalue()
    assert logging.getLogger(name).propagate is False


def test_warning_also_goes_to_fd(monkeypatch, restore_loggers, internal_fd):
    studio = io.StringIO()
    monkeypatch.setattr(sys, "stderr", studio)

    assert setup_internal_log_routing() is True
    logging.getLogger("compute").warning("hello-warn")

    entry = json.loads(internal_fd.read_text().splitlines()[0])
    assert entry["message"] == "hello-warn"
    assert entry["level"] == "warning"
    assert "hello-warn" not in studio.getvalue()


def test_level_env_widens_fd_band(monkeypatch, restore_loggers, internal_fd):
    monkeypatch.setenv(INTERNAL_LOG_LEVEL_ENV, "debug")

    assert setup_internal_log_routing() is True
    logging.getLogger("datachain").debug("hello-debug")

    entry = json.loads(internal_fd.read_text().splitlines()[0])
    assert entry["message"] == "hello-debug"
    assert entry["level"] == "debug"


def test_invalid_level_env_defaults_to_info(monkeypatch, restore_loggers, internal_fd):
    monkeypatch.setenv(INTERNAL_LOG_LEVEL_ENV, "not-a-level")

    assert setup_internal_log_routing() is True
    logging.getLogger("datachain").debug("hello-debug")
    logging.getLogger("datachain").info("hello-info")

    lines = internal_fd.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["message"] == "hello-info"


def test_traceback_is_a_single_json_line(restore_loggers, internal_fd):
    assert setup_internal_log_routing() is True
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("datachain").exception("udf failed")

    lines = internal_fd.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert "udf failed" in entry["message"]
    assert "Traceback" in entry["message"]
    assert "ValueError: boom" in entry["message"]


def test_fd_handler_self_disables_on_dead_fd(monkeypatch, restore_loggers, tmp_path):
    out_file = tmp_path / "internal.log"
    fd = os.open(out_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    monkeypatch.setenv(INTERNAL_LOG_FD_ENV, str(fd))
    studio = io.StringIO()
    monkeypatch.setattr(sys, "stderr", studio)

    assert setup_internal_log_routing() is True
    os.close(fd)
    logging.getLogger("datachain").info("first-after-close")
    logging.getLogger("datachain").info("second-after-close")

    assert "Logging error" not in studio.getvalue()
    assert "first-after-close" not in studio.getvalue()
    assert "second-after-close" not in studio.getvalue()


def test_setup_is_idempotent(monkeypatch, restore_loggers, internal_fd):
    assert setup_internal_log_routing() is True
    handlers = list(logging.getLogger("datachain").handlers)
    assert setup_internal_log_routing() is True
    assert logging.getLogger("datachain").handlers == handlers

    gc.collect()
    logging.getLogger("datachain").info("after-second-setup")
    assert "after-second-setup" in internal_fd.read_text()


def test_cli_routes_instead_of_default_handler(
    tmp_path, monkeypatch, restore_loggers, internal_fd
):
    monkeypatch.chdir(tmp_path)
    studio = io.StringIO()
    monkeypatch.setattr(sys, "stderr", studio)

    assert main(["completion"]) == 0
    dc_logger = logging.getLogger("datachain")
    assert [type(h).__name__ for h in dc_logger.handlers] == ["_FdJsonHandler"]
    assert dc_logger.propagate is False

    dc_logger.info("CLI_INFO")
    entry = json.loads(internal_fd.read_text().splitlines()[0])
    assert entry["message"] == "CLI_INFO"
    assert "CLI_INFO" not in studio.getvalue()


def test_cli_udf_worker_routes_before_dispatch(
    monkeypatch, restore_loggers, internal_fd
):
    monkeypatch.setattr("datachain.cli.handle_udf_runner", lambda: 0)

    assert main(["internal-run-udf-worker"]) == 0
    assert logging.getLogger("datachain").propagate is False

    logging.getLogger("datachain").info("UDF_INFO")
    entry = json.loads(internal_fd.read_text().splitlines()[0])
    assert entry["message"] == "UDF_INFO"


def test_cli_verbose_widens_fd_band(
    tmp_path, monkeypatch, restore_loggers, internal_fd
):
    monkeypatch.chdir(tmp_path)

    assert main(["completion", "-v"]) == 0
    dc_logger = logging.getLogger("datachain")
    assert dc_logger.level == logging.DEBUG

    dc_logger.debug("CLI_DEBUG")
    lines = internal_fd.read_text().splitlines()
    assert any(json.loads(line)["message"] == "CLI_DEBUG" for line in lines)


def test_internal_log_fds_forwards_open_fd(tmp_path, monkeypatch):
    fd = os.open(tmp_path / "internal.log", os.O_WRONLY | os.O_CREAT)
    monkeypatch.setenv(INTERNAL_LOG_FD_ENV, str(fd))
    try:
        assert internal_log_fds() == (fd,)
    finally:
        os.close(fd)


def test_internal_log_fds_empty_without_env(monkeypatch):
    monkeypatch.delenv(INTERNAL_LOG_FD_ENV, raising=False)
    assert internal_log_fds() == ()


def test_internal_log_fds_empty_on_invalid_value(monkeypatch):
    monkeypatch.setenv(INTERNAL_LOG_FD_ENV, "not-an-int")
    assert internal_log_fds() == ()


def test_internal_log_fds_empty_on_closed_fd(tmp_path, monkeypatch):
    fd = os.open(tmp_path / "internal.log", os.O_WRONLY | os.O_CREAT)
    os.close(fd)
    monkeypatch.setenv(INTERNAL_LOG_FD_ENV, str(fd))
    assert internal_log_fds() == ()


def test_cli_default_handler_without_routing_env(
    tmp_path, monkeypatch, restore_loggers
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(INTERNAL_LOG_FD_ENV, raising=False)
    dc_logger = logging.getLogger("datachain")
    dc_logger.handlers[:] = []

    assert main(["completion"]) == 0
    assert [type(h).__name__ for h in dc_logger.handlers] == ["StreamHandler"]
    assert dc_logger.level == logging.INFO
