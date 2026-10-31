import os
import subprocess
import sys

import pytest

from datachain.sql.sqlite import base


@pytest.fixture
def fresh_lookup(monkeypatch):
    base.usearch_sqlite_path.cache_clear()
    monkeypatch.delenv(base.USEARCH_SQLITE_ENV, raising=False)
    yield
    base.usearch_sqlite_path.cache_clear()


def test_sqlite_setup_does_not_import_usearch_into_the_process():
    code = (
        "import sys, datachain.data_storage.sqlite as s\n"
        "from datachain.sql.sqlite.base import usearch_sqlite_path\n"
        "usearch_sqlite_path()\n"
        "assert 'usearch' not in sys.modules\n"
        "assert 'numkong' not in sys.modules\n"
        "import os; assert 'DATACHAIN_USEARCH_SQLITE' in os.environ\n"
    )
    env = {k: v for k, v in os.environ.items() if k != base.USEARCH_SQLITE_ENV}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)  # noqa: S603


@pytest.mark.parametrize(
    "value,expected", [("", None), ("/x/usearch_sqlite", "/x/usearch_sqlite")]
)
def test_usearch_path_from_env_skips_the_lookup(
    value, expected, fresh_lookup, monkeypatch
):
    monkeypatch.setenv(base.USEARCH_SQLITE_ENV, value)
    monkeypatch.setattr(base, "_resolve_usearch_sqlite_path", pytest.fail)

    assert base.usearch_sqlite_path() == expected


def test_usearch_path_is_resolved_once_and_shared_with_children(
    fresh_lookup, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        base, "_resolve_usearch_sqlite_path", lambda: calls.append(1) or "/p/ext"
    )

    assert base.usearch_sqlite_path() == "/p/ext"
    assert base.usearch_sqlite_path() == "/p/ext"
    assert calls == [1]
    assert os.environ[base.USEARCH_SQLITE_ENV] == "/p/ext"


def test_missing_usearch_extension_is_recorded_as_empty(fresh_lookup, monkeypatch):
    monkeypatch.setattr(base, "_resolve_usearch_sqlite_path", lambda: None)

    assert base.usearch_sqlite_path() is None
    assert os.environ[base.USEARCH_SQLITE_ENV] == ""


def test_failed_usearch_lookup_subprocess_means_no_extension(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(base.subprocess, "run", fail)

    assert base._resolve_usearch_sqlite_path() is None


def test_unloadable_usearch_extension_warns_once(caplog, monkeypatch):
    import logging
    import sqlite3

    class Conn:
        def enable_load_extension(self, enabled):
            pass

        def load_extension(self, path):
            raise sqlite3.OperationalError("libsqlite3.so: cannot open shared object")

    monkeypatch.setattr(base, "usearch_sqlite_path", lambda: "/x/usearch_sqlite")
    base._warn_usearch_unloadable.cache_clear()

    with caplog.at_level(logging.WARNING, logger="datachain"):
        assert [base.load_usearch_extension(Conn()) for _ in range(3)] == [False] * 3

    warnings = [
        r.getMessage() for r in caplog.records if "does not load" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "libsqlite3.so" in warnings[0]
