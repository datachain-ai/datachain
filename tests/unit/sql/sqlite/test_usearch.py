import importlib.util
import logging
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

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
        "import sys, datachain.data_storage.sqlite\n"
        "assert 'usearch' not in sys.modules\n"
        "assert 'numkong' not in sys.modules\n"
        "import os; assert 'DATACHAIN_USEARCH_SQLITE' in os.environ\n"
    )
    env = {k: v for k, v in os.environ.items() if k != base.USEARCH_SQLITE_ENV}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)  # noqa: S603


@pytest.mark.skipif(
    sys.platform == "win32" or importlib.util.find_spec("usearch") is None,
    reason="usearch SQLite extension is unavailable",
)
def test_usearch_sqlite_path_finds_the_extension():
    path = base._resolve_usearch_sqlite_path()

    assert path
    assert list(Path(path).parent.glob(f"{Path(path).name}.*"))


def test_usearch_lookup_inherits_runtime_import_path(tmp_path, monkeypatch):
    package = tmp_path / "usearch"
    package.mkdir()
    (package / "__init__.py").write_text(
        "def sqlite_path(): return '/runtime/extension'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    assert base._resolve_usearch_sqlite_path() == "/runtime/extension"
    assert "usearch" not in sys.modules


@pytest.mark.parametrize("resolved", ["/p/usearch_sqlite", None, ""])
def test_usearch_path_is_resolved_once_and_inherited(
    resolved, fresh_lookup, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        base, "_resolve_usearch_sqlite_path", lambda: calls.append(1) or resolved
    )

    assert base.usearch_sqlite_path() == (resolved or None)
    base.usearch_sqlite_path.cache_clear()
    assert base.usearch_sqlite_path() == (resolved or None)
    assert calls == [1]


def test_missing_usearch_skips_subprocess(monkeypatch, mocker):
    monkeypatch.setattr(base.importlib.util, "find_spec", lambda name: None)
    run = mocker.patch.object(base.subprocess, "run")

    assert base._resolve_usearch_sqlite_path() is None
    run.assert_not_called()


@pytest.mark.parametrize(
    "error", [OSError("cannot start"), subprocess.CalledProcessError(1, "python")]
)
def test_failed_usearch_lookup_subprocess_means_no_extension(
    monkeypatch, mocker, error
):
    monkeypatch.setattr(base.importlib.util, "find_spec", lambda name: object())
    run = mocker.patch.object(base.subprocess, "run", side_effect=error)

    assert base._resolve_usearch_sqlite_path() is None
    run.assert_called_once()


@pytest.mark.skipif(
    not hasattr(sqlite3.Connection, "enable_load_extension"),
    reason="sqlite3 is built without extension loading",
)
def test_unloadable_usearch_extension_warns_once(
    tmp_path, fresh_lookup, monkeypatch, caplog
):
    ext_path = str(tmp_path / "usearch_sqlite")
    monkeypatch.setenv(base.USEARCH_SQLITE_ENV, ext_path)

    with (
        closing(sqlite3.connect(":memory:")) as conn,
        caplog.at_level(logging.WARNING, logger="datachain"),
    ):
        assert [base.load_usearch_extension(conn) for _ in range(3)] == [False] * 3

    assert sum(ext_path in r.getMessage() for r in caplog.records) == 1
