import importlib.util

import pytest

from datachain.lib.dc.database import _default_driver


@pytest.mark.parametrize(
    "psycopg_installed, url, expected",
    [
        (False, "postgresql://u@h/db", "postgresql+psycopg2://u@h/db"),
        (True, "postgresql://u@h/db", "postgresql://u@h/db"),
        (False, "postgresql+psycopg://u@h/db", "postgresql+psycopg://u@h/db"),
        (False, "sqlite:///x.db", "sqlite:///x.db"),
    ],
)
def test_default_driver(monkeypatch, psycopg_installed, url, expected):
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if psycopg_installed else None,
    )
    assert str(_default_driver(url)) == expected
