import importlib

import pytest

from datachain.lib.dc.database import _default_driver


@pytest.mark.parametrize(
    "psycopg_importable, url, expected",
    [
        (False, "postgresql://u@h/db", "postgresql+psycopg2://u@h/db"),
        (True, "postgresql://u@h/db", "postgresql://u@h/db"),
        (False, "postgresql+psycopg://u@h/db", "postgresql+psycopg://u@h/db"),
        (False, "sqlite:///x.db", "sqlite:///x.db"),
    ],
)
def test_default_driver(monkeypatch, psycopg_importable, url, expected):
    def import_module(name):
        if psycopg_importable:
            return object()
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", import_module)
    assert str(_default_driver(url)) == expected
