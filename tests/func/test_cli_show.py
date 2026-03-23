import re

import pytest

import datachain as dc
from datachain.cli.commands.show import show


@pytest.fixture
def saved_dataset(test_session):
    dc.read_values(
        name=["Alice", "Bob", "Charlie"],
        age=[40, 30, 20],
        session=test_session,
    ).order_by("name").save("test-show-ds")
    return test_session.catalog


def test_show_basic(capsys, saved_dataset):
    show(saved_dataset, "test-show-ds")
    out = re.sub(r"\s+", " ", capsys.readouterr().out)
    assert "name age" in out
    assert "Alice" in out
    assert "Bob" in out
    assert "Charlie" in out


def test_show_no_name_header(capsys, saved_dataset):
    show(saved_dataset, "test-show-ds")
    out = capsys.readouterr().out
    assert "Name:" not in out
    assert "Description:" not in out


def test_show_limit(capsys, saved_dataset):
    show(saved_dataset, "test-show-ds", limit=1)
    out = capsys.readouterr().out
    assert "Alice" in out
    assert "Bob" not in out
    assert "Charlie" not in out


def test_show_offset(capsys, saved_dataset):
    show(saved_dataset, "test-show-ds", limit=10, offset=1)
    out = capsys.readouterr().out
    assert "Alice" not in out
    assert "Bob" in out


def test_show_columns(capsys, saved_dataset):
    show(saved_dataset, "test-show-ds", columns=["name"])
    out = capsys.readouterr().out
    assert "name" in out
    assert "age" not in out


def test_show_schema(capsys, saved_dataset):
    show(saved_dataset, "test-show-ds", schema=True)
    out = capsys.readouterr().out
    assert "Schema:" in out


def test_show_script(capsys, saved_dataset):
    show(saved_dataset, "test-show-ds", script=True)
    out = capsys.readouterr().out
    assert out
