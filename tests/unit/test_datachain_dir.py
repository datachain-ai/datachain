from pathlib import Path

import pytest

import datachain as dc
from datachain.query.session import Session
from datachain.utils import DataChainDir


@pytest.mark.parametrize("location", [".", "nested", "nested/deeper"])
def test_find_nearest_parent_database(tmp_path, monkeypatch, location):
    parent = tmp_path / ".datachain"
    parent.mkdir()
    (parent / "db").touch()
    project = tmp_path / "project"
    root = project / ".datachain"
    root.mkdir(parents=True)
    (root / "db").touch()
    cwd = project / location
    cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(cwd)

    assert DataChainDir().root == str(root)
    assert DataChainDir.find(create=False).db == str(root / "db")
    assert not (cwd / ".datachain").exists() or cwd == project


@pytest.mark.parametrize("marker", [None, "config", "db_directory"])
def test_no_parent_database_uses_cwd(tmp_path, monkeypatch, marker):
    root = tmp_path / ".datachain"
    root.mkdir()
    if marker == "config":
        (root / "config").touch()
    elif marker == "db_directory":
        (root / "db").mkdir()
    cwd = tmp_path / "nested"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    assert DataChainDir().root == str(cwd / ".datachain")
    assert DataChainDir.find().root == str(cwd / ".datachain")
    assert (cwd / ".datachain").is_dir()


def test_root_env_overrides_parent_database(tmp_path, monkeypatch):
    root = tmp_path / ".datachain"
    root.mkdir()
    (root / "db").touch()
    monkeypatch.chdir(tmp_path)
    override = tmp_path / "override"
    monkeypatch.setenv("DATACHAIN_ROOT_DIR", str(override))

    assert DataChainDir.find().root == str(override / ".datachain")


def test_explicit_directory_overrides_parent_database(tmp_path, monkeypatch):
    root = tmp_path / ".datachain"
    root.mkdir()
    (root / "db").touch()
    monkeypatch.chdir(tmp_path)
    override = tmp_path / "override"
    monkeypatch.setenv("DATACHAIN_DIR", str(override))

    assert DataChainDir.find().root == str(override)
    assert DataChainDir(root=str(tmp_path / "explicit")).root == str(
        tmp_path / "explicit"
    )


def test_read_saved_dataset_from_nested_directory(tmp_path, monkeypatch, sqlite_db):
    monkeypatch.chdir(tmp_path)
    with Session() as session:
        dc.read_values(x=[1, 2], session=session).save("parent_dataset")

    nested = tmp_path / "nested" / "deeper"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    with Session() as session:
        assert list(
            dc.read_dataset("parent_dataset", session=session).to_values("x")
        ) == [
            1,
            2,
        ]
        assert Path(session.catalog.metastore.db.db_file) == tmp_path / ".datachain/db"
    assert not (nested / ".datachain").exists()
