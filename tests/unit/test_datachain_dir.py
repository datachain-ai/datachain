import pytest

import datachain as dc
from datachain.query.session import Session
from datachain.utils import DataChainDir
from tests.utils import skip_if_not_sqlite


def make_state(directory, marker="db"):
    root = directory / ".datachain"
    root.mkdir(parents=True, exist_ok=True)
    (root / marker).touch()
    return root


def make_git(directory, kind="directory"):
    marker = directory / ".git"
    if kind == "file":
        marker.write_text("gitdir: ../worktrees/project\n")
    else:
        marker.mkdir()


@pytest.mark.parametrize("location", [".", "nested", "nested/deeper"])
@pytest.mark.parametrize("marker", ["db", "config"])
def test_find_nearest_project_state(tmp_path, monkeypatch, location, marker):
    (tmp_path / ".git").mkdir()
    make_state(tmp_path)
    project = tmp_path / "project"
    root = make_state(project, marker)
    cwd = project / location
    cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(cwd)

    assert DataChainDir.find(create=False).root == str(root)


@pytest.mark.parametrize("git_marker", ["directory", "file"])
@pytest.mark.parametrize("existing_state", [False, True])
def test_git_root_bounds_project_discovery(
    tmp_path, monkeypatch, git_marker, existing_state
):
    make_state(tmp_path)
    repo = tmp_path / "repo"
    nested = repo / "scripts" / "nested"
    nested.mkdir(parents=True)
    make_git(repo, git_marker)
    if existing_state:
        make_state(repo)
    monkeypatch.chdir(nested)

    assert DataChainDir.find().root == str(repo / ".datachain")
    assert (repo / ".datachain").is_dir()
    assert not (nested / ".datachain").exists()


@pytest.mark.parametrize("marker", ["db", "config"])
def test_outside_git_uses_cwd(tmp_path, monkeypatch, marker):
    make_state(tmp_path, marker)
    cwd = tmp_path / "nested"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    assert DataChainDir.find().root == str(cwd / ".datachain")
    assert (cwd / ".datachain").is_dir()


@pytest.mark.parametrize("marker", [None, "cache", "tmp", "db", "config"])
def test_skip_state_without_db_or_config_file(tmp_path, monkeypatch, marker):
    (tmp_path / ".git").mkdir()
    root = make_state(tmp_path)
    child_state = tmp_path / "child" / ".datachain"
    child_state.mkdir(parents=True)
    if marker:
        (child_state / marker).mkdir()
    nested = child_state.parent / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)

    assert DataChainDir.find().root == str(root)


@pytest.mark.parametrize("git_marker", ["directory", "file"])
def test_nested_git_project_does_not_reuse_outer_state(
    tmp_path, monkeypatch, git_marker
):
    (tmp_path / ".git").mkdir()
    make_state(tmp_path)
    inner = tmp_path / "inner"
    nested = inner / "scripts"
    nested.mkdir(parents=True)
    make_git(inner, git_marker)
    monkeypatch.chdir(nested)

    assert DataChainDir.find().root == str(inner / ".datachain")


@pytest.mark.parametrize(
    "env_var,expected",
    [("DATACHAIN_ROOT_DIR", "override/.datachain"), ("DATACHAIN_DIR", "override")],
)
def test_env_var_overrides_project_state(tmp_path, monkeypatch, env_var, expected):
    (tmp_path / ".git").mkdir()
    make_state(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(env_var, str(tmp_path / "override"))

    assert DataChainDir.find().root == str(tmp_path / expected)


def test_explicit_directory_and_env_precedence(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    make_state(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATACHAIN_ROOT_DIR", str(tmp_path / "root-override"))
    monkeypatch.setenv("DATACHAIN_DIR", str(tmp_path / "dir-override"))

    assert DataChainDir.find().root == str(tmp_path / "dir-override")
    assert DataChainDir(root=str(tmp_path / "explicit")).root == str(
        tmp_path / "explicit"
    )


@skip_if_not_sqlite
def test_read_saved_dataset_from_nested_directory(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    with Session() as session:
        dc.read_values(x=[1, 2], session=session).save("parent_dataset")

    nested = tmp_path / "nested" / "deeper"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    with Session() as session:
        chain = dc.read_dataset("parent_dataset", session=session)
        assert chain.to_values("x") == [1, 2]
    assert not (nested / ".datachain").exists()


@skip_if_not_sqlite
@pytest.mark.parametrize("env_var", ["DATACHAIN_DIR", "DATACHAIN_ROOT_DIR"])
def test_session_cache_and_tmp_follow_state_directory(tmp_path, monkeypatch, env_var):
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    destination = tmp_path / "custom"
    monkeypatch.setenv(env_var, str(destination))
    root = destination if env_var == "DATACHAIN_DIR" else destination / ".datachain"

    with Session() as session:
        dc.read_values(x=[1, 2], session=session).save("saved")
        assert dc.read_dataset("saved", session=session).to_values("x") == [1, 2]
        assert session.catalog.cache.cache_dir == str(root / "cache")
        assert session.catalog.cache.tmp_dir == str(root / "tmp")
    assert not (cwd / ".datachain").exists()
