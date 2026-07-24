import json
import os

import pytest

import datachain as dc
from datachain.query.session import Session


def _make_tree(root, names):
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_text(f"content of {name}")


def test_read_storage_config_does_not_fork_session(tmp_dir, test_session):
    """Per-call config is registered on the catalog; the ambient session is
    neither replaced nor entered (no action at a distance)."""
    _make_tree(tmp_dir, ["a.txt"])
    depth = len(Session.SESSION_CONTEXTS)

    chain = dc.read_storage(tmp_dir.as_uri(), client_config={"use_symlinks": True})

    assert len(Session.SESSION_CONTEXTS) == depth
    assert chain.session is test_session
    # The session-wide default is untouched.
    assert "use_symlinks" not in chain.session.catalog.client_config

    bare = dc.read_storage(tmp_dir.as_uri())
    assert bare.session is test_session


def test_conflicting_configs_for_same_source_raise(tmp_dir, test_session):
    _make_tree(tmp_dir, ["a.txt"])
    dc.read_storage(tmp_dir.as_uri(), client_config={"use_symlinks": True})
    with pytest.raises(ValueError, match="different client_config"):
        dc.read_storage(tmp_dir.as_uri(), client_config={"use_symlinks": False})


def test_union_two_sources_with_different_configs(tmp_dir, test_session):
    """One chain over two sources, each with its own client config — not
    representable before per-source registration."""
    d1, d2 = tmp_dir / "src1", tmp_dir / "src2"
    _make_tree(d1, ["a.txt"])
    _make_tree(d2, ["b.txt"])

    c1 = dc.read_storage(d1.as_uri(), client_config={"use_symlinks": True})
    c2 = dc.read_storage(d2.as_uri(), client_config={"use_symlinks": False})
    chain = c1.union(c2)

    files = chain.to_values("file")
    assert len(files) == 2
    catalog = chain.session.catalog
    by_name = {f.path.rsplit("/", 1)[-1]: f for f in files}
    assert catalog.client_config_for(by_name["a.txt"].source) == {"use_symlinks": True}
    assert catalog.client_config_for(by_name["b.txt"].source) == {"use_symlinks": False}
    # Both files are readable through their per-source clients.
    assert {f.read_text() for f in files} == {"content of a.txt", "content of b.txt"}


def test_registered_config_reaches_parallel_workers(tmp_dir, test_session_tmpfile):
    """The registry ships to UDF worker processes via catalog init params:
    files materialized in workers resolve their per-source config."""
    _make_tree(tmp_dir, ["a.txt", "b.txt"])
    config = {"use_symlinks": True}
    main_pid = os.getpid()

    chain = (
        dc.read_storage(
            tmp_dir.as_uri(), session=test_session_tmpfile, client_config=config
        )
        .settings(parallel=2)
        .map(
            cfg=lambda file: json.dumps(
                {
                    "config": file._catalog.client_config_for(file.source),
                    "in_worker": os.getpid() != main_pid,
                }
            ),
            output=str,
        )
    )
    results = [json.loads(v) for v in chain.to_values("cfg")]
    assert len(results) == 2
    for result in results:
        assert result["config"] == config
        assert result["in_worker"] is True


def test_saved_dataset_resolves_registered_config(tmp_dir, test_session):
    """save() then read_dataset() in the same process: files still resolve
    the config their source was registered with (in-process part of #1778)."""
    _make_tree(tmp_dir, ["a.txt"])
    config = {"use_symlinks": True}

    dc.read_storage(tmp_dir.as_uri(), session=test_session, client_config=config).save(
        "cfg_ds"
    )

    files = dc.read_dataset("cfg_ds", session=test_session).to_values("file")
    assert len(files) == 1
    file = files[0]
    assert file._catalog.client_config_for(file.source) == config
    assert file.read_text() == "content of a.txt"
