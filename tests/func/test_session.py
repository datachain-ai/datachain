import gc
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from weakref import ref

import pytest

import datachain as dc
from datachain.catalog import Catalog
from datachain.data_storage import JobStatus
from datachain.query.session import Session
from tests.utils import (
    e2e_subprocess_env,
    run_test_subprocess,
    wait_for_test_subprocess,
)


def test_listing_dataset_lifecycle(tmp_path, catalog):
    p = tmp_path / "hello.txt"
    p.write_text("hello", encoding="utf-8")

    session_name = "asd3d5"
    ds_name = "my_test_ds13"

    with pytest.raises(ValueError):
        with Session(session_name, catalog=catalog):
            ds_name = "my_test_ds13"
            dc.read_storage(str(tmp_path)).exec()
            dc.read_values(key=["a", "b", "c"]).save(ds_name)
            raise ValueError("This is a test exception")

    # Datasets persist even if session fails with exception
    dataset = catalog.get_dataset(ds_name)
    assert dataset is not None

    assert dc.listings(catalog=catalog).count() == 1

    # Cleanup for test
    catalog.remove_dataset(ds_name, force=True)


def test_catalog_inside_another_session_keeps_its_database_and_cleanup(
    catalog, tmp_path, metastore_tmpfile
):
    folder = tmp_path / "input"
    folder.mkdir()
    source = folder / "data.txt"
    source.write_text("contents")
    other = catalog.copy()
    other.metastore = metastore_tmpfile
    for owner in ["first", "second"]:
        with Session(owner, catalog=other) as parent:
            assert parent.catalog.metastore is not catalog.metastore
            kept = dc.read_values(value=[1], session=catalog.session).persist()
            rows = [
                row
                for _, result in catalog.ls(
                    [str(folder) + "/"], fields=["name"], update=True
                )
                for row in result
            ]
            assert rows == [("data.txt",)]
            assert kept.to_values("value") == [1]
            assert catalog.session.catalog is catalog
            assert catalog.session.name == parent.name
            assert list(
                catalog.metastore.list_datasets_by_prefix(parent.get_temp_prefix())
            )
        assert not list(
            catalog.metastore.list_datasets_by_prefix(parent.get_temp_prefix())
        )
    assert list(catalog.listings())


def test_global_listing_releases_catalogs_and_temporary_datasets(catalog, tmp_path):
    # The process-wide job hook already retains its original session/catalog.
    Session.get(catalog=catalog).get_or_create_job()
    catalogs = []
    for i in range(5):
        folder = tmp_path / str(i)
        folder.mkdir()
        (folder / "file.txt").write_text("contents")
        with Catalog(
            metastore=catalog.metastore.clone(),
            warehouse=catalog.warehouse.clone(),
        ) as other:
            rows = [
                row
                for _, result in other.ls([str(folder) + "/"], fields=["name"])
                for row in result
            ]
            assert rows == [("file.txt",)]
            assert not list(other.metastore.list_datasets_by_prefix("session_"))
            catalogs.append(ref(other))
        del other
    gc.collect()
    assert all(c() is None for c in catalogs)
    assert len(list(catalog.listings())) == 5


def test_failed_listing_cleans_seed_and_preserves_parent(catalog, tmp_path, mocker):
    from datachain.lib import listing

    (tmp_path / "file.txt").write_text("contents")
    with Session("owner", catalog=catalog) as parent:
        kept = dc.read_values(value=[1], session=parent).persist()
        mocker.patch.object(
            listing, "list_pages", side_effect=RuntimeError("scan failed")
        )
        with pytest.raises(RuntimeError, match="scan failed"):
            list(catalog.ls([str(tmp_path) + "/"], fields=["name"]))
        assert Session.get() is parent
        assert kept.to_values("value") == [1]
        temporary = list(catalog.metastore.list_datasets_by_prefix("session_"))
        assert all(d.name.startswith(parent.get_temp_prefix()) for d in temporary)
    assert not list(catalog.metastore.list_datasets_by_prefix("session_"))


def test_listing_in_another_thread_preserves_active_session(
    catalog_tmpfile, tmp_path, monkeypatch
):
    from datachain.lib import listing as listing_lib

    catalog = catalog_tmpfile
    folder = tmp_path / "input"
    folder.mkdir()
    (folder / "file.txt").write_text("contents")
    entered = Event()
    resume = Event()
    list_pages = listing_lib.list_pages

    def paused_listing(*args, **kwargs):
        entered.set()
        assert resume.wait(15)
        return list_pages(*args, **kwargs)

    def list_in_worker():
        with Catalog(
            metastore=catalog.metastore.clone(use_new_connection=True),
            warehouse=catalog.warehouse.clone(use_new_connection=True),
        ) as worker:
            return [
                row
                for _, result in worker.ls([str(folder) + "/"], fields=["name"])
                for row in result
            ]

    with Session("owner", catalog=catalog) as parent:
        parent.get_or_create_job()
        monkeypatch.setattr(listing_lib, "list_pages", paused_listing)
        with ThreadPoolExecutor(max_workers=1) as pool:
            listing = pool.submit(list_in_worker)
            try:
                if not entered.wait(10):
                    listing.result(timeout=1)
                    pytest.fail("listing did not start")
                assert Session.get() is parent
                kept = dc.read_values(value=[1]).persist()
            finally:
                resume.set()
            assert listing.result(timeout=30) == [("file.txt",)]
        assert Session.get() is parent
        assert kept.to_values("value") == [1]


SCRIPT_WITH_IN_MEMORY_SESSIONS = """\
import datachain as dc
with dc.Session("first", in_memory=True):
    dc.read_values(x=[1]).save("in_memory")
dc.read_values(x=[1]).save("persistent")
with dc.Session("last", in_memory=True):
    dc.read_values(x=[1]).save("in_memory")
    {end}
"""


@pytest.mark.parametrize(
    ("end", "status"),
    [("pass", JobStatus.COMPLETE), ("raise KeyboardInterrupt", JobStatus.CANCELED)],
    ids=["normal", "ctrl-c"],
)
def test_script_exit_after_in_memory_sessions_finalizes_its_job_cleanly(
    tmp_path, catalog_tmpfile, end, status
):
    script = tmp_path / "script.py"
    script.write_text(SCRIPT_WITH_IN_MEMORY_SESSIONS.format(end=end))
    env = e2e_subprocess_env(catalog_tmpfile)
    env.pop("DATACHAIN_JOB_ID", None)

    process = run_test_subprocess([sys.executable, str(script)], env)
    _, _, stderr = wait_for_test_subprocess(process)

    assert "Exception ignored" not in stderr
    job = catalog_tmpfile.metastore.get_last_job_by_name(str(script))
    assert job is not None, stderr
    assert job.status == status, stderr
