import gc
import re
from weakref import ref

import pytest

import datachain as dc
from datachain.catalog import Catalog
from datachain.dataset import SESSION_DATASET_PREFIX, DatasetStatus
from datachain.error import DataChainError, DatasetNotFoundError
from datachain.query.session import Session
from tests.utils import skip_if_not_sqlite


@pytest.fixture
def project(catalog):
    return catalog.metastore.create_project("dev", "animals")


def _fqn(project, name):
    return f"{project.namespace.name}.{project.name}.{name}"


def test_ephemeral_dataset_naming(catalog, project):
    session_name = "qwer45"

    with pytest.raises(ValueError):
        Session("wrong-ds_name", catalog=catalog)

    with Session(session_name, catalog=catalog) as session:
        fqn = _fqn(project, "my_test_ds12")
        dc.read_values(name=["a"], session=session).save(fqn)
        tmp_name = session.generate_temp_dataset_name()
        ds_tmp = dc.read_dataset(fqn, session=session).save(tmp_name)
        session_uuid = f"[0-9a-fA-F]{{{Session.SESSION_UUID_LEN}}}"
        table_uuid = f"[0-9a-fA-F]{{{Session.TEMP_TABLE_UUID_LEN}}}"

        name_prefix = f"{SESSION_DATASET_PREFIX}{session_name}"
        pattern = rf"^{name_prefix}_{session_uuid}_{table_uuid}$"

        assert re.match(pattern, ds_tmp.name) is not None


def test_global_session_naming(catalog, project):
    session_uuid = f"[0-9a-fA-F]{{{Session.SESSION_UUID_LEN}}}"
    table_uuid = f"[0-9a-fA-F]{{{Session.TEMP_TABLE_UUID_LEN}}}"

    fqn = _fqn(project, "qwsd")
    global_session = Session.get(catalog=catalog)
    dc.read_values(name=["a"], session=global_session).save(fqn)
    tmp_name = global_session.generate_temp_dataset_name()
    ds_tmp = dc.read_dataset(fqn, session=global_session).save(tmp_name)
    global_prefix = f"{SESSION_DATASET_PREFIX}{Session.GLOBAL_SESSION_NAME}"
    pattern = rf"^{global_prefix}_{session_uuid}_{table_uuid}$"
    assert re.match(pattern, ds_tmp.name) is not None


def test_session_empty_name(catalog):
    with Session("", catalog=catalog) as session:
        name = session.name
    assert name.startswith(Session.GLOBAL_SESSION_NAME + "_")


@pytest.mark.parametrize(
    "name,is_temp",
    (
        ("session_global_456b5d_0cda3b", True),
        ("session_TestSession_456b5d_0cda3b", True),
        ("cats", False),
    ),
)
def test_is_temp_dataset(name, is_temp):
    assert Session.is_temp_dataset(name) is is_temp


def test_ephemeral_dataset_lifecycle(catalog, project):
    session_name = "asd3d4"
    with Session(session_name, catalog=catalog) as session:
        fqn = _fqn(project, "my_test_ds12")
        dc.read_values(name=["a"], session=session).save(fqn)
        tmp_name = session.generate_temp_dataset_name()
        ds_tmp = dc.read_dataset(fqn, session=session).save(tmp_name)

        assert ds_tmp.name != "my_test_ds12"
        assert ds_tmp.name is not None
        assert ds_tmp.name.startswith(SESSION_DATASET_PREFIX)
        assert session_name in ds_tmp.name

        ds = catalog.get_dataset(ds_tmp.name)
        assert ds is not None

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(ds_tmp.name)


def test_session_datasets_not_in_ls_datasets(catalog, project):
    session_name = "testls"
    with Session(session_name, catalog=catalog) as session:
        # Create a regular dataset
        fqn = _fqn(project, "regular_dataset")
        dc.read_values(num=[1, 2, 3], session=session).save(fqn)

        # Create a temp dataset by re-saving the regular one
        tmp_name = session.generate_temp_dataset_name()
        ds_tmp = dc.read_dataset(fqn, session=session).save(tmp_name)

        datasets = list(catalog.ls_datasets())
        dataset_names = [d.name for d in datasets]

        assert "regular_dataset" in dataset_names

        assert ds_tmp.name not in dataset_names
        assert all(not Session.is_temp_dataset(name) for name in dataset_names)


def test_cleanup_temp_datasets_all_states(catalog, project):
    session_name = "testcleanup"
    with Session(session_name, catalog=catalog) as session:
        fqn = _fqn(project, "test_dataset")
        dc.read_values(name=["a"], session=session).save(fqn)

        # Create temp datasets in different states

        # 1. CREATED state (default after save — mark it back to CREATED)
        ds_created = dc.read_dataset(fqn, session=session).save(
            session.generate_temp_dataset_name()
        )
        ds_created_record = catalog.get_dataset(ds_created.name, versions=["1.0.0"])
        catalog.metastore.update_dataset_status(
            ds_created_record, DatasetStatus.CREATED, version="1.0.0"
        )

        # 2. COMPLETE state (save already marks COMPLETE)
        ds_complete = dc.read_dataset(fqn, session=session).save(
            session.generate_temp_dataset_name()
        )

        # 3. FAILED state
        ds_failed = dc.read_dataset(fqn, session=session).save(
            session.generate_temp_dataset_name()
        )
        ds_failed_record = catalog.get_dataset(ds_failed.name, versions=["1.0.0"])
        catalog.metastore.update_dataset_status(
            ds_failed_record, DatasetStatus.FAILED, version="1.0.0"
        )

        # Verify all three exist before cleanup
        assert catalog.get_dataset(ds_created.name, include_incomplete=True)
        assert catalog.get_dataset(ds_complete.name, include_incomplete=True)
        assert catalog.get_dataset(ds_failed.name, include_incomplete=True)

    # After session exit, all temp datasets should be cleaned up
    for temp_name in [ds_created.name, ds_complete.name, ds_failed.name]:
        with pytest.raises(DatasetNotFoundError):
            catalog.get_dataset(temp_name, include_incomplete=True)


def test_get_job_returns_none_without_env_or_cached_job(catalog, monkeypatch):
    monkeypatch.delenv("DATACHAIN_JOB_ID", raising=False)
    with Session("testjob", catalog=catalog) as session:
        assert session.get_job() is None


def test_get_job_fetches_from_env_var(catalog, monkeypatch):
    from datachain.data_storage import JobQueryType, JobStatus

    job_id = catalog.metastore.create_job(
        "env-job", "", query_type=JobQueryType.PYTHON, status=JobStatus.RUNNING
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", job_id)
    with Session("testjob", catalog=catalog) as session:
        job = session.get_job()
        assert job is not None
        assert job.id == job_id
        assert Session._OWNS_JOB is False


def test_get_job_returns_cached_job(catalog, monkeypatch):
    monkeypatch.delenv("DATACHAIN_JOB_ID", raising=False)
    with Session("testjob", catalog=catalog) as session:
        first = session.get_or_create_job()
        second = session.get_job()
        assert second is first


def test_get_job_returns_none_for_missing_env_job(catalog, monkeypatch):
    monkeypatch.setenv("DATACHAIN_JOB_ID", "00000000-0000-0000-0000-000000000000")
    with Session("testjob", catalog=catalog) as session:
        assert session.get_job() is None


def test_get_or_create_job_raises_in_studio_without_env(catalog, monkeypatch):
    monkeypatch.delenv("DATACHAIN_JOB_ID", raising=False)
    monkeypatch.setenv("DATACHAIN_IS_STUDIO", "True")
    with Session("testjob", catalog=catalog) as session:
        with pytest.raises(DataChainError, match="Cannot create job in Studio"):
            session.get_or_create_job()


@pytest.mark.parametrize("catalog_view", [False, True])
def test_reenter_session_cleans_each_temporary_scope(catalog, catalog_view):
    parent = Session("reenter", catalog=catalog)
    session = parent._with_catalog(catalog.copy()) if catalog_view else parent
    for _ in range(2):
        with parent:
            chain = dc.read_values(value=[1], session=session).persist()
            temporary_name = chain.name
            assert catalog.get_dataset(temporary_name)
        with pytest.raises(DatasetNotFoundError):
            catalog.get_dataset(temporary_name)


# db.is_closed is SQLite-specific; other backends have different connection lifetimes.
@skip_if_not_sqlite
def test_session_cleanup_does_not_reopen_closed_sqlite_catalog(catalog):
    with Session("owner", catalog=catalog) as parent:
        dc.read_values(value=[1], session=parent).save("source")
        with Catalog(
            metastore=catalog.metastore.clone(),
            warehouse=catalog.warehouse.clone(),
        ) as borrowed:
            chain = dc.read_dataset("source", session=parent._with_catalog(borrowed))
            assert chain.to_values("value") == [1]
        assert borrowed.metastore.db.is_closed
    assert borrowed.metastore.db.is_closed
    assert borrowed.warehouse.db.is_closed


def test_catalog_view_keeps_cleanup_owner_alive(catalog):
    parent = Session("owner", catalog=catalog)
    view = parent._with_catalog(catalog.copy())
    owner = ref(parent)
    del parent
    gc.collect()
    assert owner() is not None
    with owner():
        chain = dc.read_values(value=[1], session=view).persist()
        temporary_name = chain.name
        assert chain.to_values("value") == [1]
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(temporary_name)


def test_catalog_view_cannot_exit_parent_scope(catalog):
    with Session("owner", catalog=catalog) as parent:
        chain = dc.read_values(value=[1], session=parent).persist()
        view = parent._with_catalog(catalog.copy())
        with pytest.raises(RuntimeError, match="parent Session"):
            with view:
                pass
        assert Session.get() is parent
        assert chain.to_values("value") == [1]


def _check_closed_catalog_cleanup(catalog, backing):
    with Session("owner", catalog=catalog) as parent:
        parent.get_or_create_job()
        with Catalog(
            metastore=backing.metastore.clone(),
            warehouse=backing.warehouse.clone(),
        ) as borrowed:
            chain = dc.read_values(value=[1], session=borrowed.session).persist()
            temporary_name = chain.name
        assert borrowed.metastore.db.is_closed
        assert borrowed.warehouse.db.is_closed
    assert borrowed.metastore.db.is_closed
    assert borrowed.warehouse.db.is_closed
    with pytest.raises(DatasetNotFoundError):
        backing.get_dataset(temporary_name)


# These tests inspect SQLite connection state after the borrowed catalog exits.
@skip_if_not_sqlite
def test_cleanup_keeps_closed_writing_catalog_closed(catalog):
    _check_closed_catalog_cleanup(catalog, catalog)


@skip_if_not_sqlite
def test_cleanup_removes_temporary_data_from_closed_separate_database(
    catalog, catalog_tmpfile
):
    _check_closed_catalog_cleanup(catalog, catalog_tmpfile)
