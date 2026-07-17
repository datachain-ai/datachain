import re

import pytest

import datachain as dc
from datachain.dataset import SESSION_DATASET_PREFIX, DatasetStatus
from datachain.error import DataChainError, DatasetNotFoundError
from datachain.query.session import Session


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


def test_get_in_memory_session_with_persistent_session_present(catalog):
    global_session = Session.get(catalog=catalog)
    assert not global_session.catalog.in_memory

    session = Session.get(in_memory=True)
    assert session is not global_session
    assert session.catalog.in_memory

    # A single in-memory session is cached and reused
    assert Session.get(in_memory=True) is session

    # Default resolution is unaffected
    assert Session.get() is global_session

    # client_config is inherited from the session it shadows
    assert session.catalog.client_config == global_session.catalog.client_config


def test_in_memory_conflicts_with_explicit_persistent_session(catalog):
    with Session("conflict1", catalog=catalog) as session:
        with pytest.raises(ValueError, match="in_memory"):
            dc.read_values(num=[1], session=session, in_memory=True)

    with pytest.raises(ValueError, match="in_memory"):
        Session.get(catalog=catalog, in_memory=True)


def test_in_memory_with_explicit_in_memory_session():
    with Session("mem1", in_memory=True) as session:
        chain = dc.read_values(num=[1], session=session, in_memory=True)
        assert chain.to_values("num") == [1]


def test_first_in_memory_call_becomes_process_default():
    Session.cleanup_for_tests()
    try:
        session = Session.get(in_memory=True)
        assert session.catalog.in_memory
        assert Session.GLOBAL_SESSION_CTX is session
        assert Session.get() is session  # legacy: unflagged calls share it
    finally:
        Session.cleanup_for_tests()


def test_first_in_memory_call_in_studio_does_not_become_default(monkeypatch):
    monkeypatch.setenv("DATACHAIN_IS_STUDIO", "True")
    Session.cleanup_for_tests()
    try:
        session = Session.get(in_memory=True)
        assert session.catalog.in_memory
        assert Session.GLOBAL_SESSION_CTX is None
        assert Session.IN_MEMORY_SESSION_CTX is session
    finally:
        Session.cleanup_for_tests()


def test_in_memory_sessions_cached_by_client_config(catalog):
    Session.get(catalog=catalog)  # global, client_config == {}

    default = Session.get(in_memory=True)  # inherits {} from global
    assert default.catalog.client_config == {}
    assert Session.get(in_memory=True) is default

    anon = Session.get(in_memory=True, client_config={"anon": True})
    assert anon is not default
    assert anon.catalog.client_config == {"anon": True}
    assert Session.get(in_memory=True, client_config={"anon": True}) is anon

    # Config inherited from the ambient session selects the same session
    # as passing it explicitly
    with Session("ambient1", client_config={"anon": True}):
        assert Session.get(in_memory=True) is anon

    # Implicit resolution must not leak contexts or change the default
    assert not Session.SESSION_CONTEXTS
    assert Session.get() is Session.GLOBAL_SESSION_CTX


def test_in_memory_session_cleanup_for_tests(catalog):
    Session.get(catalog=catalog)
    session = Session.get(in_memory=True)
    assert Session.IN_MEMORY_SESSION_CTX is session

    Session.cleanup_for_tests()
    assert Session.IN_MEMORY_SESSION_CTX is None


def test_in_memory_session_job_isolated_from_env(catalog, monkeypatch):
    from datachain.data_storage import JobQueryType, JobStatus

    # A job exists in the persistent metastore and is advertised via env,
    # as inside a Studio job run.
    job_id = catalog.metastore.create_job(
        "env-job", "", query_type=JobQueryType.PYTHON, status=JobStatus.RUNNING
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", job_id)

    Session.get(catalog=catalog)
    session = Session.get(in_memory=True)

    # The env job must not leak into the throwaway catalog
    assert session.get_job() is None

    job = session.get_or_create_job()
    assert job is not None
    assert job.id != job_id
    assert session.get_or_create_job() is job
    assert session.get_job() is job

    # ...and the session-local job must not leak into the process-wide cache
    assert Session._CURRENT_JOB is None


def test_read_storage_in_memory_keeps_listing_out_of_catalog(
    catalog_tmpfile, tmp_path, monkeypatch
):
    # catalog_tmpfile: a file-backed catalog is required here — an in-memory
    # test catalog would share the process-wide SQLite shared-cache database
    # with the throwaway catalog, defeating the isolation this test checks.
    monkeypatch.delenv("DATACHAIN_JOB_ID", raising=False)
    data_dir = tmp_path / "data"  # tmp_path itself holds the catalog's db file
    data_dir.mkdir()
    (data_dir / "a.txt").write_text("a")
    (data_dir / "b.txt").write_text("b")

    global_session = Session.get(catalog=catalog_tmpfile)

    chain = dc.read_storage(data_dir.as_uri(), in_memory=True)
    assert chain.count() == 2
    assert chain.session.catalog.in_memory

    # The listing dataset lives only in the throwaway catalog
    assert list(global_session.catalog.listings()) == []
    assert len(list(chain.session.catalog.listings())) == 1

    # Datasets saved through the chain stay in the throwaway catalog too.
    # Check the persistent catalog directly — an unflagged read_dataset()
    # would go through Studio remote fallback in environments with a
    # non-local default namespace.
    chain.save("mem_only_ds")
    assert dc.read_dataset("mem_only_ds", in_memory=True).count() == 2
    with pytest.raises(DatasetNotFoundError):
        global_session.catalog.get_dataset("mem_only_ds")


def test_in_memory_save_read_roundtrip_in_studio_job_env(catalog_tmpfile, monkeypatch):
    from datachain.data_storage import JobQueryType, JobStatus

    # Simulate a Studio job: is_studio() is true, the job lives in the
    # persistent metastore and is advertised via env, and the job's project
    # is routed via DATACHAIN_PROJECT (applies to any session — env is
    # process-global).
    job_id = catalog_tmpfile.metastore.create_job(
        "studio-job", "", query_type=JobQueryType.PYTHON, status=JobStatus.RUNNING
    )
    monkeypatch.setenv("DATACHAIN_IS_STUDIO", "True")
    monkeypatch.setenv("DATACHAIN_JOB_ID", job_id)
    monkeypatch.setenv("DATACHAIN_PROJECT", "dev.analytics")

    Session.get(catalog=catalog_tmpfile)

    # Save resolves the env project and auto-creates it (is_studio() ⇒
    # create=True) in the throwaway metastore; the session-local job is
    # used, not the env job.
    dc.read_values(num=[1, 2, 3], in_memory=True).save("gate")

    # Symmetric read: same env resolution, same throwaway catalog.
    assert dc.read_dataset("gate", in_memory=True).count() == 3
    assert dc.read_dataset("dev.analytics.gate", in_memory=True).count() == 3

    in_memory_catalog = Session.IN_MEMORY_SESSION_CTX.catalog
    ds = in_memory_catalog.get_dataset(
        "gate", namespace_name="dev", project_name="analytics", versions=None
    )
    assert ds.versions[-1].job_id != job_id  # session-local job attributed

    # Nothing leaked into the persistent catalog.
    with pytest.raises(DatasetNotFoundError):
        catalog_tmpfile.get_dataset(
            "gate", namespace_name="dev", project_name="analytics"
        )

    # And in Studio there is no remote fallback: an unknown dataset in the
    # in-memory catalog fails cleanly.
    with pytest.raises(DatasetNotFoundError):
        dc.read_dataset("missing_ds", in_memory=True)


def test_combining_in_memory_and_persistent_chains_raises(catalog_tmpfile, monkeypatch):
    monkeypatch.delenv("DATACHAIN_JOB_ID", raising=False)
    Session.get(catalog=catalog_tmpfile)

    mem = dc.read_values(num=[1, 2], in_memory=True)
    persistent = dc.read_values(num=[3, 4])

    with pytest.raises(ValueError, match="in-memory"):
        mem.union(persistent)
    with pytest.raises(ValueError, match="in-memory"):
        persistent.union(mem)
    with pytest.raises(ValueError, match="in-memory"):
        persistent.merge(mem, on="num")
    with pytest.raises(ValueError, match="in-memory"):
        persistent.subtract(mem, on="num")
