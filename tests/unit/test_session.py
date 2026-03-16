import re

import pytest

import datachain as dc
from datachain.dataset import DatasetStatus
from datachain.error import DatasetNotFoundError
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

        name_prefix = f"{Session.DATASET_PREFIX}{session_name}"
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
    global_prefix = f"{Session.DATASET_PREFIX}{Session.GLOBAL_SESSION_NAME}"
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
        assert ds_tmp.name.startswith(Session.DATASET_PREFIX)
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
        ds_created_record = catalog.get_dataset(ds_created.name)
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
        ds_failed_record = catalog.get_dataset(ds_failed.name)
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
