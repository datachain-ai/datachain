import re

import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.dataset import DatasetStatus
from datachain.error import DatasetNotFoundError
from datachain.query.dataset import DatasetQuery
from datachain.query.session import Session
from datachain.sql.types import String


@pytest.fixture
def project(catalog):
    return catalog.metastore.create_project("dev", "animals")


def test_ephemeral_dataset_naming(catalog, project):
    session_name = "qwer45"

    with pytest.raises(ValueError):
        Session("wrong-ds_name", catalog=catalog)

    with Session(session_name, catalog=catalog) as session:
        ds_name = "my_test_ds12"
        session.catalog.create_dataset(
            ds_name, project, columns=(sa.Column("name", String),)
        )
        ds_tmp = DatasetQuery(
            name=ds_name,
            namespace_name=project.namespace.name,
            project_name=project.name,
            session=session,
            catalog=session.catalog,
            include_incomplete=True,  # Test works with CREATED dataset
        ).save()
        session_uuid = f"[0-9a-fA-F]{{{Session.SESSION_UUID_LEN}}}"
        table_uuid = f"[0-9a-fA-F]{{{Session.TEMP_TABLE_UUID_LEN}}}"

        name_prefix = f"{Session.DATASET_PREFIX}{session_name}"
        pattern = rf"^{name_prefix}_{session_uuid}_{table_uuid}$"

        assert re.match(pattern, ds_tmp.name) is not None


def test_global_session_naming(catalog, project):
    session_uuid = f"[0-9a-fA-F]{{{Session.SESSION_UUID_LEN}}}"
    table_uuid = f"[0-9a-fA-F]{{{Session.TEMP_TABLE_UUID_LEN}}}"

    ds_name = "qwsd"
    catalog.create_dataset(ds_name, project, columns=(sa.Column("name", String),))
    ds_tmp = DatasetQuery(
        name=ds_name,
        namespace_name=project.namespace.name,
        project_name=project.name,
        catalog=catalog,
        include_incomplete=True,  # Test works with CREATED dataset
    ).save()
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
        ds_name = "my_test_ds12"
        session.catalog.create_dataset(
            ds_name, project, columns=(sa.Column("name", String),)
        )
        ds_tmp = DatasetQuery(
            name=ds_name,
            namespace_name=project.namespace.name,
            project_name=project.name,
            session=session,
            catalog=session.catalog,
            include_incomplete=True,  # Test works with CREATED dataset
        ).save()

        assert isinstance(ds_tmp, DatasetQuery)
        assert ds_tmp.name != ds_name
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
        ds_name = "regular_dataset"
        (
            dc.read_values(num=[1, 2, 3], session=session)
            .settings(namespace=project.namespace.name, project=project.name)
            .save(ds_name)
        )

        # Create a temp dataset
        ds_tmp = DatasetQuery(
            name=ds_name,
            namespace_name=project.namespace.name,
            project_name=project.name,
            session=session,
            catalog=session.catalog,
            include_incomplete=True,
        ).save()

        datasets = list(catalog.ls_datasets())
        dataset_names = [d.name for d in datasets]

        assert ds_name in dataset_names

        assert ds_tmp.name not in dataset_names
        assert all(not Session.is_temp_dataset(name) for name in dataset_names)


def test_cleanup_temp_datasets_all_states(catalog, project):
    session_name = "testcleanup"
    with Session(session_name, catalog=catalog) as session:
        ds_name = "test_dataset"
        session.catalog.create_dataset(
            ds_name, project, columns=(sa.Column("name", String),)
        )

        # Create temp datasets in different states

        # 1. CREATED state
        ds_created = DatasetQuery(
            name=ds_name,
            namespace_name=project.namespace.name,
            project_name=project.name,
            session=session,
            catalog=session.catalog,
            include_incomplete=True,
        ).save()

        # 2. COMPLETE state
        ds_complete = DatasetQuery(
            name=ds_name,
            namespace_name=project.namespace.name,
            project_name=project.name,
            session=session,
            catalog=session.catalog,
            include_incomplete=True,
        ).save()
        ds_complete_record = catalog.get_dataset(
            ds_complete.name, include_incomplete=True
        )
        catalog.metastore.update_dataset_status(
            ds_complete_record, DatasetStatus.COMPLETE, version="1.0.0"
        )

        # 3. FAILED state
        ds_failed = DatasetQuery(
            name=ds_name,
            namespace_name=project.namespace.name,
            project_name=project.name,
            session=session,
            catalog=session.catalog,
            include_incomplete=True,
        ).save()
        ds_failed_record = catalog.get_dataset(ds_failed.name, include_incomplete=True)
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
