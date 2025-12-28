import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from datachain.data_storage import JobQueryType, JobStatus
from datachain.dataset import DatasetStatus
from datachain.sql import types as dc_types

feature_schema = {
    "file": "File@v1",
    "_custom_types": {
        "File@v1": {
            "source": "str",
            "path": "str",
            "size": "int",
            "version": "str",
            "etag": "str",
            "is_latest": "bool",
            "last_modified": "datetime",
            "location": "Union",
            "vtype": "str",
        }
    },
}

query_script = """
    import datachain as dc
    dc.read_storage("gs://test_source").save("test-source")
    """

schema = {
    "sys__id": {"type": "UInt64"},
    "sys__rand": {"type": "UInt64"},
    "file__source": {"type": "String"},
    "file__path": {"type": "String"},
    "file__size": {"type": "Int64"},
    "file__version": {"type": "String"},
    "file__etag": {"type": "String"},
    "file__is_latest": {"type": "Boolean"},
    "file__last_modified": {"type": "DateTime"},
    "file__location": {"type": "JSON"},
    "file__vtype": {"type": "String"},
}
expected_schema = {
    "sys__id": dc_types.UInt64,
    "sys__rand": dc_types.UInt64,
    "file__source": dc_types.String,
    "file__path": dc_types.String,
    "file__size": dc_types.Int64,
    "file__version": dc_types.String,
    "file__etag": dc_types.String,
    "file__is_latest": dc_types.Boolean,
    "file__last_modified": dc_types.DateTime,
    "file__location": dc_types.JSON,
    "file__vtype": dc_types.String,
}

preview = [
    {"sys__id": 1, "signal_name": "foo"},
    {"sys__id": 2, "signal_name": "bar"},
    {"sys__id": 3, "signal_name": "baz"},
]
preview_json = json.dumps(preview, separators=(",", ":"))


def test_create_dataset(metastore):
    ds = metastore.create_dataset(
        name="test_dataset",
        status=DatasetStatus.COMPLETE,
        sources=["gs://test_source"],
        feature_schema=feature_schema,
        query_script=query_script,
        schema=schema,
        description="test dataset",
        attrs=["test", "dataset"],
    )
    assert ds.id is not None
    assert ds.name == "test_dataset"
    assert ds.status == DatasetStatus.COMPLETE
    assert ds.sources == "gs://test_source"
    assert ds.feature_schema == feature_schema
    assert ds.query_script == query_script
    assert ds.schema == expected_schema
    assert ds.description == "test dataset"
    assert ds.attrs == ["test", "dataset"]
    assert ds.created_at is not None


@pytest.mark.parametrize("ignore_if_exists", [True, False])
def test_create_dataset_exists(metastore, ignore_if_exists):
    ds1 = metastore.create_dataset(
        name="test_dataset",
        ignore_if_exists=ignore_if_exists,
    )
    assert ds1.id is not None

    if ignore_if_exists:
        ds2 = metastore.create_dataset(
            name="test_dataset",
            ignore_if_exists=ignore_if_exists,
        )
        assert ds2.id is not None
        assert ds2.id == ds1.id
    else:
        with pytest.raises(Exception, match="constraint"):
            metastore.create_dataset(
                name="test_dataset",
                ignore_if_exists=ignore_if_exists,
            )


def test_create_dataset_version(metastore):
    ds = metastore.create_dataset(
        name="test_dataset",
        feature_schema=feature_schema,
        query_script=query_script,
        schema=schema,
    )
    assert ds.id is not None

    created_at = datetime.now(timezone.utc) - timedelta(minutes=3)
    finished_at = datetime.now(timezone.utc)
    job_id = str(uuid4())
    uuid = str(uuid4())

    ds = metastore.create_dataset_version(
        dataset=ds,
        version="1.2.3",
        status=DatasetStatus.COMPLETE,
        sources="gs://test_source",
        feature_schema=feature_schema,
        query_script=query_script,
        error_message="Error message",
        error_stack="Error stack",
        script_output="Script output",
        created_at=created_at.isoformat(),
        finished_at=finished_at.isoformat(),
        schema=schema,
        num_objects=100,
        size=1000,
        preview=preview,
        job_id=job_id,
        uuid=uuid,
    )
    assert ds.id is not None
    assert len(ds.versions) == 1
    assert ds.latest_version == "1.2.3"

    dv = ds.versions[0]
    assert dv.id is not None
    assert dv.uuid == uuid
    assert dv.dataset_id == ds.id
    assert dv.version == "1.2.3"
    assert dv.status == DatasetStatus.COMPLETE
    assert dv.feature_schema == feature_schema
    assert dv.created_at == created_at
    assert dv.finished_at == finished_at
    assert dv.error_message == "Error message"
    assert dv.error_stack == "Error stack"
    assert dv.script_output == "Script output"
    assert dv.schema == expected_schema
    assert dv.num_objects == 100
    assert dv.size == 1000
    assert dv._preview_data in (preview, preview_json)
    assert dv.sources == "gs://test_source"
    assert dv.query_script == query_script
    assert dv.job_id == job_id


def test_create_dataset_version_finished_at(metastore):
    now = datetime.now(timezone.utc)
    ds = metastore.create_dataset(name="test_dataset")

    ds = metastore.create_dataset_version(
        dataset=ds,
        version="1.2.3",
        status=DatasetStatus.CREATED,
        finished_at=now.isoformat(),
    )
    assert len(ds.versions) == 1
    assert ds.versions[0].finished_at is None

    ds = metastore.create_dataset_version(
        dataset=ds,
        version="1.2.4",
        status=DatasetStatus.COMPLETE,
        finished_at=now.isoformat(),
    )
    assert len(ds.versions) == 2
    assert ds.versions[1].finished_at == now

    ds = metastore.create_dataset_version(
        dataset=ds, version="1.2.5", status=DatasetStatus.FAILED
    )
    assert len(ds.versions) == 3
    assert ds.versions[2].finished_at is not None


@pytest.mark.parametrize("ignore_if_exists", [True, False])
def test_create_dataset_version_exists(metastore, ignore_if_exists):
    ds = metastore.create_dataset(name="test_dataset")

    dv1 = metastore.create_dataset_version(
        dataset=ds, version="1.2.3", status=DatasetStatus.CREATED
    )
    assert len(dv1.versions) == 1

    if ignore_if_exists:
        dv2 = metastore.create_dataset_version(
            dataset=ds,
            version="1.2.3",
            status=DatasetStatus.COMPLETE,
            ignore_if_exists=ignore_if_exists,
        )
        assert dv2.id is not None
        assert dv2.id == dv1.id
        assert len(dv2.versions) == 1
        assert dv2.versions[0].id == dv1.versions[0].id
        assert dv2.versions[0].status == DatasetStatus.CREATED
    else:
        with pytest.raises(Exception, match="constraint"):
            metastore.create_dataset_version(
                dataset=ds,
                version="1.2.3",
                status=DatasetStatus.COMPLETE,
                ignore_if_exists=ignore_if_exists,
            )


def test_remove_dataset(metastore):
    ds = metastore.create_dataset(name="test_dataset")
    ds = metastore.create_dataset_version(
        dataset=ds, version="1.2.3", status=DatasetStatus.COMPLETE
    )

    ds_src = metastore.create_dataset(name="dataset_source")
    ds_src = metastore.create_dataset_version(
        dataset=ds_src, version="1.2.3", status=DatasetStatus.COMPLETE
    )
    metastore.add_dataset_dependency(
        ds,
        ds.latest_version,
        ds_src,
        ds_src.latest_version,
    )

    ds_deps1 = metastore.get_direct_dataset_dependencies(ds, ds.latest_version)
    assert len(ds_deps1) == 1
    assert ds_deps1[0].name == ds_src.name
    assert ds_deps1[0].version == str(ds_src.latest_version)

    ds_dep = metastore.create_dataset(name="dataset_dependant")
    ds_dep = metastore.create_dataset_version(
        dataset=ds_dep, version="1.2.3", status=DatasetStatus.COMPLETE
    )
    metastore.add_dataset_dependency(
        ds_dep,
        ds_dep.latest_version,
        ds,
        ds.latest_version,
    )

    ds_deps2 = metastore.get_direct_dataset_dependencies(ds_dep, ds_dep.latest_version)
    assert len(ds_deps2) == 1
    assert ds_deps2[0].name == ds.name
    assert ds_deps2[0].version == str(ds.latest_version)

    # query to check dependencies
    deps_query = metastore._datasets_dependencies_select(
        metastore._datasets_dependencies.c.source_dataset_id,
        metastore._datasets_dependencies.c.source_dataset_version_id,
        metastore._datasets_dependencies.c.dataset_id,
        metastore._datasets_dependencies.c.dataset_version_id,
    ).where(metastore._datasets_dependencies.c.id.in_((ds_deps1[0].id, ds_deps2[0].id)))

    # check if all dependencies are in the database
    assert set(metastore.db.execute(deps_query)) == {
        (ds.id, ds.versions[0].id, ds_src.id, ds_src.versions[0].id),
        (ds_dep.id, ds_dep.versions[0].id, ds.id, ds.versions[0].id),
    }

    metastore.remove_dataset(ds)
    with pytest.raises(Exception, match=r"Dataset .+ not found"):
        metastore.get_dataset(ds.name)

    # dependencies should also be deleted and cleaned up
    assert set(metastore.db.execute(deps_query)) == {
        (ds_dep.id, ds_dep.versions[0].id, None, None),
    }


def test_remove_dataset_not_found(metastore):
    ds = metastore.create_dataset(name="test_dataset")
    metastore.remove_dataset(ds)
    metastore.remove_dataset(ds)  # not raises an exception


def test_update_dataset(metastore):
    ds = metastore.create_dataset(name="test_dataset")
    ds = metastore.create_dataset_version(
        dataset=ds, version="1.2.3", status=DatasetStatus.CREATED
    )

    updated_ds = metastore.update_dataset(
        ds,
        name="updated_test_dataset",
        status=DatasetStatus.COMPLETE,
        sources="gs://test_source",
        feature_schema=feature_schema,
        query_script=query_script,
        schema=schema,
        description="updated test dataset",
        attrs=["updated", "dataset"],
    )

    assert updated_ds.name == "updated_test_dataset"
    assert updated_ds.status == DatasetStatus.COMPLETE
    assert updated_ds.sources == "gs://test_source"
    assert updated_ds.feature_schema == feature_schema
    assert updated_ds.query_script == query_script
    assert updated_ds.schema == expected_schema
    assert updated_ds.description == "updated test dataset"
    assert updated_ds.attrs == ["updated", "dataset"]

    # test if method input dataset param is not mutated
    assert ds.name == "test_dataset"
    assert ds.description is None

    updated_ds = metastore.update_dataset(
        updated_ds,
        feature_schema=None,
        schema=None,
        description=None,
        attrs=None,
    )

    assert updated_ds.feature_schema is None
    assert updated_ds.schema is None
    assert updated_ds.description is None
    assert updated_ds.attrs is None


def test_update_dataset_read_only_values(metastore):
    ds = metastore.create_dataset(name="test_dataset")

    ds2 = metastore.update_dataset(
        ds,
        id=ds.id + 1,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    assert ds2.id == ds.id
    assert ds2.created_at == ds.created_at


def test_update_dataset_no_empty_values(metastore):
    ds = metastore.create_dataset(
        name="test_dataset",
        status=DatasetStatus.COMPLETE,
        sources=["gs://test_source"],
        query_script=query_script,
    )
    for field in ("name", "status", "sources", "query_script"):
        with pytest.raises(ValueError, match=f"{field} cannot be None"):
            metastore.update_dataset(ds, **{field: None})
    with pytest.raises(ValueError, match="name cannot be empty"):
        metastore.update_dataset(ds, name="")


def test_update_dataset_no_changes(metastore):
    ds = metastore.create_dataset(name="test_dataset")
    ds = metastore.create_dataset_version(
        dataset=ds, version="1.2.3", status=DatasetStatus.CREATED
    )

    updated_ds = metastore.update_dataset(ds)
    assert updated_ds == ds


def test_update_dataset_version(metastore):
    ds = metastore.create_dataset(name="test_dataset")
    ds = metastore.create_dataset_version(
        dataset=ds, version="1.2.3", status=DatasetStatus.CREATED
    )
    dv = ds.versions[0]

    finished_at = datetime.now(timezone.utc)
    job_id = str(uuid4())
    uuid = str(uuid4())

    updated_dv = metastore.update_dataset_version(
        ds,
        ds.latest_version,
        status=DatasetStatus.COMPLETE,
        sources="gs://test_source",
        feature_schema=feature_schema,
        query_script=query_script,
        error_message="Error message",
        error_stack="Error stack",
        script_output="Script output",
        finished_at=finished_at.isoformat(),
        schema=schema,
        num_objects=100,
        size=1000,
        preview=preview,
        job_id=job_id,
        uuid=uuid,
    )

    assert updated_dv.id is not None
    assert updated_dv.uuid == uuid
    assert updated_dv.dataset_id == ds.id
    assert updated_dv.version == "1.2.3"
    assert updated_dv.status == DatasetStatus.COMPLETE
    assert updated_dv.feature_schema == feature_schema
    assert updated_dv.finished_at == finished_at.isoformat()
    assert updated_dv.error_message == "Error message"
    assert updated_dv.error_stack == "Error stack"
    assert updated_dv.script_output == "Script output"
    assert updated_dv.schema == expected_schema
    assert updated_dv.num_objects == 100
    assert updated_dv.size == 1000
    assert updated_dv._preview_data == preview
    assert updated_dv.sources == "gs://test_source"
    assert updated_dv.query_script == query_script
    assert updated_dv.job_id == job_id

    # test if method input dataset param is also mutated
    assert dv.status == DatasetStatus.COMPLETE

    updated_dv = metastore.update_dataset_version(
        ds,
        ds.latest_version,
        feature_schema=None,
        finished_at=None,
        schema=None,
        num_objects=None,
        size=None,
        preview=None,
        job_id=None,
    )

    assert updated_dv.id is not None
    assert updated_dv.feature_schema is None
    assert updated_dv.finished_at is None
    assert updated_dv.schema is None
    assert updated_dv.num_objects is None
    assert updated_dv.size is None
    assert updated_dv._preview_data is None
    assert updated_dv.job_id is None


def test_update_dataset_version_read_only_values(metastore):
    ds = metastore.create_dataset(name="test_dataset")
    ds = metastore.create_dataset_version(
        dataset=ds, version="1.2.3", status=DatasetStatus.CREATED
    )
    dv = ds.versions[0]

    dv2 = metastore.update_dataset_version(
        ds,
        ds.latest_version,
        id=dv.id + 1,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    assert dv2.id == dv.id
    assert dv2.created_at == dv.created_at


def test_update_dataset_version_no_empty_values(metastore):
    ds = metastore.create_dataset(name="test_dataset")
    ds = metastore.create_dataset_version(
        dataset=ds,
        version="1.2.3",
        status=DatasetStatus.CREATED,
        sources="gs://test_source",
        query_script=query_script,
        error_message="Error message",
        error_stack="Error stack",
        script_output="Script output",
        uuid=str(uuid4()),
    )

    for field in (
        "status",
        "sources",
        "query_script",
        "error_message",
        "error_stack",
        "script_output",
        "uuid",
    ):
        with pytest.raises(ValueError, match=f"{field} cannot be None"):
            metastore.update_dataset_version(ds, ds.latest_version, **{field: None})


def test_update_dataset_version_bad_preview(metastore):
    ds = metastore.create_dataset(name="test_dataset")
    ds = metastore.create_dataset_version(
        dataset=ds,
        version="1.2.3",
        status=DatasetStatus.CREATED,
        sources="gs://test_source",
        query_script=query_script,
        error_message="Error message",
        error_stack="Error stack",
        script_output="Script output",
        uuid=str(uuid4()),
    )

    with pytest.raises(ValueError, match="must be a list"):
        metastore.update_dataset_version(ds, ds.latest_version, preview="not a list")


def test_update_dataset_version_no_changes(metastore):
    ds = metastore.create_dataset(name="test_dataset")
    ds = metastore.create_dataset_version(
        dataset=ds, version="1.2.3", status=DatasetStatus.CREATED
    )
    dv = ds.versions[0]

    updated_dv = metastore.update_dataset_version(ds, ds.latest_version)
    assert updated_dv == dv


def test_update_dataset_version_not_found(metastore):
    ds = metastore.create_dataset(name="test_dataset")
    ds = metastore.create_dataset_version(
        dataset=ds, version="1.2.3", status=DatasetStatus.CREATED
    )

    with pytest.raises(Exception, match="does not have version"):
        metastore.update_dataset_version(ds, "9.9.9", status=DatasetStatus.COMPLETE)


def test_list_datasets(metastore):
    assert [ds.name for ds in metastore.list_datasets()] == []

    ds1 = metastore.create_dataset(name="dataset1")
    metastore.create_dataset_version(
        dataset=ds1, version="1.0.0", status=DatasetStatus.CREATED
    )
    ds2 = metastore.create_dataset(name="dataset2")
    metastore.create_dataset_version(
        dataset=ds2, version="2.0.0", status=DatasetStatus.COMPLETE
    )
    ds3 = metastore.create_dataset(name="dataset3")
    metastore.create_dataset_version(
        dataset=ds3, version="3.0.0", status=DatasetStatus.FAILED
    )

    datasets = list(metastore.list_datasets())
    assert {"dataset2"} == {ds.name for ds in datasets}
    # Each dataset should have at least one version
    for ds in datasets:
        assert hasattr(ds, "versions")
        assert len(ds.versions) >= 1


def test_list_datasets_by_project_id(metastore, project):
    assert [ds.name for ds in metastore.list_datasets()] == []

    ds1 = metastore.create_dataset(name="dataset1", project_id=project.id)
    metastore.create_dataset_version(
        dataset=ds1, version="1.0.0", status=DatasetStatus.COMPLETE
    )
    ds2 = metastore.create_dataset(name="dataset2", project_id=project.id)
    metastore.create_dataset_version(
        dataset=ds2, version="2.0.0", status=DatasetStatus.COMPLETE
    )
    ds3 = metastore.create_dataset(name="dataset3")  # default project
    metastore.create_dataset_version(
        dataset=ds3, version="3.0.0", status=DatasetStatus.COMPLETE
    )

    datasets = list(metastore.list_datasets(project_id=project.id))
    assert {"dataset1", "dataset2"} == {ds.name for ds in datasets}


def test_list_datasets_by_prefix(metastore):
    ds1 = metastore.create_dataset(name="prefix_foo")
    metastore.create_dataset_version(
        dataset=ds1, version="1.0.0", status=DatasetStatus.CREATED
    )
    ds2 = metastore.create_dataset(name="prefix_bar")
    metastore.create_dataset_version(
        dataset=ds2, version="2.0.0", status=DatasetStatus.COMPLETE
    )
    ds3 = metastore.create_dataset(name="other_baz")
    metastore.create_dataset_version(
        dataset=ds3, version="3.0.0", status=DatasetStatus.FAILED
    )

    datasets = list(metastore.list_datasets_by_prefix("prefix_"))
    assert {"prefix_bar"} == {ds.name for ds in datasets}
    for ds in datasets:
        assert hasattr(ds, "versions")
        assert len(ds.versions) >= 1

    assert [ds.name for ds in metastore.list_datasets_by_prefix("foo")] == []


def test_get_dataset(metastore):
    with pytest.raises(Exception, match="not found"):
        metastore.get_dataset(str(uuid4()))

    ds = metastore.create_dataset(name="my_dataset")
    fetched = metastore.get_dataset("my_dataset")
    assert fetched.name == "my_dataset"

    # Add a version to the dataset and fetch it
    metastore.create_dataset_version(
        dataset=ds, version="1.0.0", status=DatasetStatus.CREATED
    )
    fetched = metastore.get_dataset("my_dataset")
    assert fetched.name == "my_dataset"
    assert hasattr(fetched, "versions")
    assert len(fetched.versions) == 1
    assert fetched.versions[0].version == "1.0.0"

    # Add another version and fetch again
    metastore.create_dataset_version(
        dataset=fetched, version="2.0.0", status=DatasetStatus.COMPLETE
    )
    fetched2 = metastore.get_dataset("my_dataset")
    assert fetched2.name == "my_dataset"
    assert len(fetched2.versions) == 2
    assert {v.version for v in fetched2.versions} == {"1.0.0", "2.0.0"}


def test_remove_dataset_version(metastore):
    ds = metastore.create_dataset(name="ds")
    ds = metastore.create_dataset_version(
        dataset=ds, version="1.0.0", status=DatasetStatus.CREATED
    )
    ds = metastore.create_dataset_version(
        dataset=ds, version="2.0.0", status=DatasetStatus.COMPLETE
    )
    assert len(ds.versions) == 2

    # Removing non-existent version should raise an exception
    with pytest.raises(Exception, match="not found"):
        metastore.remove_dataset_version(ds, "9.9.9")

    # Remove one version, dataset should still exist with one version
    ds = metastore.remove_dataset_version(ds, "1.0.0")
    assert ds.name == "ds"
    assert len(ds.versions) == 1
    assert ds.versions[0].version == "2.0.0"
    # Dataset can still be fetched
    fetched = metastore.get_dataset("ds")
    assert len(fetched.versions) == 1
    assert fetched.versions[0].version == "2.0.0"

    # Remove last version, dataset should be deleted
    metastore.remove_dataset_version(ds, "2.0.0")
    with pytest.raises(Exception, match="not found"):
        metastore.get_dataset("ds")


def test_remove_dataset_version_cleans_dependencies(metastore):
    ds1 = metastore.create_dataset(name="ds1")
    ds1 = metastore.create_dataset_version(
        dataset=ds1, version="1.0.0", status=DatasetStatus.CREATED
    )
    ds1 = metastore.create_dataset_version(
        dataset=ds1, version="2.0.0", status=DatasetStatus.COMPLETE
    )

    ds2 = metastore.create_dataset(name="ds2")
    ds2 = metastore.create_dataset_version(
        dataset=ds2, version="1.0.0", status=DatasetStatus.CREATED
    )

    metastore.add_dataset_dependency(ds1, "1.0.0", ds2, "1.0.0")

    # Check dependency exists
    assert len(metastore.get_direct_dataset_dependencies(ds1, "1.0.0")) == 1

    # Remove ds1 v1.0.0, dependency should be cleaned
    metastore.remove_dataset_version(ds1, "1.0.0")
    assert len(metastore.get_direct_dataset_dependencies(ds1, "2.0.0")) == 0


def test_update_dataset_status(metastore):
    ds = metastore.create_dataset(name="ds_status")
    ds = metastore.create_dataset_version(
        dataset=ds, version="1.0.0", status=DatasetStatus.CREATED
    )

    # Update dataset status only
    ds = metastore.update_dataset_status(ds, DatasetStatus.COMPLETE)
    assert ds.status == DatasetStatus.COMPLETE

    # Update dataset and version status, with error fields
    ds = metastore.create_dataset_version(
        dataset=ds, version="2.0.0", status=DatasetStatus.CREATED
    )
    ds = metastore.update_dataset_status(
        ds,
        DatasetStatus.FAILED,
        version="2.0.0",
        error_message="err",
        error_stack="stack",
        script_output="out",
    )
    assert ds.status == DatasetStatus.FAILED
    dsv = next(v for v in ds.versions if v.version == "2.0.0")
    assert dsv.status == DatasetStatus.FAILED
    assert dsv.error_message == "err"
    assert dsv.error_stack == "stack"
    # script_output is set only if provided and status is COMPLETE/FAILED
    assert dsv.script_output == "out"
    # finished_at is set for final states
    assert dsv.finished_at is not None

    # If version does not exist, should raise
    with pytest.raises(Exception, match="does not have version"):
        metastore.update_dataset_status(
            ds, DatasetStatus.COMPLETE, version="nonexistent"
        )


def test_update_dataset_dependency_source(metastore):
    src1 = metastore.create_dataset(name="src1")
    src1 = metastore.create_dataset_version(
        dataset=src1, version="1.0.0", status=DatasetStatus.COMPLETE
    )
    src2 = metastore.create_dataset(name="src2")
    src2 = metastore.create_dataset_version(
        dataset=src2, version="1.0.0", status=DatasetStatus.COMPLETE
    )
    tgt = metastore.create_dataset(name="tgt")
    tgt = metastore.create_dataset_version(
        dataset=tgt, version="1.0.0", status=DatasetStatus.COMPLETE
    )

    # Add dependency: src1@1.0.0 -> tgt@1.0.0
    metastore.add_dataset_dependency(src1, "1.0.0", tgt, "1.0.0")
    deps = metastore.get_direct_dataset_dependencies(src1, "1.0.0")
    assert len(deps) == 1
    assert deps[0].name == "tgt"
    assert deps[0].version == "1.0.0"

    # Update dependency to src2@1.0.0
    metastore.update_dataset_dependency_source(
        src1, "1.0.0", new_source_dataset=src2, new_source_dataset_version="1.0.0"
    )
    # Now src1 should have no dependencies, src2 should have the dependency
    deps_src1 = metastore.get_direct_dataset_dependencies(src1, "1.0.0")
    deps_src2 = metastore.get_direct_dataset_dependencies(src2, "1.0.0")
    assert len(deps_src1) == 0
    assert len(deps_src2) == 1
    assert deps_src2[0].name == "tgt"
    assert deps_src2[0].version == "1.0.0"


def test_update_dataset_dependency_source_default_new_source(metastore):
    src = metastore.create_dataset(name="src")
    src = metastore.create_dataset_version(
        dataset=src, version="1.0.0", status=DatasetStatus.COMPLETE
    )
    src = metastore.create_dataset_version(
        dataset=src, version="2.0.0", status=DatasetStatus.COMPLETE
    )
    tgt = metastore.create_dataset(name="tgt")
    tgt = metastore.create_dataset_version(
        dataset=tgt, version="1.0.0", status=DatasetStatus.COMPLETE
    )

    # Add dependency: src@1.0.0 -> tgt@1.0.0
    metastore.add_dataset_dependency(src, "1.0.0", tgt, "1.0.0")
    deps = metastore.get_direct_dataset_dependencies(src, "1.0.0")
    assert len(deps) == 1
    assert deps[0].name == "tgt"
    assert deps[0].version == "1.0.0"

    # Call update_dataset_dependency_source without new_source_dataset
    metastore.update_dataset_dependency_source(
        src, "1.0.0", new_source_dataset_version="2.0.0"
    )
    assert len(metastore.get_direct_dataset_dependencies(src, "1.0.0")) == 0
    deps_after = metastore.get_direct_dataset_dependencies(src, "2.0.0")
    assert len(deps_after) == 1
    assert deps_after[0].name == "tgt"
    assert deps_after[0].version == "1.0.0"


def test_list_jobs_by_ids(metastore):
    job_id1 = metastore.create_job(
        name="job1",
        query="SELECT 1",
        query_type=JobQueryType.PYTHON,
        status=JobStatus.CREATED,
        workers=1,
    )
    job_id2 = metastore.create_job(
        name="job2",
        query="SELECT 2",
        query_type=JobQueryType.PYTHON,
        status=JobStatus.FAILED,
        workers=2,
    )
    job_id3 = metastore.create_job(
        name="job3",
        query="SELECT 3",
        query_type=JobQueryType.PYTHON,
        status=JobStatus.RUNNING,
        workers=3,
    )

    assert list(metastore.list_jobs_by_ids([])) == []

    jobs = list(metastore.list_jobs_by_ids([job_id1]))
    assert [job_id1] == [job.id for job in jobs]

    jobs = list(metastore.list_jobs_by_ids([job_id1, job_id2, job_id3]))
    assert {job_id1, job_id2, job_id3} == {job.id for job in jobs}


def test_create_and_get_job(metastore):
    assert metastore.get_job(str(uuid4())) is None

    job_id = metastore.create_job(
        name="test_job",
        query="SELECT 1",
        query_type=JobQueryType.PYTHON,
        status=JobStatus.CREATED,
        workers=2,
        python_version="3.10",
        params={"foo": "bar"},
    )
    assert job_id
    job = metastore.get_job(job_id)
    assert job is not None
    assert job.id == job_id
    assert job.name == "test_job"
    assert job.query == "SELECT 1"
    assert job.query_type == JobQueryType.PYTHON
    assert job.status == JobStatus.CREATED
    assert job.workers == 2
    assert job.python_version == "3.10"
    assert job.params == {"foo": "bar"}


def test_update_job(metastore):
    job_id = metastore.create_job(
        name="update_job",
        query="SELECT 2",
        query_type=JobQueryType.PYTHON,
        status=JobStatus.CREATED,
        workers=1,
    )
    job = metastore.get_job(job_id)
    assert job.status == JobStatus.CREATED

    updated = metastore.update_job(
        job_id,
        status=JobStatus.FAILED,
        error_message="err",
        error_stack="stack",
        finished_at=datetime.now(timezone.utc),
        metrics={"acc": 0.99},
    )
    assert updated.status == JobStatus.FAILED
    assert updated.error_message == "err"
    assert updated.error_stack == "stack"
    assert updated.finished_at is not None
    assert updated.metrics == {"acc": 0.99}


def test_set_job_status(metastore):
    job_id = metastore.create_job(
        name="status_job",
        query="SELECT 3",
        query_type=JobQueryType.PYTHON,
        status=JobStatus.CREATED,
        workers=1,
    )

    metastore.set_job_status(
        job_id, JobStatus.FAILED, error_message="fail", error_stack="trace"
    )
    job = metastore.get_job(job_id)
    assert job.status == JobStatus.FAILED
    assert job.error_message == "fail"
    assert job.error_stack == "trace"


def test_get_job_status(metastore):
    assert metastore.get_job_status(str(uuid4())) is None

    job_id = metastore.create_job(
        name="status_job2",
        query="SELECT 4",
        query_type=JobQueryType.PYTHON,
        status=JobStatus.CREATED,
        workers=1,
    )

    status = metastore.get_job_status(job_id)
    assert status == JobStatus.CREATED

    metastore.set_job_status(job_id, JobStatus.RUNNING)
    status2 = metastore.get_job_status(job_id)
    assert status2 == JobStatus.RUNNING


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 5])
def test_get_ancestor_job_ids(metastore, depth):
    """Test get_ancestor_job_ids with different hierarchy depths."""
    # Create a chain of jobs with parent relationships
    # depth=0: single job with no parent
    # depth=1: job -> parent
    # depth=2: job -> parent -> grandparent

    job_ids = []
    parent_id = None

    # Create jobs from root to leaf
    for i in range(depth + 1):
        job_id = metastore.create_job(
            name=f"job_{i}",
            query=f"SELECT {i}",
            query_type=JobQueryType.PYTHON,
            status=JobStatus.CREATED,
            workers=1,
            parent_job_id=parent_id,
        )
        job_ids.append(job_id)
        parent_id = job_id

    # The last job is the leaf (youngest)
    leaf_job_id = job_ids[-1]

    ancestors = metastore.get_ancestor_job_ids(leaf_job_id)

    # Should return all ancestors except the leaf itself, in order from parent to root
    expected_ancestors = list(reversed(job_ids[:-1]))

    assert ancestors == expected_ancestors
    assert len(ancestors) == depth
