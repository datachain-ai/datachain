import pytest
import requests
import sqlalchemy as sa

import datachain as dc
from datachain import json
from datachain.client.fsspec import Client
from datachain.dataset import DatasetStatus
from datachain.error import DataChainError, DatasetNotFoundError
from datachain.query.session import Session
from datachain.sql.types import String
from datachain.utils import STUDIO_URL
from tests.conftest import (
    REMOTE_DATASET_UUID,
    REMOTE_NAMESPACE_NAME,
    REMOTE_NAMESPACE_UUID,
    REMOTE_PROJECT_NAME,
    REMOTE_PROJECT_UUID,
)
from tests.utils import (
    assert_row_names,
    skip_if_not_sqlite,
    tree_from_path,
)


@pytest.fixture
def remote_dataset_version(
    remote_dataset_schema, dataset_rows, remote_file_feature_schema
):
    return {
        "id": 1,
        "uuid": REMOTE_DATASET_UUID,
        "dataset_id": 1,
        "version": "1.0.0",
        "status": 4,
        "feature_schema": remote_file_feature_schema,
        "created_at": "2024-02-23T10:42:31.842944+00:00",
        "finished_at": "2024-02-23T10:43:31.842944+00:00",
        "error_message": "",
        "error_stack": "",
        "num_objects": 5,
        "size": 1073741824,
        "preview": json.loads(json.dumps(dataset_rows, serialize_bytes=True)),
        "script_output": "",
        "schema": remote_dataset_schema,
        "sources": "",
        "query_script": (
            'from datachain.query.dataset import DatasetQuery\nDatasetQuery(path="s3://ldb-public")',
        ),
        "created_by_id": 1,
    }


@pytest.fixture
def remote_dataset(
    remote_project,
    remote_dataset_version,
    remote_dataset_schema,
    remote_file_feature_schema,
):
    return {
        "id": 1,
        "name": "dogs",
        "project": remote_project,
        "description": "",
        "attrs": [],
        "schema": remote_dataset_schema,
        "status": 4,
        "feature_schema": remote_file_feature_schema,
        "created_at": "2024-02-23T10:42:31.842944+00:00",
        "finished_at": "2024-02-23T10:43:31.842944+00:00",
        "error_message": "",
        "error_stack": "",
        "script_output": "",
        "job_id": "f74ec414-58b7-437d-81c5-d41e5365abba",
        "sources": "",
        "query_script": "",
        "team_id": 1,
        "warehouse_id": None,
        "created_by_id": 1,
        "versions": [remote_dataset_version],
    }


@pytest.fixture
def remote_dataset_chunk_url():
    return (
        "https://studio-blobvault.s3.amazonaws.com/datachain_ds_export_1_0.parquet.lz4"
    )


@pytest.fixture
def remote_dataset_info(requests_mock, remote_dataset):
    requests_mock.get(f"{STUDIO_URL}/api/datachain/datasets/info", json=remote_dataset)


@pytest.fixture
def dataset_export(requests_mock, remote_dataset_chunk_url):
    requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/export",
        json={"export_id": 1, "signed_urls": [remote_dataset_chunk_url]},
    )


@pytest.fixture
def dataset_export_status(requests_mock):
    requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/export-status",
        json={"status": "completed"},
    )


@pytest.fixture
def dataset_export_data_chunk(
    requests_mock, remote_dataset_chunk_url, mock_parquet_data_cloud
):
    requests_mock.get(remote_dataset_chunk_url, content=mock_parquet_data_cloud)


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@pytest.mark.parametrize(
    "dataset_uri",
    [
        f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0",
        f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
    ],
)
@pytest.mark.parametrize("local_ds_name", [None, "other"])
@pytest.mark.parametrize("local_ds_version", [None, "2.0.0"])
@pytest.mark.parametrize("instantiate", [True, False])
@skip_if_not_sqlite
def test_pull_dataset_success(
    mocker,
    studio_token,
    cloud_test_catalog,
    remote_dataset_info,
    dataset_export,
    dataset_export_status,
    dataset_export_data_chunk,
    dataset_uri,
    local_ds_name,
    local_ds_version,
    instantiate,
):
    mocker.patch(
        "datachain.catalog.catalog.DatasetRowsFetcher.should_check_for_status",
        return_value=True,
    )

    src_uri = cloud_test_catalog.src_uri
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog

    dest = None

    if instantiate:
        dest = working_dir / "data"
        dest.mkdir()
        catalog.pull_dataset(
            dataset_uri,
            output=str(dest),
            local_ds_name=local_ds_name,
            local_ds_version=local_ds_version,
            cp=True,
        )
    else:
        # Verify idempotency
        for _ in range(2):
            catalog.pull_dataset(
                dataset_uri,
                local_ds_name=local_ds_name,
                local_ds_version=local_ds_version,
                cp=False,
            )

    project = catalog.metastore.get_project(REMOTE_PROJECT_NAME, REMOTE_NAMESPACE_NAME)
    dataset = catalog.get_dataset(
        local_ds_name or "dogs",
        namespace_name=project.namespace.name,
        project_name=project.name,
    )
    assert dataset.project.namespace.uuid == REMOTE_NAMESPACE_UUID
    assert dataset.project.uuid == REMOTE_PROJECT_UUID

    assert [v.version for v in dataset.versions] == [local_ds_version or "1.0.0"]
    assert dataset.status == DatasetStatus.COMPLETE
    assert dataset.created_at
    assert dataset.finished_at
    assert dataset.schema
    dataset_version = dataset.get_version(local_ds_version or "1.0.0")
    assert dataset_version.status == DatasetStatus.COMPLETE
    assert dataset_version.created_at
    assert dataset_version.finished_at
    assert dataset_version.schema
    assert dataset_version.num_objects == 4
    assert dataset_version.size == 15
    assert dataset_version.uuid == REMOTE_DATASET_UUID

    assert_row_names(
        catalog,
        dataset,
        local_ds_version or "1.0.0",
        {
            "dog1",
            "dog2",
            "dog3",
            "dog4",
        },
    )

    client = Client.get_client(src_uri, None)

    if instantiate:
        assert tree_from_path(dest) == {
            f"{client.name}": {
                "dogs": {
                    "dog1": "woof",
                    "dog2": "arf",
                    "dog3": "bark",
                    "others": {"dog4": "ruff"},
                }
            }
        }


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@pytest.mark.parametrize("is_studio", (False,))
@skip_if_not_sqlite
def test_datachain_read_dataset_pull(
    studio_token,
    cloud_test_catalog,
    remote_dataset_info,
    dataset_export,
    dataset_export_status,
    dataset_export_data_chunk,
):
    catalog = cloud_test_catalog.catalog

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("dogs")

    with Session("testSession", catalog=catalog):
        ds = dc.read_dataset(
            name=f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
            version="1.0.0",
        )

    assert ds.dataset.name == "dogs"
    assert ds.dataset.latest_version == "1.0.0"
    assert ds.dataset.status == DatasetStatus.COMPLETE

    project = catalog.metastore.get_project(REMOTE_PROJECT_NAME, REMOTE_NAMESPACE_NAME)
    dataset = catalog.get_dataset(
        "dogs", namespace_name=project.namespace.name, project_name=project.name
    )
    assert dataset.name == "dogs"


@skip_if_not_sqlite
def test_pull_dataset_wrong_dataset_uri_format(
    studio_token,
    requests_mock,
    catalog,
    remote_dataset,
):
    with pytest.raises(DataChainError) as exc_info:
        catalog.pull_dataset("wrong")
    assert str(exc_info.value) == "Error when parsing dataset uri"


@skip_if_not_sqlite
def test_pull_dataset_wrong_version(
    studio_token,
    requests_mock,
    catalog,
    remote_dataset_info,
):
    with pytest.raises(DataChainError) as exc_info:
        catalog.pull_dataset(
            f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v5"
        )
    assert str(exc_info.value) == "Dataset dogs doesn't have version 5 on server"


@skip_if_not_sqlite
def test_pull_dataset_not_found_in_remote(
    studio_token,
    requests_mock,
    catalog,
):
    requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/info",
        status_code=404,
        json={"message": "Dataset not found"},
    )

    full_name = f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs"

    with pytest.raises(DatasetNotFoundError) as exc_info:
        catalog.pull_dataset(
            f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0"
        )
    assert str(exc_info.value) == f"Dataset {full_name} not found"


@pytest.mark.parametrize("export_status", ["failed", "removed"])
@skip_if_not_sqlite
def test_pull_dataset_exporting_dataset_failed_in_remote(
    studio_token,
    requests_mock,
    catalog,
    remote_dataset_info,
    dataset_export,
    export_status,
):
    requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/export-status",
        json={"status": export_status},
    )

    with pytest.raises(DataChainError) as exc_info:
        catalog.pull_dataset(
            f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0"
        )
    assert str(exc_info.value) == f"Dataset export {export_status} in Studio"


@skip_if_not_sqlite
def test_pull_dataset_empty_parquet(
    studio_token,
    requests_mock,
    catalog,
    remote_dataset_info,
    dataset_export,
    dataset_export_status,
    remote_dataset_chunk_url,
):
    requests_mock.get(remote_dataset_chunk_url, content=b"")

    with pytest.raises(RuntimeError):
        catalog.pull_dataset(
            f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0"
        )


@skip_if_not_sqlite
def test_pull_dataset_already_exists_locally(
    studio_token,
    catalog,
    remote_dataset_info,
    dataset_export,
    dataset_export_status,
    remote_dataset_chunk_url,
    requests_mock,
    mock_parquet_data,
):
    requests_mock.get(remote_dataset_chunk_url, content=mock_parquet_data)

    catalog.pull_dataset(
        f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0",
        local_ds_name="other",
    )
    catalog.pull_dataset(
        f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0"
    )

    project = catalog.metastore.get_project(REMOTE_PROJECT_NAME, REMOTE_NAMESPACE_NAME)
    other = catalog.get_dataset(
        "other", namespace_name=project.namespace.name, project_name=project.name
    )
    other_version = other.get_version("1.0.0")
    assert other_version.uuid == REMOTE_DATASET_UUID
    assert other_version.num_objects == 4
    assert other_version.size == 15

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(
            "dogs", namespace_name=project.namespace.name, project_name=project.name
        )


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@pytest.mark.parametrize("local_ds_name", [None])
@skip_if_not_sqlite
def test_pull_dataset_local_name_already_exists(
    studio_token,
    cloud_test_catalog,
    remote_dataset_info,
    dataset_export,
    dataset_export_status,
    dataset_export_data_chunk,
    local_ds_name,
):
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri

    project = catalog.metastore.create_project(
        REMOTE_NAMESPACE_NAME, REMOTE_PROJECT_NAME
    )
    catalog.create_dataset_from_sources(
        local_ds_name or "dogs", [f"{src_uri}/dogs/*"], recursive=True, project=project
    )
    with pytest.raises(DataChainError) as exc_info:
        catalog.pull_dataset(
            f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0",
            local_ds_name=local_ds_name,
        )

    assert str(exc_info.value) == (
        f"Local dataset ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}."
        f"{local_ds_name or 'dogs'}"
        "@v1.0.0 already exists with"
        " different uuid, please choose different local dataset name or version"
    )

    catalog.pull_dataset(
        f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0",
        local_ds_name=local_ds_name,
        local_ds_version="2.0.0",
    )


@skip_if_not_sqlite
def test_pull_dataset_empty_signed_urls(
    studio_token,
    catalog,
    remote_dataset_info,
    dataset_export_status,
    requests_mock,
):
    requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/export",
        json={"export_id": 1, "signed_urls": []},
    )

    catalog.pull_dataset(
        f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0"
    )

    dataset = catalog.get_dataset(
        "dogs",
        namespace_name=REMOTE_NAMESPACE_NAME,
        project_name=REMOTE_PROJECT_NAME,
    )
    assert dataset.status == DatasetStatus.COMPLETE

    version = dataset.get_version("1.0.0")
    assert version.num_objects == 0

    assert catalog.warehouse.get_temp_table_names() == []


@skip_if_not_sqlite
def test_pull_export_api_failure_cleans_temp_table_and_retry_succeeds(
    studio_token,
    catalog,
    remote_dataset_info,
    dataset_export_status,
    remote_dataset_chunk_url,
    requests_mock,
    mock_parquet_data,
):
    requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/export",
        status_code=400,
        json={"message": "Export not available"},
    )

    with pytest.raises(DataChainError, match="Export not available"):
        catalog.pull_dataset(
            f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0"
        )

    assert catalog.warehouse.get_temp_table_names() == []

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(
            "dogs",
            namespace_name=REMOTE_NAMESPACE_NAME,
            project_name=REMOTE_PROJECT_NAME,
            include_incomplete=True,
        )

    requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/export",
        json={"export_id": 1, "signed_urls": [remote_dataset_chunk_url]},
    )
    requests_mock.get(remote_dataset_chunk_url, content=mock_parquet_data)
    catalog.pull_dataset(
        f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0"
    )

    dataset = catalog.get_dataset(
        "dogs",
        namespace_name=REMOTE_NAMESPACE_NAME,
        project_name=REMOTE_PROJECT_NAME,
    )
    assert dataset.status == DatasetStatus.COMPLETE


@skip_if_not_sqlite
def test_pull_download_failure_leaves_no_state_and_retry_succeeds(
    studio_token,
    catalog,
    remote_dataset_info,
    dataset_export,
    dataset_export_status,
    remote_dataset_chunk_url,
    requests_mock,
    mock_parquet_data,
):
    requests_mock.get(remote_dataset_chunk_url, status_code=500, text="Server error")

    with pytest.raises(requests.HTTPError):
        catalog.pull_dataset(
            f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0"
        )

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(
            "dogs",
            namespace_name=REMOTE_NAMESPACE_NAME,
            project_name=REMOTE_PROJECT_NAME,
            include_incomplete=True,
        )

    requests_mock.get(remote_dataset_chunk_url, content=mock_parquet_data)
    catalog.pull_dataset(
        f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0"
    )

    dataset = catalog.get_dataset(
        "dogs",
        namespace_name=REMOTE_NAMESPACE_NAME,
        project_name=REMOTE_PROJECT_NAME,
    )
    assert dataset.status == DatasetStatus.COMPLETE


@skip_if_not_sqlite
def test_pull_corrupt_parquet_leaves_no_state_and_retry_succeeds(
    studio_token,
    catalog,
    remote_dataset_info,
    dataset_export,
    dataset_export_status,
    remote_dataset_chunk_url,
    requests_mock,
    mock_parquet_data,
):
    requests_mock.get(remote_dataset_chunk_url, content=b"not valid parquet")

    with pytest.raises(RuntimeError):
        catalog.pull_dataset(
            f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0"
        )

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(
            "dogs",
            namespace_name=REMOTE_NAMESPACE_NAME,
            project_name=REMOTE_PROJECT_NAME,
            include_incomplete=True,
        )

    requests_mock.get(remote_dataset_chunk_url, content=mock_parquet_data)
    catalog.pull_dataset(
        f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0"
    )

    dataset = catalog.get_dataset(
        "dogs",
        namespace_name=REMOTE_NAMESPACE_NAME,
        project_name=REMOTE_PROJECT_NAME,
    )
    assert dataset.status == DatasetStatus.COMPLETE


@skip_if_not_sqlite
def test_pull_insertion_failure_leaves_no_state_and_retry_succeeds(
    mocker,
    studio_token,
    catalog,
    remote_dataset_info,
    dataset_export,
    dataset_export_status,
    remote_dataset_chunk_url,
    requests_mock,
    mock_parquet_data,
):
    requests_mock.get(remote_dataset_chunk_url, content=mock_parquet_data)

    mock_insert = mocker.patch(
        "datachain.data_storage.warehouse.AbstractWarehouse.insert_dataframe_to_table",
        side_effect=RuntimeError("Insert failed"),
    )

    with pytest.raises(RuntimeError, match="Insert failed"):
        catalog.pull_dataset(
            f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0"
        )

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(
            "dogs",
            namespace_name=REMOTE_NAMESPACE_NAME,
            project_name=REMOTE_PROJECT_NAME,
            include_incomplete=True,
        )

    mocker.stop(mock_insert)
    catalog.pull_dataset(
        f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0"
    )

    dataset = catalog.get_dataset(
        "dogs",
        namespace_name=REMOTE_NAMESPACE_NAME,
        project_name=REMOTE_PROJECT_NAME,
    )
    assert dataset.status == DatasetStatus.COMPLETE


@skip_if_not_sqlite
def test_pull_failure_after_saving_leaves_incomplete_version_and_retry_succeeds(
    mocker,
    studio_token,
    catalog,
    remote_dataset_info,
    dataset_export,
    dataset_export_status,
    remote_dataset_chunk_url,
    requests_mock,
    mock_parquet_data,
    capsys,
):
    requests_mock.get(remote_dataset_chunk_url, content=mock_parquet_data)

    mock_status = mocker.patch(
        "datachain.catalog.catalog.Catalog.update_dataset_version_with_warehouse_info",
        side_effect=RuntimeError("Post-save update failed"),
    )

    with pytest.raises(RuntimeError, match="Post-save update failed"):
        catalog.pull_dataset(
            f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0"
        )

    assert catalog.warehouse.get_temp_table_names() == []

    dataset = catalog.get_dataset(
        "dogs",
        namespace_name=REMOTE_NAMESPACE_NAME,
        project_name=REMOTE_PROJECT_NAME,
        include_incomplete=True,
    )
    assert dataset.get_version("1.0.0").status != DatasetStatus.COMPLETE

    mocker.stop(mock_status)
    capsys.readouterr()  # clear captured output
    catalog.pull_dataset(
        f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0"
    )

    captured = capsys.readouterr()
    assert "Cleaning up stale existing dataset version" in captured.out

    dataset = catalog.get_dataset(
        "dogs",
        namespace_name=REMOTE_NAMESPACE_NAME,
        project_name=REMOTE_PROJECT_NAME,
    )
    assert dataset.status == DatasetStatus.COMPLETE


@skip_if_not_sqlite
def test_pull_cleans_stale_incomplete_version_with_different_uuid(
    studio_token,
    catalog,
    remote_dataset_info,
    dataset_export,
    dataset_export_status,
    remote_dataset_chunk_url,
    requests_mock,
    mock_parquet_data,
    capsys,
):
    """If the target name+version is occupied by an incomplete version with a
    different UUID (e.g. a previous failed pull of a different remote version),
    pull should clean it up and proceed instead of crashing."""
    requests_mock.get(remote_dataset_chunk_url, content=mock_parquet_data)

    # Pre-create namespace/project and an incomplete version with a DIFFERENT uuid
    namespace = catalog.metastore.create_namespace(REMOTE_NAMESPACE_NAME)
    project = catalog.metastore.create_project(namespace.name, REMOTE_PROJECT_NAME)
    stale_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    catalog.create_dataset(
        "dogs",
        project,
        "1.0.0",
        columns=(sa.Column("name", String),),
        uuid=stale_uuid,
        validate_version=False,
    )

    # Verify it exists as CREATED (incomplete)
    ds = catalog.get_dataset(
        "dogs",
        namespace_name=REMOTE_NAMESPACE_NAME,
        project_name=REMOTE_PROJECT_NAME,
        include_incomplete=True,
    )
    assert ds.get_version("1.0.0").uuid == stale_uuid
    assert ds.get_version("1.0.0").status == DatasetStatus.CREATED

    # Pull the real remote dataset — should clean up the stale version and succeed
    capsys.readouterr()
    catalog.pull_dataset(
        f"ds://{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs@v1.0.0"
    )

    captured = capsys.readouterr()
    assert "Cleaning up stale incomplete version" in captured.out
    assert stale_uuid in captured.out

    dataset = catalog.get_dataset(
        "dogs",
        namespace_name=REMOTE_NAMESPACE_NAME,
        project_name=REMOTE_PROJECT_NAME,
    )
    assert dataset.status == DatasetStatus.COMPLETE
    assert dataset.get_version("1.0.0").uuid == REMOTE_DATASET_UUID
