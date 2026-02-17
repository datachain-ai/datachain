"""
Functional tests for read_dataset when accessing remote (Studio) datasets.
"""

from urllib.parse import parse_qs, urlparse

import pytest

import datachain as dc
from datachain import json
from datachain.error import (
    DataChainError,
    DatasetNotFoundError,
    DatasetVersionNotFoundError,
)
from datachain.utils import STUDIO_URL
from tests.conftest import (
    REMOTE_DATASET_UUID,
    REMOTE_DATASET_UUID_V2,
    REMOTE_NAMESPACE_NAME,
    REMOTE_PROJECT_NAME,
)
from tests.utils import skip_if_not_sqlite


@pytest.fixture
def remote_dataset_version_v1(
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
        "num_objects": 1,
        "size": 1024,
        "preview": json.loads(json.dumps(dataset_rows, serialize_bytes=True)),
        "script_output": "",
        "schema": remote_dataset_schema,
        "sources": "",
        "query_script": (
            "from datachain.query.dataset import DatasetQuery\n"
            'DatasetQuery(path="s3://test-bucket")',
        ),
        "created_by_id": 1,
    }


@pytest.fixture
def remote_dataset_version_v2(
    remote_dataset_schema, dataset_rows, remote_file_feature_schema
):
    return {
        "id": 2,
        "uuid": REMOTE_DATASET_UUID_V2,
        "dataset_id": 1,
        "version": "2.0.0",
        "status": 4,
        "feature_schema": remote_file_feature_schema,
        "created_at": "2024-02-24T10:42:31.842944+00:00",
        "finished_at": "2024-02-24T10:43:31.842944+00:00",
        "error_message": "",
        "error_stack": "",
        "num_objects": 1,
        "size": 2048,
        "preview": json.loads(json.dumps(dataset_rows, serialize_bytes=True)),
        "script_output": "",
        "schema": remote_dataset_schema,
        "sources": "",
        "query_script": (
            "from datachain.query.dataset import DatasetQuery\n"
            'DatasetQuery(path="s3://test-bucket")',
        ),
        "created_by_id": 1,
    }


@pytest.fixture
def remote_dataset_single_version(
    remote_project,
    remote_dataset_version_v1,
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
        "versions": [remote_dataset_version_v1],
    }


@pytest.fixture
def remote_dataset_multi_version(
    remote_project,
    remote_dataset_version_v1,
    remote_dataset_version_v2,
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
        "versions": [remote_dataset_version_v1, remote_dataset_version_v2],
    }


@pytest.fixture
def mock_dataset_info_endpoint(requests_mock):
    """Mock the dataset info endpoint to return dataset information."""

    def _mock_info(dataset_data):
        return requests_mock.get(
            f"{STUDIO_URL}/api/datachain/datasets/info", json=dataset_data
        )

    return _mock_info


@pytest.fixture
def mock_dataset_info_not_found(requests_mock):
    """Mock the dataset info endpoint to return 404 not found."""
    return requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/info",
        status_code=404,
        json={"message": "Dataset not found"},
    )


def _get_version_from_request(request, default="1.0.0"):
    parsed_url = urlparse(request.url)
    query_params = parse_qs(parsed_url.query)
    return query_params.get("version", [default])[0]


@pytest.fixture
def mock_export_endpoint_with_urls(requests_mock):
    """Mock the export endpoint to return export_id + signed_urls."""

    def _mock_export_response(request, context):
        version_param = _get_version_from_request(request)
        version_file = version_param.replace(".", "_")
        return {
            "export_id": 1,
            "signed_urls": [
                f"https://studio-blobvault.s3.amazonaws.com/"
                f"datachain_ds_export_{version_file}.parquet.lz4"
            ],
        }

    return requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/export", json=_mock_export_response
    )


@pytest.fixture
def mock_export_status_completed(requests_mock):
    """Mock the export status endpoint to return completed status based on version."""

    def _mock_status_response(request, context):
        return {
            "status": "completed",
            "files_done": 1,
            "num_files": 1,
        }

    return requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/export-status", json=_mock_status_response
    )


@pytest.fixture
def mock_export_status_failed(requests_mock):
    """Mock the export status endpoint to return failed status based on version."""

    def _mock_failed_response(request, context):
        return {
            "status": "failed",
            "files_done": 0,
            "num_files": 1,
            "error_message": "Export failed",
        }

    return requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/export-status", json=_mock_failed_response
    )


@pytest.fixture
def mock_s3_parquet_download(requests_mock, compressed_parquet_data, dog_entries):
    """Mock S3 parquet file download for all versions."""

    def _mock_download():
        # Generate different data for each version
        for version in ["1.0.0", "2.0.0"]:
            parquet_data = compressed_parquet_data(dog_entries(version))
            requests_mock.get(
                f"https://studio-blobvault.s3.amazonaws.com/"
                f"datachain_ds_export_{version.replace('.', '_')}.parquet.lz4",
                content=parquet_data,
            )

    return _mock_download


@pytest.fixture
def mock_dataset_rows_fetcher_status_check(mocker):
    """Mock DatasetRowsFetcher.should_check_for_status to return True."""
    return mocker.patch(
        "datachain.catalog.catalog.DatasetRowsFetcher.should_check_for_status",
        return_value=True,
    )


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_basic(
    studio_token,
    test_session,
    remote_dataset_single_version,
    mock_dataset_info_endpoint,
    mock_export_endpoint_with_urls,
    mock_export_status_completed,
    mock_s3_parquet_download,
    mock_dataset_rows_fetcher_status_check,
):
    """Test basic read_dataset functionality with remote dataset."""
    mock_dataset_info_endpoint(remote_dataset_single_version)
    mock_s3_parquet_download()

    # Ensure dataset is not available locally at first
    with pytest.raises(DatasetNotFoundError):
        dc.read_dataset("dogs", session=test_session)

    assert (
        dc.read_dataset(
            f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
            version="1.0.0",
            session=test_session,
        ).to_values("version")[0]
        == "1.0.0"
    )


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_already_exists(
    studio_token,
    test_session,
    requests_mock,
    remote_dataset_single_version,
    mock_dataset_info_endpoint,
    mock_export_endpoint_with_urls,
    mock_export_status_completed,
    mock_s3_parquet_download,
    mock_dataset_rows_fetcher_status_check,
):
    """Test read_dataset when dataset already exists locally from previous read."""

    # Mock the Studio API responses
    mock_dataset_info_endpoint(remote_dataset_single_version)
    mock_s3_parquet_download()

    # First read - downloads from remote
    ds1 = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        version="1.0.0",
        session=test_session,
    )

    assert ds1.to_values("version")[0] == "1.0.0"
    assert ds1.dataset.name == "dogs"
    assert dc.datasets().to_values("version") == ["1.0.0"]

    # Second read - should use local dataset without calling remote
    # Clear the mock to ensure no new remote calls are made
    requests_mock.reset_mock()

    ds2 = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        version="1.0.0",
        session=test_session,
    )

    assert ds2.to_values("version")[0] == "1.0.0"
    assert ds2.dataset.name == "dogs"
    assert dc.datasets().to_values("version") == ["1.0.0"]
    assert ds2.dataset.versions[0].uuid == REMOTE_DATASET_UUID

    # Verify no remote calls were made for the second read
    assert not requests_mock.called


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_update_flag(
    studio_token,
    test_session,
    remote_dataset_multi_version,
    mock_dataset_info_endpoint,
    mock_export_endpoint_with_urls,
    mock_export_status_completed,
    mock_s3_parquet_download,
    mock_dataset_rows_fetcher_status_check,
    requests_mock,
):
    """Test read_dataset with update=True flag to force remote check."""

    # Mock the Studio API responses
    mock_dataset_info_endpoint(remote_dataset_multi_version)
    mock_s3_parquet_download()

    # First read - downloads version 1.0.0
    ds1 = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        version="1.0.0",
        session=test_session,
    )
    assert dc.datasets().to_values("version") == ["1.0.0"]
    assert ds1.to_values("version")[0] == "1.0.0"

    # Read without update and version returns a cached version
    ds1 = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        session=test_session,
    )
    assert dc.datasets().to_values("version") == ["1.0.0"]
    assert ds1.to_values("version")[0] == "1.0.0"

    # Second read with update=True with the exact version
    # returns the same dataset version
    ds2 = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        version="1.0.0",
        update=True,
        session=test_session,
    )
    assert dc.datasets().to_values("version") == ["1.0.0"]
    assert ds2.to_values("version")[0] == "1.0.0"

    # Third read with update=False even with version specifier
    # that allows for newer version still bring the same version
    # as the one already downloaded
    ds3 = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        version=">=1.0.0",
        update=False,
        session=test_session,
    )
    assert dc.datasets().to_values("version") == ["1.0.0"]
    assert ds3.to_values("version")[0] == "1.0.0"

    # Finally, read with update=True brings the latest version
    ds4 = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        version=">=1.0.0",
        update=True,
        session=test_session,
    )

    assert ds4.to_values("version")[0] == "2.0.0"
    assert dc.datasets().to_values("version") == ["1.0.0", "2.0.0"]


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_update_flag_no_version(
    studio_token,
    test_session,
    remote_dataset_multi_version,
    mock_dataset_info_endpoint,
    mock_export_endpoint_with_urls,
    mock_export_status_completed,
    mock_s3_parquet_download,
    mock_dataset_rows_fetcher_status_check,
    requests_mock,
):
    """Test read_dataset with update=True flag to force remote check."""

    # Mock the Studio API responses
    mock_dataset_info_endpoint(remote_dataset_multi_version)
    mock_s3_parquet_download()

    # First read - downloads version 1.0.0
    ds1 = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        version="1.0.0",
        session=test_session,
    )
    assert dc.datasets().to_values("version") == ["1.0.0"]
    assert ds1.to_values("version")[0] == "1.0.0"

    # Read with update=True w/o version specifier also
    # checks the most recent remote version and brings it
    ds4 = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        update=True,
        session=test_session,
    )

    assert ds4.to_values("version")[0] == "2.0.0"
    assert dc.datasets().to_values("version") == ["1.0.0", "2.0.0"]


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_version_specifiers(
    studio_token,
    test_session,
    remote_dataset_multi_version,
    mock_dataset_info_endpoint,
    mock_export_endpoint_with_urls,
    mock_export_status_completed,
    mock_s3_parquet_download,
    mock_dataset_rows_fetcher_status_check,
):
    """Test read_dataset with version specifiers on remote datasets."""

    # Mock the Studio API responses
    mock_dataset_info_endpoint(remote_dataset_multi_version)
    mock_s3_parquet_download()

    # Test reading with version specifier ">=1.0.0" should get latest (2.0.0)
    ds = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        version=">=1.0.0",
        session=test_session,
    )

    # Should get the latest version matching the specifier (2.0.0)
    assert ds.dataset.name == "dogs"
    dataset_version = ds.dataset.get_version("2.0.0")
    assert dataset_version is not None
    assert dataset_version.uuid == REMOTE_DATASET_UUID_V2
    assert dc.datasets().to_values("version") == ["2.0.0"]
    assert ds.to_values("version")[0] == "2.0.0"


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_version_specifier_no_match(
    studio_token,
    test_session,
    remote_dataset_multi_version,
    mock_dataset_info_endpoint,
    mock_dataset_rows_fetcher_status_check,
):
    """Test read_dataset with version specifier that doesn't match."""

    mock_dataset_info_endpoint(remote_dataset_multi_version)

    # Test version specifier that doesn't match any existing version
    with pytest.raises(DatasetVersionNotFoundError) as exc_info:
        dc.read_dataset(
            f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
            version=">=3.0.0",
            session=test_session,
        )

    assert "No dataset" in str(exc_info.value)
    assert "version matching specifier >=3.0.0" in str(exc_info.value)


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_not_found(
    studio_token,
    test_session,
    mock_dataset_info_not_found,
):
    """Test read_dataset when remote dataset is not found."""

    with pytest.raises(DatasetNotFoundError) as exc_info:
        dc.read_dataset(
            f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.nonexistent",
            version="1.0.0",
            session=test_session,
        )

    expected_msg = (
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.nonexistent not found"
    )
    assert expected_msg in str(exc_info.value)


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_version_not_found(
    studio_token,
    test_session,
    remote_dataset_single_version,
    mock_dataset_info_endpoint,
    mock_dataset_rows_fetcher_status_check,
):
    """Test read_dataset when requested version doesn't exist on remote."""

    mock_dataset_info_endpoint(remote_dataset_single_version)

    with pytest.raises(DataChainError) as exc_info:
        dc.read_dataset(
            f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
            version="5.0.0",
            session=test_session,
        )

    assert "Dataset dogs doesn't have version 5.0.0 on server" in str(exc_info.value)


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_latest_version(
    studio_token,
    test_session,
    remote_dataset_multi_version,
    mock_dataset_info_endpoint,
    mock_export_endpoint_with_urls,
    mock_export_status_completed,
    mock_s3_parquet_download,
    mock_dataset_rows_fetcher_status_check,
):
    """Test read_dataset without version parameter should get latest version."""

    # Mock the Studio API responses
    mock_dataset_info_endpoint(remote_dataset_multi_version)
    mock_s3_parquet_download()

    # Read without specifying version should get latest
    ds = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        session=test_session,
    )

    # Should get the latest version (2.0.0)
    assert ds.dataset.name == "dogs"
    dataset_version = ds.dataset.get_version("2.0.0")
    assert dataset_version is not None
    assert dataset_version.uuid == REMOTE_DATASET_UUID_V2
    assert dc.datasets().to_values("version") == ["2.0.0"]
    assert ds.to_values("version")[0] == "2.0.0"


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_export_failed(
    studio_token,
    test_session,
    remote_dataset_single_version,
    mock_dataset_info_endpoint,
    mock_export_endpoint_with_urls,
    mock_export_status_failed,
    mock_dataset_rows_fetcher_status_check,
):
    """Test read_dataset when remote dataset export fails."""

    mock_dataset_info_endpoint(remote_dataset_single_version)

    with pytest.raises(DataChainError) as exc_info:
        dc.read_dataset(
            f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
            version="1.0.0",
            session=test_session,
        )

    assert "Dataset export failed in Studio" in str(exc_info.value)
