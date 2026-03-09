import http.server
import os
import socketserver
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse

import pytest
import requests

import datachain as dc
from datachain import json
from datachain.dataset import DatasetStatus
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
    def _mock_info(dataset_data):
        return requests_mock.get(
            f"{STUDIO_URL}/api/datachain/datasets/info", json=dataset_data
        )

    return _mock_info


@pytest.fixture
def mock_dataset_info_not_found(requests_mock):
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
def mock_s3_parquet_download(requests_mock, compressed_parquet_data, dog_entries):
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
    return mocker.patch(
        "datachain.catalog.catalog.DatasetRowsFetcher.should_check_for_status",
        return_value=True,
    )


@pytest.fixture
def mock_studio_server(
    remote_dataset_single_version, compressed_parquet_data, dog_entries
):
    parquet_data = compressed_parquet_data(dog_entries("1.0.0"))

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            port = self.server.server_address[1]
            if "/api/datachain/datasets/export-status" in self.path:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    b'{"status": "completed", "files_done": 1, "num_files": 1}'
                )
            elif "/api/datachain/datasets/export" in self.path:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    f'{{"export_id": 1, "signed_urls": ["http://localhost:{port}/data.parquet.lz4"]}}'.encode()
                )
            elif "/api/datachain/datasets/info" in self.path:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(remote_dataset_single_version).encode())
            elif "/data.parquet.lz4" in self.path:
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                self.wfile.write(parquet_data)
            else:
                self.send_response(404)
                self.end_headers()

    server = socketserver.TCPServer(("", 0), Handler)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    yield port

    server.shutdown()
    server.server_close()


@pytest.fixture
def run_script(tmp_path, mock_studio_server):
    port = mock_studio_server
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    processes = []
    script_counter = 0

    def _run_script(script, capture_output=False, hang_point=None, signal_file=None):
        """
        Run a Python script in a subprocess.

        Args:
            script: The Python code to run
            capture_output: Whether to capture stdout/stderr
            hang_point: Optional tuple of (module_path, class_name, method_name)
                       e.g., ("datachain.catalog.catalog", "DatasetRowsFetcher",
                              "get_parquet_content")
                       The method will be patched to create signal_file and hang.
            signal_file: Path to signal file (required if hang_point is set)
        """
        if hang_point:
            if signal_file is None:
                raise ValueError("signal_file required when hang_point is set")
            module_path, class_name, method_name = hang_point
            # Generate wrapper that patches at module level (works across threads)
            wrapper = f"""
import sys
import time
from pathlib import Path

# Import the target module and patch BEFORE any other imports
import {module_path} as target_module

_original_method = getattr(target_module.{class_name}, "{method_name}")
_signal_file = Path("{signal_file.as_posix()}")

def _hang_wrapper(self, *args, **kwargs):
    _signal_file.touch()
    while True:
        time.sleep(1)

setattr(target_module.{class_name}, "{method_name}", _hang_wrapper)

# Now run the user script
"""
            script = wrapper + script

        nonlocal script_counter
        script_file = tmp_path / f"worker_{script_counter}.py"
        script_counter += 1
        script_file.write_text(script)
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": f"{project_root}/src",
                "DATACHAIN_ROOT_DIR": str(tmp_path),
                "DATACHAIN_STUDIO_URL": f"http://localhost:{port}",
                "DATACHAIN_STUDIO_TOKEN": "test",
                "DATACHAIN_STUDIO_TEAM": "test-team",
            }
        )
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, str(script_file)],
            stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL,
            stderr=subprocess.PIPE if capture_output else subprocess.DEVNULL,
            env=env,
        )
        processes.append(proc)
        return proc

    yield _run_script

    # Cleanup: kill any processes still running and close pipes
    for proc in processes:
        if proc.poll() is None:
            proc.kill()
        # communicate() waits and closes pipes properly
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()


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
    mock_dataset_info_endpoint(remote_dataset_single_version)
    mock_s3_parquet_download()

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
def test_read_dataset_remote_uses_local_when_cached(
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
    mock_dataset_info_endpoint(remote_dataset_single_version)
    mock_s3_parquet_download()

    ds1 = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        version="1.0.0",
        session=test_session,
    )

    assert ds1.to_values("version")[0] == "1.0.0"
    assert ds1.dataset.name == "dogs"
    assert dc.datasets().to_values("version") == ["1.0.0"]

    # Second read - should use local dataset without calling remote
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
    mock_dataset_info_endpoint(remote_dataset_multi_version)
    mock_s3_parquet_download()

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

    ds2 = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        version="1.0.0",
        update=True,
        session=test_session,
    )
    assert dc.datasets().to_values("version") == ["1.0.0"]
    assert ds2.to_values("version")[0] == "1.0.0"

    ds3 = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        version=">=1.0.0",
        update=False,
        session=test_session,
    )
    assert dc.datasets().to_values("version") == ["1.0.0"]
    assert ds3.to_values("version")[0] == "1.0.0"

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
    mock_dataset_info_endpoint(remote_dataset_multi_version)
    mock_s3_parquet_download()

    ds1 = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        version="1.0.0",
        session=test_session,
    )
    assert dc.datasets().to_values("version") == ["1.0.0"]
    assert ds1.to_values("version")[0] == "1.0.0"

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
    mock_dataset_info_endpoint(remote_dataset_multi_version)
    mock_s3_parquet_download()

    ds = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        version=">=1.0.0",
        session=test_session,
    )

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
    mock_dataset_info_endpoint(remote_dataset_multi_version)

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
def test_read_dataset_remote_latest_version_by_default(
    studio_token,
    test_session,
    remote_dataset_multi_version,
    mock_dataset_info_endpoint,
    mock_export_endpoint_with_urls,
    mock_export_status_completed,
    mock_s3_parquet_download,
    mock_dataset_rows_fetcher_status_check,
):
    mock_dataset_info_endpoint(remote_dataset_multi_version)
    mock_s3_parquet_download()

    ds = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        session=test_session,
    )

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
    mock_s3_parquet_download,
    mock_dataset_rows_fetcher_status_check,
    requests_mock,
):
    mock_dataset_info_endpoint(remote_dataset_single_version)
    mock_s3_parquet_download()

    # Mock failed export status
    requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/export-status",
        json={
            "status": "failed",
            "files_done": 0,
            "num_files": 1,
            "error_message": "Export failed",
        },
    )

    with pytest.raises(DataChainError) as exc_info:
        dc.read_dataset(
            f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
            version="1.0.0",
            session=test_session,
        )

    assert "Dataset export failed in Studio" in str(exc_info.value)

    _verify_cleanup_and_retry_success(
        test_session, requests_mock, mock_s3_parquet_download
    )


def _verify_cleanup_and_retry_success(
    test_session, requests_mock, mock_s3_parquet_download
):
    with pytest.raises(DatasetNotFoundError):
        test_session.catalog.get_dataset(
            "dogs",
            namespace_name=REMOTE_NAMESPACE_NAME,
            project_name=REMOTE_PROJECT_NAME,
            include_incomplete=True,
        )

    requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/export-status",
        json={
            "status": "completed",
            "files_done": 1,
            "num_files": 1,
        },
    )
    mock_s3_parquet_download()

    ds = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        version="1.0.0",
        session=test_session,
    )

    assert ds.to_values("version")[0] == "1.0.0"
    assert dc.datasets().to_values("version") == ["1.0.0"]


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_cleanup_on_download_failure(
    studio_token,
    test_session,
    remote_dataset_single_version,
    mock_dataset_info_endpoint,
    mock_export_endpoint_with_urls,
    mock_export_status_completed,
    mock_s3_parquet_download,
    mock_dataset_rows_fetcher_status_check,
    requests_mock,
):
    mock_dataset_info_endpoint(remote_dataset_single_version)

    requests_mock.get(
        "https://studio-blobvault.s3.amazonaws.com/datachain_ds_export_1_0_0.parquet.lz4",
        status_code=500,
        text="Server error",
    )

    with pytest.raises(requests.HTTPError):
        dc.read_dataset(
            f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
            version="1.0.0",
            session=test_session,
        )

    _verify_cleanup_and_retry_success(
        test_session, requests_mock, mock_s3_parquet_download
    )


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_cleanup_on_parse_failure(
    studio_token,
    test_session,
    remote_dataset_single_version,
    mock_dataset_info_endpoint,
    mock_export_endpoint_with_urls,
    mock_export_status_completed,
    mock_s3_parquet_download,
    mock_dataset_rows_fetcher_status_check,
    requests_mock,
):
    mock_dataset_info_endpoint(remote_dataset_single_version)

    requests_mock.get(
        "https://studio-blobvault.s3.amazonaws.com/datachain_ds_export_1_0_0.parquet.lz4",
        content=b"not valid parquet",
    )

    with pytest.raises(RuntimeError):
        dc.read_dataset(
            f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
            version="1.0.0",
            session=test_session,
        )

    _verify_cleanup_and_retry_success(
        test_session, requests_mock, mock_s3_parquet_download
    )


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_cleanup_on_insertion_failure(
    mocker,
    studio_token,
    test_session,
    remote_dataset_single_version,
    mock_dataset_info_endpoint,
    mock_export_endpoint_with_urls,
    mock_export_status_completed,
    mock_s3_parquet_download,
    mock_dataset_rows_fetcher_status_check,
    requests_mock,
):
    mock_dataset_info_endpoint(remote_dataset_single_version)
    mock_s3_parquet_download()

    # Mock the insert_dataframe_to_table method used in atomic pull flow
    mock_insert = mocker.patch(
        "datachain.data_storage.warehouse.AbstractWarehouse.insert_dataframe_to_table",
        side_effect=RuntimeError("Insert failed"),
    )

    with pytest.raises(RuntimeError, match="Insert failed"):
        dc.read_dataset(
            f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
            version="1.0.0",
            session=test_session,
        )

    mocker.stop(mock_insert)

    _verify_cleanup_and_retry_success(
        test_session, requests_mock, mock_s3_parquet_download
    )


def _wait_for_signal_file(proc, signal_file, timeout=30, poll_interval=0.05):
    """Wait for a subprocess to create a signal file.

    Raises AssertionError if the process exits early or the timeout expires.
    """
    deadline = time.time() + timeout
    while not signal_file.exists():
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise AssertionError(
                f"Subprocess exited early (code {proc.returncode})\n"
                f"stdout: {stdout.decode(errors='replace')}\n"
                f"stderr: {stderr.decode(errors='replace')}"
            )
        if time.time() >= deadline:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise AssertionError(
                f"Subprocess did not create signal file within {timeout}s\n"
                f"stdout: {stdout.decode(errors='replace')}\n"
                f"stderr: {stderr.decode(errors='replace')}"
            )
        time.sleep(poll_interval)


def _wait_for_subprocess(proc, timeout=60):
    """Wait for a subprocess to finish with timeout and diagnostic output.

    Returns (returncode, stdout, stderr). Raises AssertionError on timeout.
    """
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise AssertionError(
            f"Subprocess timed out after {timeout}s\n"
            f"stdout: {stdout.decode(errors='replace')}\n"
            f"stderr: {stderr.decode(errors='replace')}"
        ) from None
    return (
        proc.returncode,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_sigkill_then_retry_succeeds(
    tmp_path,
    run_script,
):
    signal_file = tmp_path / "ready_to_kill"

    # Start pull, hang when get_parquet_content is called (during download)
    proc = run_script(
        """
import datachain as dc
dc.read_dataset("dev.animals.dogs", version="1.0.0")
""",
        capture_output=True,
        hang_point=(
            "datachain.catalog.catalog",
            "DatasetRowsFetcher",
            "get_parquet_content",
        ),
        signal_file=signal_file,
    )

    _wait_for_signal_file(proc, signal_file)

    proc.kill()
    proc.communicate()

    # Retry - should succeed and produce correct data
    results_file = tmp_path / "results.json"
    retry = run_script(
        f"""
import json, datachain as dc
from pathlib import Path
ds = dc.read_dataset("dev.animals.dogs", version="1.0.0")
Path("{results_file.as_posix()}").write_text(json.dumps({{
    "row_count": len(ds.to_values("version")),
    "datasets": dc.datasets().to_values("version"),
}}))
""",
        capture_output=True,
    )
    rc, stdout, stderr = _wait_for_subprocess(retry)
    assert rc == 0, (
        f"Retry after crash should succeed with atomic pull\n"
        f"stdout: {stdout}\nstderr: {stderr}"
    )

    results = json.loads(results_file.read_text())
    assert results["row_count"] == 4, f"Expected 4 dog rows, got {results}"
    assert results["datasets"] == ["1.0.0"]


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_sigkill_during_insertion_then_retry_succeeds(
    tmp_path,
    run_script,
):
    signal_file = tmp_path / "ready_to_kill"

    # Start pull, hang when inserting data into temp table
    proc = run_script(
        """
import datachain as dc
dc.read_dataset("dev.animals.dogs", version="1.0.0")
""",
        capture_output=True,
        hang_point=(
            "datachain.data_storage.warehouse",
            "AbstractWarehouse",
            "insert_dataframe_to_table",
        ),
        signal_file=signal_file,
    )

    _wait_for_signal_file(proc, signal_file)

    proc.kill()
    proc.communicate()

    # Retry - orphaned tmp_ table should not block the new pull
    results_file = tmp_path / "results.json"
    retry = run_script(
        f"""
import json, datachain as dc
from pathlib import Path
ds = dc.read_dataset("dev.animals.dogs", version="1.0.0")
Path("{results_file.as_posix()}").write_text(json.dumps({{
    "row_count": len(ds.to_values("version")),
    "datasets": dc.datasets().to_values("version"),
}}))
""",
        capture_output=True,
    )
    rc, stdout, stderr = _wait_for_subprocess(retry)
    assert rc == 0, (
        f"Retry after crash during insertion should succeed\n"
        f"stdout: {stdout}\nstderr: {stderr}"
    )

    results = json.loads(results_file.read_text())
    assert results["row_count"] == 4, f"Expected 4 dog rows, got {results}"
    assert results["datasets"] == ["1.0.0"]


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_cleanup_on_update_dataset_status_fail(
    mocker,
    studio_token,
    test_session,
    remote_dataset_single_version,
    mock_dataset_info_endpoint,
    mock_export_endpoint_with_urls,
    mock_export_status_completed,
    mock_s3_parquet_download,
    mock_dataset_rows_fetcher_status_check,
    requests_mock,
):
    mock_dataset_info_endpoint(remote_dataset_single_version)
    mock_s3_parquet_download()

    catalog = test_session.catalog

    mock_status = mocker.patch(
        "datachain.data_storage.metastore.AbstractDBMetastore.update_dataset_status",
        side_effect=RuntimeError("Status update failed"),
    )

    with pytest.raises(RuntimeError, match="Status update failed"):
        dc.read_dataset(
            f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
            version="1.0.0",
            session=test_session,
        )

    # Temp tables should be gone (renamed to final name)
    assert catalog.warehouse.get_temp_table_names() == []

    # But incomplete metadata should exist (CREATED status, not COMPLETE)
    dataset = catalog.get_dataset(
        "dogs",
        namespace_name=REMOTE_NAMESPACE_NAME,
        project_name=REMOTE_PROJECT_NAME,
        include_incomplete=True,
    )
    assert dataset.get_version("1.0.0").status != DatasetStatus.COMPLETE

    # The final-name data table should exist (it was renamed from temp)
    final_table_name = catalog.warehouse.dataset_table_name(dataset, "1.0.0")
    from sqlalchemy import inspect as sa_inspect

    table_names = sa_inspect(catalog.warehouse.db.engine).get_table_names()
    assert final_table_name in table_names, (
        f"Expected final table {final_table_name} to exist after post-rename failure"
    )

    # Now retry — should clean up the incomplete version and succeed
    mocker.stop(mock_status)

    ds = dc.read_dataset(
        f"{REMOTE_NAMESPACE_NAME}.{REMOTE_PROJECT_NAME}.dogs",
        version="1.0.0",
        session=test_session,
    )

    assert ds.to_values("version")[0] == "1.0.0"
    assert len(ds.to_values("version")) == 4
    assert dc.datasets().to_values("version") == ["1.0.0"]


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_gc_cleans_orphaned_tmp_tables(
    tmp_path,
    run_script,
):
    signal_file = tmp_path / "ready_to_kill"

    proc = run_script(
        """
import datachain as dc
dc.read_dataset("dev.animals.dogs", version="1.0.0")
""",
        capture_output=True,
        hang_point=(
            "datachain.catalog.catalog",
            "DatasetRowsFetcher",
            "get_parquet_content",
        ),
        signal_file=signal_file,
    )

    _wait_for_signal_file(proc, signal_file)
    proc.kill()
    proc.communicate()

    # Verify orphaned tmp_ table exists, then run gc to clean it
    results_file = tmp_path / "gc_results.json"
    gc_proc = run_script(
        f"""
import json
from pathlib import Path
from datachain.catalog import get_catalog
from datachain.cli import main

catalog = get_catalog()
tmp_before = catalog.get_temp_table_names()
catalog.close()

main(["gc"])

catalog = get_catalog()
tmp_after = catalog.get_temp_table_names()
catalog.close()

Path("{results_file.as_posix()}").write_text(json.dumps({{
    "tmp_before": tmp_before,
    "tmp_after": tmp_after,
}}))
""",
        capture_output=True,
    )
    rc, stdout, stderr = _wait_for_subprocess(gc_proc)
    assert rc == 0, f"gc subprocess failed\nstdout: {stdout}\nstderr: {stderr}"

    results = json.loads(results_file.read_text())
    assert len(results["tmp_before"]) > 0, "Expected orphaned tmp_ table after SIGKILL"
    assert results["tmp_after"] == [], "gc should have cleaned all tmp_ tables"


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_concurrent_pulls(
    tmp_path,
    run_script,
):
    results_a = tmp_path / "results_a.json"
    results_b = tmp_path / "results_b.json"

    script_template = """
import json, datachain as dc
from pathlib import Path
ds = dc.read_dataset("dev.animals.dogs", version="1.0.0")
Path("{results_file}").write_text(json.dumps({{
    "row_count": len(ds.to_values("version")),
    "datasets": dc.datasets().to_values("version"),
}}))
"""

    proc_a = run_script(
        script_template.format(results_file=results_a.as_posix()),
        capture_output=True,
    )
    proc_b = run_script(
        script_template.format(results_file=results_b.as_posix()),
        capture_output=True,
    )

    rc_a, stdout_a, stderr_a = _wait_for_subprocess(proc_a)
    rc_b, stdout_b, stderr_b = _wait_for_subprocess(proc_b)

    # Both processes must succeed
    assert rc_a == 0, (
        f"Process A failed (rc={rc_a}):\n  stdout: {stdout_a}\n  stderr: {stderr_a}"
    )
    assert rc_b == 0, (
        f"Process B failed (rc={rc_b}):\n  stdout: {stdout_b}\n  stderr: {stderr_b}"
    )

    # Verify both produced correct data
    for name, path in [("A", results_a), ("B", results_b)]:
        results = json.loads(path.read_text())
        assert results["row_count"] == 4, (
            f"Process {name}: expected 4 rows, got {results}"
        )
        assert results["datasets"] == ["1.0.0"], (
            f"Process {name}: unexpected datasets {results['datasets']}"
        )
