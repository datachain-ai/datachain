import http.server
import os
import signal
import socket
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

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(*_):
            return None

        def do_GET(self):
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

    server = socketserver.TCPServer(("", port), Handler)
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

    def _run_script(script):
        script_file = tmp_path / "worker.py"
        script_file.write_text(script)
        env = {
            "PYTHONPATH": f"{project_root}/src",
            "DATACHAIN_ROOT_DIR": str(tmp_path),
            "DATACHAIN_STUDIO_URL": f"http://localhost:{port}",
            "DATACHAIN_STUDIO_TOKEN": "test",
            "DATACHAIN_STUDIO_TEAM": "test-team",
        }
        return subprocess.Popen(  # noqa: S603
            [sys.executable, str(script_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

    return _run_script


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

    mock_insert = mocker.patch(
        "datachain.data_storage.sqlite.SQLiteWarehouse.insert_dataset_rows",
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


@skip_if_not_sqlite
@pytest.mark.parametrize("is_studio", (False,))
def test_read_dataset_remote_sigkill_then_retry_succeeds(
    tmp_path,
    run_script,
):
    signal_file = tmp_path / "ready_to_kill"

    # Start pull, signal when downloading, then hang
    proc = run_script(f"""
from pathlib import Path
from unittest.mock import patch
import time

def hang_on_download(self, url):
    Path("{signal_file}").touch()
    while True:
        time.sleep(1)

import datachain.catalog.catalog as catalog_module

with patch.object(
    catalog_module.DatasetRowsFetcher,
    "get_parquet_content",
    hang_on_download,
):
    import datachain as dc
    dc.read_dataset("dev.animals.dogs", version="1.0.0")
""")

    deadline = time.time() + 10
    while not signal_file.exists():
        assert time.time() < deadline, "Child never reached download point"
        time.sleep(0.01)

    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()

    # Retry - should succeed with atomic pull (no partial state blocking)
    proc = run_script("""
import datachain as dc
dc.read_dataset("dev.animals.dogs", version="1.0.0")
""")
    assert proc.wait() == 0, "Retry after crash should succeed with atomic pull"
