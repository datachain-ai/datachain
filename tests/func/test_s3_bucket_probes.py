import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fsspec.asyn import get_loop, sync
from s3fs import S3FileSystem

import datachain as dc
from datachain.client import s3
from datachain.client.s3 import ClientS3

_CREDENTIALS = {"key": "synthetic-key", "secret": "synthetic-secret"}


@pytest.fixture(autouse=True)
def isolated_aws(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    config = tmp_path / "config"
    config.write_text("")
    credentials = tmp_path / "credentials"
    credentials.write_text("")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setattr(s3, "_BUCKET_REGIONS", {})
    classes = (S3FileSystem, s3._S3FileSystem)
    existing = {cls: set(cls._cache) for cls in classes}
    yield
    # Async filesystems are owned by their caller and have no GC finalizer.
    for cls in classes:
        for key in set(cls._cache) - existing[cls]:
            fs = cls._cache.pop(key)
            if creator := getattr(fs, "_s3creator", None):
                sync(get_loop(), creator.__aexit__, None, None, None)


@pytest.fixture
def endpoints():
    servers = []

    def start(region="us-east-1", statuses=(200, 200)):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_HEAD(self):
                signed = bool(self.headers.get("Authorization"))
                requests.append(("HEAD", signed))
                self.send_response(statuses[int(signed)])
                if statuses[int(signed)] == 200:
                    self.send_header("x-amz-bucket-region", region)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                auth = self.headers.get("Authorization", "")
                requests.append(("GET", bool(auth)))
                correct = not auth or f"/{region}/s3/aws4_request" in auth
                body = (
                    b"<ListBucketResult><Name>bucket</Name><IsTruncated>false</IsTruncated>"
                    b'<Contents><Key>a.txt</Key><ETag>"abc"</ETag><Size>1</Size>'
                    b"<LastModified>2026-01-01T00:00:00Z</LastModified></Contents>"
                    b"</ListBucketResult>"
                    if correct
                    else b"<Error><Code>SignatureDoesNotMatch</Code></Error>"
                )
                self.send_response(200 if correct else 403)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
        )
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}", requests

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_async_filesystem_supports_probe_and_read_storage(
    endpoints, test_session, monkeypatch, asynchronous
):
    endpoint, requests = endpoints()
    config = {
        "endpoint_url": endpoint,
        "asynchronous": asynchronous,
    }
    monkeypatch.setattr("datachain.client.listing_worker.LIST_PROCESSES", 0)

    assert ClientS3.bucket_status("bucket", **config).access == "anonymous"
    assert (
        dc.read_storage(
            "s3://bucket/", client_config=config, versions=False, session=test_session
        ).count()
        == 1
    )
    assert any(method == "GET" for method, _ in requests)


@pytest.mark.parametrize("profile", [None, "review"])
@pytest.mark.parametrize("services", [False, True])
@pytest.mark.parametrize("rewrite", [False, True], ids=["switch-file", "rewrite-file"])
def test_shared_config_endpoint_does_not_reuse_another_stores_region(
    endpoints, tmp_path, monkeypatch, profile, services, rewrite
):
    first, _ = endpoints("us-west-1")
    second, requests = endpoints("eu-west-2")
    section = "default" if profile is None else f"profile {profile}"

    def configure(endpoint, name):
        path = tmp_path / name
        if services:
            body = (
                f"[{section}]\nservices = local\n"
                f"[services local]\ns3 =\n  endpoint_url = {endpoint}\n"
            )
        else:
            body = f"[{section}]\nendpoint_url = {endpoint}\n"
        path.write_text(body)
        monkeypatch.setenv("AWS_CONFIG_FILE", str(path))

    kwargs = {"skip_instance_cache": True}
    if profile:
        kwargs["profile"] = profile
    configure(first, "first")
    assert ClientS3.bucket_status("bucket", **kwargs).access == "anonymous"
    assert not s3._BUCKET_REGIONS

    # A prior AWS probe can have filled this entry before an endpoint was set.
    s3._BUCKET_REGIONS["bucket", profile, None] = "us-west-1"
    configure(second, "first" if rewrite else "second")
    client = ClientS3.from_name(
        "bucket", None, kwargs | _CREDENTIALS | {"version_aware": False}
    )
    assert "region_name" not in client.fs_kwargs
    page = sync(get_loop(), client._list_page, "", "", None)
    assert len(page.rows) == 1
    assert ("GET", True) in requests


@pytest.mark.parametrize("services", [False, True])
@pytest.mark.parametrize("path_style", ["absolute", "environment", "relative"])
def test_adding_shared_endpoint_stops_reusing_a_cached_aws_region(
    endpoints, tmp_path, monkeypatch, services, path_style
):
    path = tmp_path / "shared-config"
    path.write_text("[default]\n")
    if path_style == "environment":
        monkeypatch.setenv("DATACHAIN_TEST_AWS_CONFIG_DIR", str(tmp_path))
        config_path = "${DATACHAIN_TEST_AWS_CONFIG_DIR}/shared-config"
    elif path_style == "relative":
        monkeypatch.chdir(tmp_path)
        config_path = "shared-config"
    else:
        config_path = str(path)
    monkeypatch.setenv("AWS_CONFIG_FILE", config_path)
    s3._BUCKET_REGIONS["bucket", None, None] = "us-west-1"
    kwargs = _CREDENTIALS | {"version_aware": False, "skip_instance_cache": True}
    assert (
        ClientS3.from_name("bucket", None, kwargs).fs_kwargs["region_name"]
        == "us-west-1"
    )

    endpoint, requests = endpoints("eu-west-2")
    if services:
        body = (
            "[default]\nservices = local\n[services local]\n"
            f"s3 =\n  endpoint_url = {endpoint}\n"
        )
    else:
        body = f"[default]\nendpoint_url = {endpoint}\n"
    path.write_text(body)
    client = ClientS3.from_name("bucket", None, kwargs)
    page = sync(get_loop(), client._list_page, "", "", None)
    assert len(page.rows) == 1
    assert ("GET", True) in requests


@pytest.mark.parametrize("statuses", [(429, 429), (403, 429), (403, 400)])
def test_probe_preserves_service_errors(endpoints, statuses):
    endpoint, requests = endpoints(statuses=statuses)
    with pytest.raises(OSError, match="429" if statuses[-1] == 429 else "Bad Request"):
        ClientS3.bucket_status(
            "bucket",
            endpoint_url=endpoint,
            **_CREDENTIALS,
            config_kwargs={"retries": {"max_attempts": 0}},
            skip_instance_cache=True,
        )
    assert len(requests) == (1 if statuses[0] == 429 else 2)


def test_r2_anonymous_bad_request_still_allows_authenticated_probe(endpoints):
    endpoint, requests = endpoints(statuses=(400, 200))
    status = ClientS3.bucket_status(
        "bucket", endpoint_url=endpoint, **_CREDENTIALS, skip_instance_cache=True
    )
    assert status.access == "authenticated"
    assert requests == [("HEAD", False), ("HEAD", True)]
