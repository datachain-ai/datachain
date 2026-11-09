import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from aiobotocore.session import AioSession
from botocore.exceptions import ClientError, PartialCredentialsError
from fsspec.asyn import get_loop, sync
from s3fs import S3FileSystem

import datachain as dc
from datachain.client import s3
from datachain.client.fsspec import BucketStatus
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
    monkeypatch.setenv("BOTO_CONFIG", str(tmp_path / "boto"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setattr(s3, "_BUCKET_REGIONS", {})
    monkeypatch.setattr(s3, "_CREDENTIAL_CHECKS", {})
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

    def start(region="us-east-1", statuses=(200, 200), redirect_once=False):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_HEAD(self):
                signed = bool(self.headers.get("Authorization"))
                requests.append(("HEAD", signed))
                status = (
                    301
                    if redirect_once and len(requests) == 1
                    else statuses[int(signed)]
                )
                self.send_response(status)
                if status in (200, 301):
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
@pytest.mark.parametrize("provided_session", [False, True])
def test_async_filesystem_supports_probe_and_read_storage(
    endpoints, test_session, monkeypatch, asynchronous, provided_session
):
    endpoint, requests = endpoints()
    config = {
        "endpoint_url": endpoint,
        "asynchronous": asynchronous,
    }
    if provided_session:
        config["session"] = AioSession()
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


@pytest.mark.parametrize(
    "statuses,credentials,expected,signed",
    [
        ((200, 200), {}, BucketStatus(True, "anonymous"), [False]),
        (
            (404, 404),
            _CREDENTIALS,
            BucketStatus(False, "denied", "S3 bucket 'bucket' not found"),
            [False],
        ),
        ((403, 200), _CREDENTIALS, BucketStatus(True, "authenticated"), [False, True]),
        ((400, 200), _CREDENTIALS, BucketStatus(True, "authenticated"), [False, True]),
        (
            (403, 403),
            {},
            BucketStatus(
                True,
                "denied",
                "Access denied to S3 bucket 'bucket'"
                " — check AWS credentials/permissions",
            ),
            [False],
        ),
        (
            (403, 403),
            _CREDENTIALS,
            BucketStatus(
                True,
                "denied",
                "Access denied to S3 bucket 'bucket'"
                " — check AWS credentials/permissions",
            ),
            [False, True],
        ),
        (
            (403, 404),
            _CREDENTIALS,
            BucketStatus(False, "denied", "S3 bucket 'bucket' not found"),
            [False, True],
        ),
    ],
    ids=[
        "public",
        "missing",
        "authenticated",
        "r2",
        "no-credentials",
        "denied",
        "authenticated-missing",
    ],
)
def test_bucket_status(endpoints, statuses, credentials, expected, signed):
    endpoint, seen = endpoints(statuses=statuses)
    status = ClientS3.bucket_status(
        "bucket", endpoint_url=endpoint, skip_instance_cache=True, **credentials
    )
    assert status == expected
    assert seen == [("HEAD", value) for value in signed]


def test_probe_follows_a_regional_redirect(endpoints):
    endpoint, requests = endpoints(region="eu-west-2", redirect_once=True)
    status = ClientS3.bucket_status(
        "bucket",
        endpoint_url=endpoint,
        region_name="us-east-1",
        skip_instance_cache=True,
    )
    assert status == BucketStatus(True, "anonymous")
    assert requests == [("HEAD", False), ("HEAD", False)]


@pytest.mark.parametrize("statuses", [(301, 200), (403, 301)])
def test_probe_preserves_an_unresolved_redirect(endpoints, statuses):
    endpoint, requests = endpoints(region="eu-west-2", statuses=statuses)
    with pytest.raises(ClientError) as exc:
        ClientS3.bucket_status(
            "bucket", endpoint_url=endpoint, **_CREDENTIALS, skip_instance_cache=True
        )
    assert exc.value.response["ResponseMetadata"]["HTTPStatusCode"] == 301
    assert (
        exc.value.response["ResponseMetadata"]["HTTPHeaders"]["x-amz-bucket-region"]
        == "eu-west-2"
    )
    assert requests[-1] == ("HEAD", statuses[0] != 301)


@pytest.mark.parametrize("in_loop", [False, True])
@pytest.mark.parametrize("session", ["default", "none", "credentialless"])
def test_filesystem_without_credentials_is_anonymous(in_loop, session):
    async def create():
        fs = ClientS3.create_fs(cache_regions=False, **kwargs)
        await fs.set_session()
        return fs

    kwargs = {"skip_instance_cache": True}
    if session != "default":
        kwargs["session"] = AioSession() if session == "credentialless" else None
    if in_loop:
        fs = sync(get_loop(), create)
    else:
        fs = ClientS3.create_fs(cache_regions=False, **kwargs)
        fs.connect()
    assert fs.anon


def test_credential_discovery_does_not_block_the_filesystem_loop(monkeypatch):
    loop_thread = None

    def credentials(self):
        assert threading.get_ident() != loop_thread

    async def connect():
        nonlocal loop_thread
        loop_thread = threading.get_ident()
        await ClientS3.create_fs(
            cache_regions=False, skip_instance_cache=True
        ).set_session()

    monkeypatch.setattr(s3.Session, "get_credentials", credentials)
    sync(get_loop(), connect)


@pytest.mark.parametrize("source", ["session", "kwargs", "client_kwargs"])
def test_supplied_session_preserves_explicit_credentials(source):
    session = AioSession()
    kwargs = {}
    if source == "session":
        session.set_credentials("key", "secret")
    elif source == "kwargs":
        kwargs = {"key": "key", "secret": "secret"}
    else:
        kwargs = {
            "client_kwargs": {
                "aws_access_key_id": "key",
                "aws_secret_access_key": "secret",
            }
        }
    fs = ClientS3.create_fs(
        session=session, cache_regions=False, skip_instance_cache=True, **kwargs
    )
    fs.connect()
    assert not fs.anon
    assert fs.s3._request_signer._credentials.access_key == "key"


@pytest.mark.parametrize("explicit_anon", [False, True])
def test_refresh_rechecks_supplied_session_credentials(explicit_anon):
    session = AioSession()
    fs = ClientS3.create_fs(
        session=session,
        anon=explicit_anon,
        cache_regions=False,
        skip_instance_cache=True,
    )
    fs.connect()
    assert fs.anon
    session.set_credentials("key", "secret")
    fs.connect(refresh=True)
    assert fs.anon is explicit_anon
    if not explicit_anon:
        assert fs.s3._request_signer._credentials.access_key == "key"


def test_refresh_rechecks_default_credentials(monkeypatch):
    fs = ClientS3.create_fs(cache_regions=False, skip_instance_cache=True)
    fs.connect()
    assert fs.anon
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    fs.connect(refresh=True)
    assert not fs.anon
    assert fs.s3._request_signer._credentials.access_key == "key"


@pytest.mark.parametrize("kwargs", [{}, {"session": None}])
def test_shared_config_region_does_not_reuse_another_partitions_bucket(kwargs):
    path = Path(os.environ["AWS_CONFIG_FILE"])
    path.write_text("[default]\nregion = us-west-2\n")
    key = s3._region_key("bucket", {})
    assert key is not None
    s3._BUCKET_REGIONS[key] = "us-west-2"
    client = ClientS3.from_name("bucket", None, _CREDENTIALS | kwargs)
    assert client.fs_kwargs["region_name"] == "us-west-2"
    path.write_text("[default]\nregion = cn-north-1\n")
    client = ClientS3.from_name("bucket", None, _CREDENTIALS | kwargs)
    assert client.fs_kwargs.get("region_name") != "us-west-2"


@pytest.mark.parametrize(
    "credentials",
    [
        {"username": "key", "password": "secret"},
        _CREDENTIALS,
        {
            "client_kwargs": {
                "aws_access_key_id": "key",
                "aws_secret_access_key": "secret",
            }
        },
    ],
)
def test_filesystem_creation_does_not_send_s3_requests(monkeypatch, credentials):
    requests = []

    async def unexpected(*args, **kwargs):
        requests.append((args, kwargs))

    monkeypatch.setattr("aiobotocore.endpoint.AioEndpoint.make_request", unexpected)
    fs = ClientS3.create_fs(
        cache_regions=False, skip_instance_cache=True, **credentials
    )
    fs.connect()
    assert not fs.anon
    assert not requests


@pytest.mark.parametrize(
    "credentials",
    [
        {"secret": "secret"},
        {"aws_secret": "secret"},
        {"password": "secret"},
        {"client_kwargs": {"aws_secret_access_key": "secret"}},
    ],
)
def test_partial_credentials_do_not_fall_back_to_anonymous(credentials):
    with pytest.raises(PartialCredentialsError):
        fs = ClientS3.create_fs(
            cache_regions=False, skip_instance_cache=True, **credentials
        )
        fs.sign("s3://bucket/object")


@pytest.mark.parametrize("source", ["environment", "credentials-file"])
def test_filesystem_rechecks_changed_credentials(monkeypatch, source):
    fs = ClientS3.create_fs(cache_regions=False)
    fs.connect()
    assert fs.anon
    if source == "environment":
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    else:
        Path(os.environ["AWS_SHARED_CREDENTIALS_FILE"]).write_text(
            "[default]\naws_access_key_id = key\naws_secret_access_key = secret\n"
        )
    fs = ClientS3.create_fs(cache_regions=False)
    fs.connect()
    assert not fs.anon
    assert ClientS3.create_fs(cache_regions=False) is fs


def test_missing_credentials_are_rechecked_after_ttl(monkeypatch, mocker):
    probe = mocker.patch("datachain.client.s3.Session")
    probe.return_value.get_credentials.side_effect = [None, object()]
    now = mocker.patch("datachain.client.s3.monotonic", return_value=59.0)

    def connect():
        fs = ClientS3.create_fs(cache_regions=False)
        fs.connect()
        return fs

    assert connect().anon
    now.return_value = 60.5
    assert connect().anon
    now.return_value = 119.5
    assert not connect().anon
    assert probe.call_count == 2
