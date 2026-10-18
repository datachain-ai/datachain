import pytest
from moto.s3.responses import S3Response

import datachain as dc
from datachain.client import bucket_status
from datachain.client.s3 import ClientS3

NOT_IMPLEMENTED = (
    b'<?xml version="1.0" encoding="UTF-8"?><Error><Code>NotImplemented</Code>'
    b"<Message>ListObjectVersions not implemented</Message></Error>"
)


def _signed(request) -> bool:
    return "Authorization" in request.headers or "X-Amz-Signature" in request.url


@pytest.fixture
def r2_like(monkeypatch):
    """Answer like Cloudflare R2: 400 to unsigned requests, no ListObjectVersions."""
    bucket_response = S3Response._bucket_response
    key_response = S3Response._key_response

    def _bucket(self, request, full_url):
        if not _signed(request):
            return 400, {}, b""
        if request.method == "GET" and "versions" in self.querystring:
            return 501, {"Content-Type": "application/xml"}, NOT_IMPLEMENTED
        return bucket_response(self, request, full_url)

    def _key(self, request, full_url):
        if not _signed(request):
            return 400, {}, b""
        return key_response(self, request, full_url)

    monkeypatch.setattr(S3Response, "_bucket_response", _bucket)
    monkeypatch.setattr(S3Response, "_key_response", _key)


def _config(cloud_server) -> dict:
    # credentials from the environment, like an R2 profile: the probe runs
    return {
        k: v
        for k, v in cloud_server.client_config.items()
        if k not in ClientS3.CREDENTIAL_KEYS
    }


@pytest.mark.parametrize("cloud_type,version_aware", [("s3", False)], indirect=True)
def test_bucket_status_anonymous_bad_request(
    cloud_server, cloud_server_credentials, r2_like
):
    status = bucket_status(cloud_server.src_uri, **_config(cloud_server))
    assert (status.exists, status.access) == (True, "authenticated")


@pytest.mark.parametrize("cloud_type,version_aware", [("s3", False)], indirect=True)
@pytest.mark.parametrize("versions", [True, False])
def test_read_storage_store_without_versions(
    cloud_server, cloud_server_credentials, r2_like, versions
):
    chain = dc.read_storage(
        cloud_server.src_uri,
        client_config=_config(cloud_server),
        versions=versions,
        update=True,
    )
    assert "anon" not in chain.session.catalog.client_config
    files = chain.order_by("file.path").to_values("file")
    assert [f.path for f in files] == [
        "cats/cat1",
        "cats/cat2",
        "description",
        "dogs/dog1",
        "dogs/dog2",
        "dogs/dog3",
        "dogs/others/dog4",
    ]
    assert {f.version for f in files} == {""}
    assert all(f.is_latest for f in files)
    assert files[0].read_text() == "meow"


@pytest.mark.parametrize("cloud_type,version_aware", [("s3", False)], indirect=True)
def test_read_storage_store_without_versions_single_file(
    cloud_server, cloud_server_credentials, r2_like
):
    uri = f"{cloud_server.src_uri.rstrip('/')}/description"
    chain = dc.read_storage(uri, client_config=_config(cloud_server), update=True)
    (file,) = chain.to_values("file")
    assert file.path == "description"
    assert file.read_text() == "Cats and Dogs"
