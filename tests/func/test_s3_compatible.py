import pytest
from moto.s3.responses import S3Response

import datachain as dc
from datachain.client import bucket_status
from tests.data import ENTRIES

pytestmark = pytest.mark.parametrize(
    "cloud_type,version_aware", [("s3", False)], indirect=True
)


@pytest.fixture
def r2_config(cloud_server, cloud_server_credentials, monkeypatch):
    bucket_response = S3Response._bucket_response

    def response(self, request, full_url):
        if "Authorization" not in request.headers:
            return 400, {}, b""
        if "versions" in self.querystring:
            return 501, {}, b"<Error><Code>NotImplemented</Code></Error>"
        return bucket_response(self, request, full_url)

    monkeypatch.setattr(S3Response, "_bucket_response", response)
    monkeypatch.setenv("MOTO_S3_DEFAULT_MAX_KEYS", "2")
    return {"aws_endpoint_url": cloud_server.client_config["aws_endpoint_url"]}


def test_bucket_status_anonymous_bad_request(cloud_server, r2_config):
    status = bucket_status(cloud_server.src_uri, **r2_config)
    assert (status.exists, status.access) == (True, "authenticated")


def test_read_storage_store_without_versions(cloud_server, r2_config):
    chain = dc.read_storage(cloud_server.src_uri, client_config=r2_config, update=True)
    files = chain.order_by("file.path").to_values("file")
    assert [f.path for f in files] == sorted(f.path for f in ENTRIES)
    assert {f.version for f in files} == {""}
    assert all(f.is_latest for f in files)
    assert files[0].read_text() == "meow"
