import aiohttp
import pytest
from yarl import URL

from datachain.client import BucketStatus, bucket_status
from datachain.client.s3 import ClientS3


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
def test_bucket_status_exists(cloud_server, cloud_server_credentials):
    uri = cloud_server.src_uri
    status = bucket_status(uri, **cloud_server.client_config)
    assert isinstance(status, BucketStatus)
    assert status.exists is True
    assert status.access in ("anonymous", "authenticated")


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
def test_bucket_status_not_found(cloud_server, cloud_server_credentials):
    base_uri = cloud_server.src_uri.rstrip("/")
    nonexistent_uri = base_uri + "-does-not-exist-xyz/"
    status = bucket_status(nonexistent_uri, **cloud_server.client_config)
    assert isinstance(status, BucketStatus)
    assert status.exists is False
    assert status.access == "denied"


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
def test_anonymous_probe_uses_the_profiles_endpoint(
    cloud_server, tmp_path, monkeypatch
):
    endpoint = cloud_server.client_config["aws_endpoint_url"]
    config = tmp_path / "profiles"
    config.write_text(
        f"[profile store]\nendpoint_url = {endpoint}\nregion = us-east-1\n"
    )
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    for name in ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3"):
        monkeypatch.delenv(name, raising=False)
    seen = []
    request = aiohttp.ClientSession._request

    async def record(self, method, url, **kwargs):
        assert URL(url).origin() == URL(endpoint).origin()
        seen.append((method, bool(kwargs.get("headers", {}).get("Authorization"))))
        return await request(self, method, url, **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, "_request", record)
    name, _ = ClientS3.split_url(cloud_server.src_uri)
    status = ClientS3.bucket_status(name, profile="store", key="key")

    assert status.access == "anonymous"
    assert seen and all(request == ("HEAD", False) for request in seen)
