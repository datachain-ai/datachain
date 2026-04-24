import pytest

from datachain.client import BucketStatus, bucket_status


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
