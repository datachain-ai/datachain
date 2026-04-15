from unittest.mock import MagicMock, patch

import pytest

from datachain.client import bucket_status
from datachain.client.azure import AzureClient
from datachain.client.fsspec import BucketStatus
from datachain.client.gcs import GCSClient
from datachain.client.s3 import ClientS3


@pytest.mark.parametrize(
    "uri",
    [
        "s3://my-bucket/some/path",
        "gs://my-bucket/dir",
        "az://my-container/blob",
    ],
)
def test_bucket_status_rejects_path_component(uri):
    with pytest.raises(ValueError, match="path in a bucket is not allowed"):
        bucket_status(uri)


@patch("datachain.client.s3.S3FileSystem")
@patch("datachain.client.s3.sync")
def test_s3_anon_only_kwarg_anonymous(mock_sync, mock_s3fs_cls):
    anon_fs = MagicMock()
    anon_fs._info.return_value = {"name": "my-bucket", "type": "directory"}
    mock_s3fs_cls.return_value = anon_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = ClientS3.bucket_status("my-bucket", anon=True)

    assert result == BucketStatus(exists=True, access="anonymous")


@patch("datachain.client.s3.S3FileSystem")
@patch("datachain.client.s3.sync")
def test_s3_anon_only_kwarg_denied(mock_sync, mock_s3fs_cls):
    anon_fs = MagicMock()
    anon_fs._info.side_effect = PermissionError("AccessDenied")
    mock_s3fs_cls.return_value = anon_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = ClientS3.bucket_status("my-bucket", anon=True)

    assert result.exists is True
    assert result.access == "denied"
    assert result.error is not None


@patch.object(GCSClient, "create_fs")
@patch("datachain.client.gcs.sync")
def test_gcs_anonymous(mock_sync, mock_create_fs):
    anon_fs = MagicMock()
    anon_fs._ls.return_value = []
    mock_create_fs.return_value = anon_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = GCSClient.bucket_status("my-bucket")

    assert result == BucketStatus(exists=True, access="anonymous")


@patch.object(GCSClient, "create_fs")
@patch("datachain.client.gcs.sync")
def test_gcs_authenticated_via_http_error(mock_sync, mock_create_fs):
    from gcsfs.retry import HttpError

    anon_fs = MagicMock()
    anon_fs._ls.side_effect = HttpError({"code": 401, "message": "Permission denied"})

    auth_fs = MagicMock()
    auth_fs._info.return_value = {"name": "my-bucket", "type": "directory"}

    def make_fs(**kwargs):
        return anon_fs if kwargs.get("anon") else auth_fs

    mock_create_fs.side_effect = make_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = GCSClient.bucket_status("my-bucket")

    assert result == BucketStatus(exists=True, access="authenticated")


@patch.object(GCSClient, "create_fs")
@patch("datachain.client.gcs.sync")
def test_gcs_authenticated(mock_sync, mock_create_fs):
    anon_fs = MagicMock()
    anon_fs._ls.side_effect = PermissionError("403")

    auth_fs = MagicMock()
    auth_fs._info.return_value = {"name": "my-bucket", "type": "directory"}

    def make_fs(**kwargs):
        return anon_fs if kwargs.get("anon") else auth_fs

    mock_create_fs.side_effect = make_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = GCSClient.bucket_status("my-bucket")

    assert result == BucketStatus(exists=True, access="authenticated")


@patch.object(GCSClient, "create_fs")
@patch("datachain.client.gcs.sync")
def test_gcs_denied(mock_sync, mock_create_fs):
    anon_fs = MagicMock()
    anon_fs._ls.side_effect = PermissionError("403")

    auth_fs = MagicMock()
    auth_fs._info.side_effect = PermissionError("403")

    def make_fs(**kwargs):
        return anon_fs if kwargs.get("anon") else auth_fs

    mock_create_fs.side_effect = make_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = GCSClient.bucket_status("my-bucket")

    assert result.exists is True
    assert result.access == "denied"
    assert result.error is not None


@patch.object(GCSClient, "create_fs")
@patch("datachain.client.gcs.sync")
def test_gcs_not_found(mock_sync, mock_create_fs):
    anon_fs = MagicMock()
    anon_fs._ls.side_effect = FileNotFoundError("bucket not found")
    mock_create_fs.return_value = anon_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = GCSClient.bucket_status("my-bucket")

    assert result.exists is False
    assert result.access == "denied"
    assert result.error is not None


@patch("datachain.client.azure.BlobServiceClient")
def test_azure_anonymous(mock_blob_svc_cls):
    mock_container = MagicMock()
    mock_container.get_container_properties.return_value = {"name": "my-container"}
    mock_client = MagicMock()
    mock_client.get_container_client.return_value = mock_container
    mock_blob_svc_cls.return_value = mock_client

    result = AzureClient.bucket_status("my-container", account_name="my-account")

    assert result == BucketStatus(exists=True, access="anonymous")
    mock_blob_svc_cls.assert_called_once_with(
        account_url="https://my-account.blob.core.windows.net"
    )


@patch.object(AzureClient, "create_fs")
@patch("datachain.client.azure.sync")
@patch("datachain.client.azure.BlobServiceClient")
def test_azure_authenticated_via_client_auth_error(
    mock_blob_svc_cls, mock_sync, mock_create_fs
):
    """Anon probe gets HTTP 401 (ClientAuthenticationError), falls through to auth."""
    from azure.core.exceptions import ClientAuthenticationError

    mock_container = MagicMock()
    mock_container.get_container_properties.side_effect = ClientAuthenticationError(
        "NoAuthenticationInformation"
    )
    mock_client = MagicMock()
    mock_client.get_container_client.return_value = mock_container
    mock_blob_svc_cls.return_value = mock_client

    auth_fs = MagicMock()
    auth_fs._info.return_value = {"name": "my-container", "type": "directory"}
    mock_create_fs.return_value = auth_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = AzureClient.bucket_status("my-container", account_name="my-account")

    assert result == BucketStatus(exists=True, access="authenticated")


@patch.object(AzureClient, "create_fs")
@patch("datachain.client.azure.sync")
def test_azure_authenticated_no_account_name(mock_sync, mock_create_fs):
    """No account_name: anon probe skipped, auth probe succeeds."""
    auth_fs = MagicMock()
    auth_fs._info.return_value = {"name": "my-container", "type": "directory"}
    mock_create_fs.return_value = auth_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = AzureClient.bucket_status("my-container")

    assert result == BucketStatus(exists=True, access="authenticated")


@patch.object(AzureClient, "create_fs")
@patch("datachain.client.azure.sync")
@patch("datachain.client.azure.BlobServiceClient")
def test_azure_denied(mock_blob_svc_cls, mock_sync, mock_create_fs):
    from azure.core.exceptions import ClientAuthenticationError

    mock_container = MagicMock()
    mock_container.get_container_properties.side_effect = ClientAuthenticationError(
        "NoAuth"
    )
    mock_client = MagicMock()
    mock_client.get_container_client.return_value = mock_container
    mock_blob_svc_cls.return_value = mock_client

    auth_fs = MagicMock()
    auth_fs._info.side_effect = PermissionError("403")
    mock_create_fs.return_value = auth_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = AzureClient.bucket_status("my-container", account_name="my-account")

    assert result.exists is True
    assert result.access == "denied"
    assert result.error is not None


@patch("datachain.client.azure.BlobServiceClient")
def test_azure_not_found(mock_blob_svc_cls):
    from azure.core.exceptions import ResourceNotFoundError

    mock_container = MagicMock()
    mock_container.get_container_properties.side_effect = ResourceNotFoundError(
        "container not found"
    )
    mock_client = MagicMock()
    mock_client.get_container_client.return_value = mock_container
    mock_blob_svc_cls.return_value = mock_client

    result = AzureClient.bucket_status("my-container", account_name="my-account")

    assert result.exists is False
    assert result.access == "denied"
    assert result.error is not None


@patch.object(AzureClient, "create_fs")
@patch("datachain.client.azure.sync")
def test_azure_no_account_name_no_creds(mock_sync, mock_create_fs):
    """No account_name: anon skipped, auth raises ValueError."""
    mock_create_fs.side_effect = ValueError(
        "Must provide either a connection_string or account_name with credentials!!"
    )
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = AzureClient.bucket_status("my-container")

    assert result.exists is False
    assert result.access == "denied"
    assert result.error is not None


@patch.object(AzureClient, "create_fs")
@patch("datachain.client.azure.sync")
@patch("datachain.client.azure.BlobServiceClient")
def test_azure_public_with_incompatible_creds(
    mock_blob_svc_cls, mock_sync, mock_create_fs
):
    """Public container: anon succeeds even when env creds are incompatible."""
    mock_container = MagicMock()
    mock_container.get_container_properties.return_value = {"name": "my-container"}
    mock_client = MagicMock()
    mock_client.get_container_client.return_value = mock_container
    mock_blob_svc_cls.return_value = mock_client

    # Auth probe would fail, but anon probe succeeds first.
    mock_create_fs.side_effect = PermissionError("Wrong account credentials")
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = AzureClient.bucket_status("my-container", account_name="my-account")

    assert result == BucketStatus(exists=True, access="anonymous")
