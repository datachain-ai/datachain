from unittest.mock import MagicMock, patch

from datachain.client.azure import AzureClient
from datachain.client.fsspec import BucketStatus
from datachain.client.gcs import GCSClient
from datachain.client.s3 import ClientS3

# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------


@patch("datachain.client.s3.S3FileSystem")
@patch("datachain.client.s3.sync")
def test_s3_anonymous(mock_sync, mock_s3fs_cls):
    anon_fs = MagicMock()
    anon_fs._info.return_value = {"name": "my-bucket", "type": "directory"}
    mock_s3fs_cls.return_value = anon_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = ClientS3.bucket_status("my-bucket")

    assert result == BucketStatus(exists=True, access="anonymous")


@patch("datachain.client.s3.S3FileSystem")
@patch("datachain.client.s3.sync")
def test_s3_authenticated(mock_sync, mock_s3fs_cls):
    anon_fs = MagicMock()
    anon_fs._info.side_effect = PermissionError("AccessDenied")
    auth_fs = MagicMock()
    auth_fs._info.return_value = {"name": "my-bucket", "type": "directory"}

    def make_fs(**kwargs):
        return anon_fs if kwargs.get("anon") else auth_fs

    mock_s3fs_cls.side_effect = make_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = ClientS3.bucket_status("my-bucket")

    assert result == BucketStatus(exists=True, access="authenticated")


@patch("datachain.client.s3.S3FileSystem")
@patch("datachain.client.s3.sync")
def test_s3_denied_no_credentials(mock_sync, mock_s3fs_cls):
    from botocore.exceptions import NoCredentialsError

    anon_fs = MagicMock()
    anon_fs._info.side_effect = PermissionError("AccessDenied")
    auth_fs = MagicMock()
    auth_fs._info.side_effect = NoCredentialsError()

    def make_fs(**kwargs):
        return anon_fs if kwargs.get("anon") else auth_fs

    mock_s3fs_cls.side_effect = make_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = ClientS3.bucket_status("my-bucket")

    assert result.exists is True
    assert result.access == "denied"
    assert result.error is not None


@patch("datachain.client.s3.S3FileSystem")
@patch("datachain.client.s3.sync")
def test_s3_denied_permission_error(mock_sync, mock_s3fs_cls):
    anon_fs = MagicMock()
    anon_fs._info.side_effect = PermissionError("AccessDenied")
    auth_fs = MagicMock()
    auth_fs._info.side_effect = PermissionError("AccessDenied")

    def make_fs(**kwargs):
        return anon_fs if kwargs.get("anon") else auth_fs

    mock_s3fs_cls.side_effect = make_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = ClientS3.bucket_status("my-bucket")

    assert result.exists is True
    assert result.access == "denied"
    assert result.error is not None


@patch("datachain.client.s3.S3FileSystem")
@patch("datachain.client.s3.sync")
def test_s3_not_found(mock_sync, mock_s3fs_cls):
    anon_fs = MagicMock()
    anon_fs._info.side_effect = FileNotFoundError("NoSuchBucket")
    mock_s3fs_cls.return_value = anon_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = ClientS3.bucket_status("my-bucket")

    assert result.exists is False
    assert result.access == "denied"
    assert result.error is not None


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


# ---------------------------------------------------------------------------
# GCS
# ---------------------------------------------------------------------------


@patch("datachain.client.gcs.GCSFileSystem")
@patch("datachain.client.gcs.sync")
def test_gcs_anonymous(mock_sync, mock_gcsfs_cls):
    anon_fs = MagicMock()
    anon_fs._ls.return_value = []
    mock_gcsfs_cls.return_value = anon_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = GCSClient.bucket_status("my-bucket")

    assert result == BucketStatus(exists=True, access="anonymous")


@patch.object(GCSClient, "create_fs")
@patch("datachain.client.gcs.GCSFileSystem")
@patch("datachain.client.gcs.sync")
def test_gcs_authenticated_via_http_error(mock_sync, mock_gcsfs_cls, mock_create_fs):
    from gcsfs.retry import HttpError

    anon_fs = MagicMock()
    anon_fs._ls.side_effect = HttpError({"code": 401, "message": "Permission denied"})
    mock_gcsfs_cls.return_value = anon_fs

    auth_fs = MagicMock()
    auth_fs._info.return_value = {"name": "my-bucket", "type": "directory"}
    mock_create_fs.return_value = auth_fs

    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = GCSClient.bucket_status("my-bucket")

    assert result == BucketStatus(exists=True, access="authenticated")


@patch.object(GCSClient, "create_fs")
@patch("datachain.client.gcs.GCSFileSystem")
@patch("datachain.client.gcs.sync")
def test_gcs_authenticated(mock_sync, mock_gcsfs_cls, mock_create_fs):
    anon_fs = MagicMock()
    anon_fs._ls.side_effect = PermissionError("403")
    mock_gcsfs_cls.return_value = anon_fs

    auth_fs = MagicMock()
    auth_fs._info.return_value = {"name": "my-bucket", "type": "directory"}
    mock_create_fs.return_value = auth_fs

    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = GCSClient.bucket_status("my-bucket")

    assert result == BucketStatus(exists=True, access="authenticated")


@patch.object(GCSClient, "create_fs")
@patch("datachain.client.gcs.GCSFileSystem")
@patch("datachain.client.gcs.sync")
def test_gcs_denied(mock_sync, mock_gcsfs_cls, mock_create_fs):
    anon_fs = MagicMock()
    anon_fs._ls.side_effect = PermissionError("403")
    mock_gcsfs_cls.return_value = anon_fs

    auth_fs = MagicMock()
    auth_fs._info.side_effect = PermissionError("403")
    mock_create_fs.return_value = auth_fs

    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = GCSClient.bucket_status("my-bucket")

    assert result.exists is True
    assert result.access == "denied"
    assert result.error is not None


@patch("datachain.client.gcs.GCSFileSystem")
@patch("datachain.client.gcs.sync")
def test_gcs_not_found(mock_sync, mock_gcsfs_cls):
    anon_fs = MagicMock()
    anon_fs._ls.side_effect = FileNotFoundError("bucket not found")
    mock_gcsfs_cls.return_value = anon_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = GCSClient.bucket_status("my-bucket")

    assert result.exists is False
    assert result.access == "denied"
    assert result.error is not None


# ---------------------------------------------------------------------------
# Azure
# ---------------------------------------------------------------------------


@patch.object(AzureClient, "create_fs")
@patch("datachain.client.azure.AzureBlobFileSystem")
@patch("datachain.client.azure.sync")
def test_azure_anonymous(mock_sync, mock_adlfs_cls, mock_create_fs):
    # Auth probe fails (no account_name), falls through to anonymous probe.
    mock_create_fs.side_effect = ValueError("Must provide account_name")

    anon_fs = MagicMock()
    anon_fs._info.return_value = {"name": "my-container", "type": "directory"}
    mock_adlfs_cls.return_value = anon_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = AzureClient.bucket_status("my-container")

    assert result == BucketStatus(exists=True, access="anonymous")


@patch.object(AzureClient, "create_fs")
@patch("datachain.client.azure.AzureBlobFileSystem")
@patch("datachain.client.azure.sync")
def test_azure_authenticated(mock_sync, mock_adlfs_cls, mock_create_fs):
    auth_fs = MagicMock()
    auth_fs._info.return_value = {"name": "my-container", "type": "directory"}
    mock_create_fs.return_value = auth_fs

    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = AzureClient.bucket_status("my-container")

    assert result == BucketStatus(exists=True, access="authenticated")


@patch.object(AzureClient, "create_fs")
@patch("datachain.client.azure.AzureBlobFileSystem")
@patch("datachain.client.azure.sync")
def test_azure_denied(mock_sync, mock_adlfs_cls, mock_create_fs):
    auth_fs = MagicMock()
    auth_fs._info.side_effect = PermissionError("403")
    mock_create_fs.return_value = auth_fs

    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = AzureClient.bucket_status("my-container")

    assert result.exists is True
    assert result.access == "denied"
    assert result.error is not None


@patch.object(AzureClient, "create_fs")
@patch("datachain.client.azure.AzureBlobFileSystem")
@patch("datachain.client.azure.sync")
def test_azure_not_found(mock_sync, mock_adlfs_cls, mock_create_fs):
    auth_fs = MagicMock()
    auth_fs._info.side_effect = FileNotFoundError("container not found")
    mock_create_fs.return_value = auth_fs

    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = AzureClient.bucket_status("my-container")

    assert result.exists is False
    assert result.access == "denied"
    assert result.error is not None


@patch.object(AzureClient, "create_fs")
@patch("datachain.client.azure.AzureBlobFileSystem")
@patch("datachain.client.azure.sync")
def test_azure_no_account_name_no_public_access(
    mock_sync, mock_adlfs_cls, mock_create_fs
):
    """create_fs() raises ValueError (no credentials); anon probe also denied."""
    mock_create_fs.side_effect = ValueError(
        "Must provide either a connection_string or account_name with credentials!!"
    )

    anon_fs = MagicMock()
    anon_fs._info.side_effect = PermissionError("anonymous access denied")
    mock_adlfs_cls.return_value = anon_fs
    mock_sync.side_effect = lambda _loop, fn, *args, **kwargs: fn(*args, **kwargs)

    result = AzureClient.bucket_status("my-container")

    assert result.exists is False
    assert result.access == "denied"
    assert result.error is not None
