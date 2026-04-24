from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
)
from botocore.exceptions import NoCredentialsError
from gcsfs.retry import HttpError
from google.api_core import exceptions as google_exceptions

from datachain.cli import handle_bucket_command, handle_command
from datachain.client import bucket_status
from datachain.client.azure import AzureClient
from datachain.client.fsspec import BucketStatus
from datachain.client.gcs import GCSClient
from datachain.client.s3 import ClientS3

S3_DENIED = "Access denied to S3 bucket 'b' — check AWS credentials/permissions"
AZ_DENIED = "Access denied to Azure container 'c' — check credentials/configuration"
GCS_DENIED = "Access denied to GCS bucket 'b' — check credentials/permissions"


def _azure_http_error(status_code):
    resp = MagicMock(status_code=status_code)
    return HttpResponseError(message="test", response=resp)


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


def test_normalize_s3_kwargs_secret_and_token():
    result = ClientS3._normalize_s3_kwargs({"aws_secret": "s", "aws_token": "t"})
    assert result == {"secret": "s", "token": "t"}


@pytest.mark.parametrize(
    "anon_effect, auth_effect, expected",
    [
        (None, None, BucketStatus(True, "anonymous")),
        (
            FileNotFoundError,
            None,
            BucketStatus(False, "denied", "S3 bucket 'b' not found"),
        ),
        (PermissionError, None, BucketStatus(True, "authenticated")),
        (PermissionError, NoCredentialsError, BucketStatus(True, "denied", S3_DENIED)),
        (PermissionError, PermissionError, BucketStatus(True, "denied", S3_DENIED)),
        (
            PermissionError,
            FileNotFoundError,
            BucketStatus(False, "denied", "S3 bucket 'b' not found"),
        ),
    ],
    ids=["anon-ok", "anon-404", "auth-ok", "auth-no-creds", "auth-perm", "auth-404"],
)
@patch("datachain.client.s3.S3FileSystem")
def test_s3_bucket_status(mock_fs_cls, anon_effect, auth_effect, expected):
    anon_fs, auth_fs = MagicMock(), MagicMock()
    mock_fs_cls.side_effect = [anon_fs, auth_fs]
    anon_fs._info = AsyncMock(side_effect=anon_effect)
    auth_fs._info = AsyncMock(side_effect=auth_effect)
    assert ClientS3.bucket_status("b") == expected


@pytest.mark.parametrize(
    "anon_effect, auth_effect, expected",
    [
        (None, None, BucketStatus(True, "anonymous")),
        (
            FileNotFoundError,
            None,
            BucketStatus(False, "denied", "GCS bucket 'b' not found"),
        ),
        (
            HttpError({"code": 404, "message": ""}),
            None,
            BucketStatus(False, "denied", "GCS bucket 'b' not found"),
        ),
        (PermissionError, None, BucketStatus(True, "authenticated")),
        (
            PermissionError,
            FileNotFoundError,
            BucketStatus(False, "denied", "GCS bucket 'b' not found"),
        ),
        (
            PermissionError,
            google_exceptions.Forbidden("no access"),
            BucketStatus(True, "denied", "403 no access"),
        ),
        (
            PermissionError,
            google_exceptions.PermissionDenied("denied"),
            BucketStatus(True, "denied", "403 denied"),
        ),
        (
            PermissionError,
            HttpError({"code": 404, "message": ""}),
            BucketStatus(False, "denied", "GCS bucket 'b' not found"),
        ),
        (PermissionError, OSError("boom"), BucketStatus(True, "denied", GCS_DENIED)),
    ],
    ids=[
        "anon-ok",
        "anon-404",
        "anon-http-404",
        "auth-ok",
        "auth-404",
        "auth-forbidden",
        "auth-perm-denied",
        "auth-http-404",
        "auth-oserror",
    ],
)
@patch.object(GCSClient, "create_fs")
def test_gcs_bucket_status(mock_create_fs, anon_effect, auth_effect, expected):
    anon_fs, auth_fs = MagicMock(), MagicMock()
    mock_create_fs.side_effect = [anon_fs, auth_fs]
    anon_fs._ls = AsyncMock(side_effect=anon_effect)
    auth_fs._info = AsyncMock(side_effect=auth_effect)
    assert GCSClient.bucket_status("b") == expected


@pytest.mark.parametrize(
    "anon_effect, auth_effect, kwargs, expected",
    [
        (None, None, {"account_name": "a"}, BucketStatus(True, "anonymous")),
        (
            ClientAuthenticationError,
            None,
            {"account_name": "a"},
            BucketStatus(True, "authenticated"),
        ),
        (
            ResourceNotFoundError,
            None,
            {"account_name": "a"},
            BucketStatus(False, "denied", "Azure container 'c' not found"),
        ),
        (
            _azure_http_error(401),
            None,
            {"account_name": "a"},
            BucketStatus(True, "authenticated"),
        ),
        (None, None, {}, BucketStatus(True, "authenticated")),
        (
            ClientAuthenticationError,
            PermissionError,
            {"account_name": "a"},
            BucketStatus(True, "denied", AZ_DENIED),
        ),
        (
            ClientAuthenticationError,
            ClientAuthenticationError,
            {"account_name": "a"},
            BucketStatus(True, "denied", AZ_DENIED),
        ),
        (
            ClientAuthenticationError,
            FileNotFoundError,
            {"account_name": "a"},
            BucketStatus(False, "denied", "Azure container 'c' not found"),
        ),
        (
            ClientAuthenticationError,
            ValueError("bad"),
            {"account_name": "a"},
            BucketStatus(False, "denied", "bad"),
        ),
    ],
    ids=[
        "anon-ok",
        "anon-auth-err",
        "anon-404",
        "anon-http-401",
        "no-account",
        "auth-perm",
        "auth-client-auth",
        "auth-404",
        "auth-value-err",
    ],
)
@patch("datachain.client.azure.BlobServiceClient")
@patch.object(AzureClient, "create_fs")
def test_azure_bucket_status(
    mock_create_fs, mock_blob_svc, anon_effect, auth_effect, kwargs, expected
):
    container = MagicMock()
    mock_blob_svc.return_value.get_container_client.return_value = container
    if anon_effect is not None:
        exc = anon_effect if isinstance(anon_effect, BaseException) else anon_effect()
        container.get_container_properties.side_effect = exc

    auth_fs = MagicMock()
    mock_create_fs.return_value = auth_fs
    auth_fs._info = AsyncMock(side_effect=auth_effect)

    assert AzureClient.bucket_status("c", **kwargs) == expected


@patch("datachain.client.azure.BlobServiceClient")
@patch.object(AzureClient, "create_fs")
def test_azure_bucket_status_http_500_reraises(_mock_create_fs, mock_blob_svc):
    container = MagicMock()
    mock_blob_svc.return_value.get_container_client.return_value = container
    container.get_container_properties.side_effect = _azure_http_error(500)
    with pytest.raises(HttpResponseError):
        AzureClient.bucket_status("c", account_name="a")


def test_handle_command_invalid():
    args = MagicMock(command="nonexistent-cmd")
    assert handle_command(args, MagicMock(), {}) == 1


def test_handle_bucket_command_no_subcommand(capsys):
    args = MagicMock(command="bucket", bucket_cmd=None)
    assert handle_bucket_command(args, {}) == 1
    assert "--help" in capsys.readouterr().err


@patch("datachain.cli.bucket_status_cmd", return_value=0)
def test_handle_bucket_command_status(mock_cmd):
    args = MagicMock(bucket_cmd="status", uri="s3://b", account_name=None)
    assert handle_bucket_command(args, {"anon": True}) == 0
    mock_cmd.assert_called_once_with("s3://b", client_config={"anon": True})


@patch("datachain.cli.bucket_status_cmd", return_value=0)
def test_handle_bucket_command_status_with_account_name(mock_cmd):
    args = MagicMock(bucket_cmd="status", uri="az://c", account_name="acct")
    assert handle_bucket_command(args, {}) == 0
    mock_cmd.assert_called_once_with("az://c", client_config={"account_name": "acct"})


def test_handle_bucket_command_unexpected_subcommand():
    args = MagicMock(bucket_cmd="foobar")
    with pytest.raises(NotImplementedError, match="Unexpected bucket subcommand"):
        handle_bucket_command(args, {})
