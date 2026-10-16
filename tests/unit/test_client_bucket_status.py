from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
)
from botocore.exceptions import ClientError, NoCredentialsError
from fsspec.asyn import get_loop, sync
from gcsfs.retry import HttpError
from google.api_core import exceptions as google_exceptions
from s3fs import S3FileSystem

from datachain.cli import handle_bucket_command, handle_command
from datachain.client import bucket_status, s3
from datachain.client.azure import AzureClient
from datachain.client.fsspec import BucketStatus
from datachain.client.gcs import GCSClient
from datachain.client.s3 import ClientS3, _head_bucket

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
        ((200, "r"), None, BucketStatus(True, "anonymous")),
        ((404, None), None, BucketStatus(False, "denied", "S3 bucket 'b' not found")),
        ((403, "r"), (200, "r"), BucketStatus(True, "authenticated")),
        ((403, "r"), NoCredentialsError(), BucketStatus(True, "denied", S3_DENIED)),
        ((403, "r"), (403, "r"), BucketStatus(True, "denied", S3_DENIED)),
        (
            (403, None),
            (404, None),
            BucketStatus(False, "denied", "S3 bucket 'b' not found"),
        ),
        ((301, None), None, BucketStatus(False, "denied", "S3 bucket 'b' not found")),
        ((301, "r"), None, BucketStatus(False, "denied", "S3 bucket 'b' not found")),
        (
            (403, None),
            (301, None),
            BucketStatus(False, "denied", "S3 bucket 'b' not found"),
        ),
    ],
    ids=[
        "anon-ok",
        "anon-404",
        "auth-ok",
        "auth-no-creds",
        "auth-perm",
        "auth-404",
        "anon-301",
        "anon-301-region",
        "auth-301",
    ],
)
@patch("datachain.client.s3.S3FileSystem")
@patch("datachain.client.s3._head_bucket")
def test_s3_bucket_status(head, _fs_cls, anon_effect, auth_effect, expected):
    head.side_effect = [anon_effect, auth_effect]
    assert ClientS3.bucket_status("b") == expected


@pytest.fixture(autouse=True)
def regions(monkeypatch):
    monkeypatch.setattr(s3, "_BUCKET_REGIONS", {})
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_PROFILE", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    return s3._BUCKET_REGIONS


@patch("datachain.client.s3.S3FileSystem")
@patch("datachain.client.s3._head_bucket", return_value=(200, "eu-west-3"))
def test_s3_bucket_region_reaches_the_listing_client(_head, _fs_cls, regions):
    ClientS3.bucket_status("regional")

    client = ClientS3.from_name("regional", MagicMock(), {})
    assert client.fs_kwargs["region_name"] == "eu-west-3"
    assert "region_name" not in ClientS3.from_name("other", MagicMock(), {}).fs_kwargs


@patch("datachain.client.s3.S3FileSystem")
@patch("datachain.client.s3._head_bucket", return_value=(200, "eu-west-3"))
def test_s3_bucket_region_is_kept_per_default_profile(
    _head, _fs_cls, regions, monkeypatch
):
    monkeypatch.setenv("AWS_DEFAULT_PROFILE", "one")
    ClientS3.bucket_status("regional")
    monkeypatch.setenv("AWS_DEFAULT_PROFILE", "two")

    assert (
        "region_name" not in ClientS3.from_name("regional", MagicMock(), {}).fs_kwargs
    )


@pytest.mark.parametrize(
    "config",
    [{"endpoint_url": "http://e"}, {"client_kwargs": {"endpoint_url": "http://e"}}],
)
@patch("datachain.client.s3.S3FileSystem")
@patch("datachain.client.s3._head_bucket", return_value=(200, "eu-west-3"))
def test_s3_bucket_region_is_not_kept_for_other_endpoints(
    _head, _fs_cls, regions, config
):
    ClientS3.bucket_status("regional")
    ClientS3.bucket_status("regional", **config)

    assert len(regions) == 1
    client = ClientS3.from_name("regional", MagicMock(), config)
    assert "region_name" not in client.fs_kwargs


@patch("datachain.client.s3.S3FileSystem")
@patch(
    "datachain.client.s3._head_bucket", side_effect=[(403, "eu-west-3"), (200, None)]
)
def test_s3_bucket_status_keeps_caller_config(_head, fs_cls, regions):
    config = {"config_kwargs": {"max_pool_connections": 64}}

    ClientS3.bucket_status("b", **config)
    ClientS3.create_fs(anon=True, **config)

    assert config == {"config_kwargs": {"max_pool_connections": 64}}
    auth_kwargs = fs_cls.call_args_list[1].kwargs
    assert auth_kwargs["config_kwargs"]["region_name"] == "eu-west-3"


@pytest.mark.parametrize("client_kwargs", [None, {}])
def test_s3_client_takes_client_kwargs_none(client_kwargs):
    client = ClientS3.from_name(
        "b", MagicMock(), {"anon": True, "client_kwargs": client_kwargs}
    )
    assert client.fs.anon


def _client_error(status):
    return ClientError(
        {
            "Error": {"Code": str(status)},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "HeadBucket",
    )


def test_head_bucket_retries_slow_down():
    fs = S3FileSystem(anon=True, skip_instance_cache=True)
    slow_down = ClientError(
        {"Error": {"Code": "SlowDown"}, "ResponseMetadata": {"HTTPStatusCode": 503}},
        "HeadBucket",
    )
    ok = {"ResponseMetadata": {"HTTPStatusCode": 200, "HTTPHeaders": {}}}
    head = AsyncMock(side_effect=[slow_down, ok])
    head.__name__ = "head_bucket"
    fs.get_s3 = AsyncMock(return_value=MagicMock(head_bucket=head))

    with patch("asyncio.sleep", AsyncMock()):
        assert sync(get_loop(), _head_bucket, fs, "b") == (200, None)
    assert head.call_count == 2


@pytest.mark.parametrize("status", [301, 403, 404, 503])
def test_head_bucket(status):
    fs = S3FileSystem(anon=True, s3_additional_kwargs={"ExpectedBucketOwner": "1"})
    head = AsyncMock(side_effect=_client_error(status))
    head.__name__ = "head_bucket"
    fs.get_s3 = AsyncMock(return_value=MagicMock(head_bucket=head))

    if status == 503:
        with pytest.raises(OSError):  # as s3fs raises it
            sync(get_loop(), _head_bucket, fs, "b")
    else:
        assert sync(get_loop(), _head_bucket, fs, "b") == (status, None)
    assert head.call_args.kwargs == {"Bucket": "b", "ExpectedBucketOwner": "1"}


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


@patch("datachain.client.s3._head_bucket", return_value=(200, "eu-west-3"))
def test_s3_bucket_region_is_kept_per_profile(_head, regions):
    ClientS3.bucket_status("b", profile="one")

    client = ClientS3.from_name("b", MagicMock(), {"profile": "one"})
    assert client.fs_kwargs["region_name"] == "eu-west-3"
    assert "region_name" not in ClientS3.from_name("b", MagicMock(), {}).fs_kwargs


def test_s3_bucket_region_reaches_the_listing_filesystem(mocker, regions):
    head = AsyncMock(
        return_value={
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "HTTPHeaders": {"x-amz-bucket-region": "eu-west-3"},
            }
        }
    )
    head.__name__ = "head_bucket"
    mocker.patch.object(
        S3FileSystem, "get_s3", AsyncMock(return_value=MagicMock(head_bucket=head))
    )

    assert ClientS3.bucket_status("b").access == "anonymous"
    fs = ClientS3.from_name("b", MagicMock(), {"anon": True}).fs

    assert head.call_count == 1
    assert fs.config_kwargs["region_name"] == "eu-west-3"
    assert not fs.cache_regions


@patch("datachain.client.s3._head_bucket", return_value=(200, "cn-north-1"))
def test_s3_bucket_region_is_kept_per_configured_region(_head, regions):
    ClientS3.bucket_status("shared", client_kwargs={"region_name": "cn-north-1"})

    assert "region_name" not in ClientS3.from_name("shared", MagicMock(), {}).fs_kwargs


def test_head_bucket_retries_transient_errors(mocker):
    from botocore.exceptions import ReadTimeoutError

    mocker.patch("asyncio.sleep")
    fs = S3FileSystem(anon=True, skip_instance_cache=True)
    ok = {"ResponseMetadata": {"HTTPStatusCode": 200, "HTTPHeaders": {}}}
    head = AsyncMock(side_effect=[ReadTimeoutError(endpoint_url="e"), ok])
    head.__name__ = "head_bucket"
    fs.get_s3 = AsyncMock(return_value=MagicMock(head_bucket=head))

    assert sync(get_loop(), _head_bucket, fs, "b") == (200, None)
    assert head.call_count == 2
