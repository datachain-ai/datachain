from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.exceptions import ReadTimeoutError
from s3fs import S3FileSystem

from datachain.client.s3 import ClientS3, _head_bucket


def test_normalize_s3_kwargs_secret_and_token():
    result = ClientS3._normalize_s3_kwargs({"aws_secret": "s", "aws_token": "t"})
    assert result == {"secret": "s", "token": "t"}


@patch("datachain.client.s3.S3FileSystem")
@patch(
    "datachain.client.s3._head_bucket", side_effect=[(403, "eu-west-3"), (200, None)]
)
def test_s3_bucket_status_keeps_caller_config(_head, fs_cls):
    config = {"cache_regions": True, "config_kwargs": {"max_pool_connections": 64}}

    ClientS3.bucket_status("b", **config)
    ClientS3.create_fs(anon=True, **config)

    assert config == {
        "cache_regions": True,
        "config_kwargs": {"max_pool_connections": 64},
    }
    assert all(call.kwargs["cache_regions"] is False for call in fs_cls.call_args_list)
    auth_kwargs = fs_cls.call_args_list[1].kwargs
    assert auth_kwargs["config_kwargs"]["region_name"] == "eu-west-3"


@pytest.mark.parametrize("client_kwargs", [None, {}])
def test_s3_client_takes_client_kwargs_none(client_kwargs):
    client = ClientS3.from_name(
        "b", MagicMock(), {"anon": True, "client_kwargs": client_kwargs}
    )
    assert client.fs.anon


def test_head_bucket_retries_transient_errors(mocker):
    mocker.patch("asyncio.sleep")
    fs = S3FileSystem(anon=True, skip_instance_cache=True)
    ok = {"ResponseMetadata": {"HTTPStatusCode": 200, "HTTPHeaders": {}}}
    head = AsyncMock(side_effect=[ReadTimeoutError(endpoint_url="e"), ok])
    head.__name__ = "head_bucket"
    fs.get_s3 = AsyncMock(return_value=MagicMock(head_bucket=head))

    assert _head_bucket(fs, "b") == (200, None)
    assert head.call_count == 2


@patch("datachain.client.s3._head_bucket", side_effect=[(403, None), (200, None)])
@patch("datachain.client.s3.S3FileSystem")
@pytest.mark.parametrize("config_region", [None, "eu-west-1"])
def test_s3_bucket_status_takes_region_name(fs_cls, head, config_region):
    config = {} if config_region is None else {"region_name": config_region}
    ClientS3.bucket_status("b", region_name="us-west-2", config_kwargs=config)
    for call in fs_cls.call_args_list:
        assert "region_name" not in call.kwargs
        assert call.kwargs["config_kwargs"]["region_name"] == (
            config_region or "us-west-2"
        )
    assert config == ({} if config_region is None else {"region_name": config_region})
