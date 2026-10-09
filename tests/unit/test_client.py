import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from datachain.client import Client
from datachain.client.gcs import GCSClient
from datachain.client.http import HTTPSClient
from datachain.client.local import FileClient
from datachain.client.s3 import ClientS3
from datachain.client.writeconfig import WriteConfig
from datachain.lib.file import File


def test_bad_protocol():
    with pytest.raises(NotImplementedError):
        Client.get_implementation("bogus://bucket")


def test_write_kwargs_base_default_ignores_content_settings_and_metadata():
    # Backends that don't override _write_kwargs (e.g. HF) fall back to the base,
    # which has no native mapping for content settings or metadata and drops them.
    cfg = WriteConfig(
        content_type="application/pdf",
        content_disposition="attachment",
        metadata={"a": "b"},
    )
    assert Client._write_kwargs(cfg, streaming=True) == {}


def test_write_kwargs_base_default_rejects_write_options():
    # The base rejects the raw escape hatch rather than crash or silently drop it.
    cfg = WriteConfig(write_options={"foo": "bar"})
    with pytest.raises(NotImplementedError, match="write_options"):
        Client._write_kwargs(cfg, streaming=True)


def test_win_paths_are_recognized():
    if sys.platform != "win32":
        pytest.skip()

    assert Client.get_implementation("file://C:/bucket") == FileClient
    assert Client.get_implementation("file://C:\\bucket") == FileClient
    assert Client.get_implementation("file://\\bucket") == FileClient
    assert Client.get_implementation("file:///bucket") == FileClient
    assert Client.get_implementation("C://bucket") == FileClient
    assert Client.get_implementation("C:\\bucket") == FileClient
    assert Client.get_implementation("\bucket") == FileClient


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
def test_parse_file_path_ends_with_slash(cloud_type):
    uri, rel_part = Client.parse_url("./animals/".replace("/", os.sep))
    assert uri == (Path().absolute() / Path("animals")).as_uri()
    assert rel_part == ""


@pytest.mark.parametrize(
    "client_cls,source",
    [
        (ClientS3, "s3://bucket"),
        (GCSClient, "gs://bucket"),
        (HTTPSClient, "https://example.com"),
    ],
)
@pytest.mark.parametrize("block_size", [None, 1024 * 1024])
def test_open_object_passes_block_size(client_cls, source, block_size):
    cache = MagicMock()
    cache.get_path.return_value = None
    client = client_cls("bucket", {}, cache)
    client._fs = MagicMock()

    client.open_object(File(source=source, path="a.mp4"), block_size=block_size)

    assert client._fs.open.call_args.kwargs["block_size"] == block_size
