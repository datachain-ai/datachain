import io
import tarfile
from unittest.mock import Mock

import pytest
from fsspec.caching import BlockCache, ReadAheadCache, caches
from gcsfs.core import GCSFile
from huggingface_hub.hf_file_system import (
    HfFileSystemFile,
    HfFileSystemResolvedPath,
)
from s3fs.core import S3File

from datachain.cache import Cache
from datachain.client.gcs import GCSClient
from datachain.client.hf import HfClient
from datachain.client.s3 import ClientS3
from datachain.lib.file import File
from datachain.lib.tar import process_tar


@pytest.mark.parametrize("cache_type", ["readahead", "bytes"])
@pytest.mark.parametrize("block_size", [None, 0, 256 * 1024])
def test_open_object_resolved_read_ahead(tmp_path, monkeypatch, cache_type, block_size):
    data = b"01234567" * (256 * 1024)
    default = 512 * 1024
    ceiling = 100 * 1024
    client = ClientS3(
        "bucket",
        {"anon": True, "default_block_size": default, "default_cache_type": cache_type},
        Cache(str(tmp_path / "cache"), str(tmp_path / "tmp")),
    )
    fs = client.fs
    monkeypatch.setattr(fs, "info", lambda *a, **kw: {"size": len(data)})
    calls = []

    def fetch_range(file, start, end):
        calls.append((start, min(end, len(data))))
        return data[start:end]

    monkeypatch.setattr(S3File, "_fetch_range", fetch_range)
    file = File(source=client.uri, path="archive.tar")
    with client.open_object(
        file, block_size=block_size, max_block_size=ceiling
    ) as stream:
        opened = stream.fileobj
        assert isinstance(opened, S3File)
        expected = ceiling if block_size is None else block_size
        assert opened.blocksize == expected
        assert opened.cache.blocksize == expected
        stream.seek(512)
        assert stream.read(ceiling) == data[512 : 512 + ceiling]
        stream.seek(len(data) - 17)
        assert stream.read() == data[-17:]
        assert stream.read() == b""
    assert opened.closed
    assert calls[0][1] - calls[0][0] <= ceiling + expected


def test_open_object_preserves_smaller_backend_default(tmp_path, monkeypatch):
    client = ClientS3(
        "bucket",
        {"anon": True, "default_block_size": 4096},
        Cache(str(tmp_path / "cache"), str(tmp_path / "tmp")),
    )
    monkeypatch.setattr(client.fs, "info", lambda *a, **kw: {"size": 1024 * 1024})
    monkeypatch.setattr(
        S3File, "_fetch_range", lambda file, start, end: bytes(end - start)
    )
    file = File(source=client.uri, path="archive.tar")
    with client.open_object(file, max_block_size=100 * 1024) as stream:
        assert stream.fileobj.blocksize == 4096
        assert stream.fileobj.cache.blocksize == 4096
        assert stream.read(1) == b"\0"


@pytest.mark.parametrize("cache_type", ["blockcache", "custom-readahead"])
def test_open_object_preserves_other_cache_policies(tmp_path, monkeypatch, cache_type):
    class CustomReadAheadCache(ReadAheadCache):
        pass

    monkeypatch.setitem(caches, "custom-readahead", CustomReadAheadCache)
    data = b"abcdefgh" * (128 * 1024)
    default = 256 * 1024
    client = ClientS3(
        "bucket",
        {"anon": True, "default_block_size": default, "default_cache_type": cache_type},
        Cache(str(tmp_path / "cache"), str(tmp_path / "tmp")),
    )
    monkeypatch.setattr(client.fs, "info", lambda *a, **kw: {"size": len(data)})
    monkeypatch.setattr(
        S3File, "_fetch_range", lambda file, start, end: data[start:end]
    )
    file = File(source=client.uri, path="archive.tar")
    with client.open_object(file, max_block_size=4096) as stream:
        cache = stream.fileobj.cache
        assert type(cache) is (
            BlockCache if cache_type == "blockcache" else CustomReadAheadCache
        )
        assert stream.fileobj.blocksize == default
        assert cache.blocksize == default
        stream.seek(len(data) - 17)
        assert stream.read() == data[-17:]
        stream.seek(3)
        assert stream.read(9) == data[3:12]


def test_tar_member_resolves_huggingface_file_default(tmp_path, monkeypatch):
    data = b"m" * (100 * 1024)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        info = tarfile.TarInfo("member.bin")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        padding = tarfile.TarInfo("padding.bin")
        padding.size = 2 * 1024 * 1024
        tar.addfile(padding, io.BytesIO(bytes(padding.size)))
    raw = archive.getvalue()
    client = HfClient(
        "datasets/fictional/fixture",
        {"token": False},
        Cache(str(tmp_path / "cache"), str(tmp_path / "tmp")),
    )
    fs = client.fs.sync_fs
    assert fs.block_size is None
    monkeypatch.setattr(
        fs,
        "resolve_path",
        lambda *a, **kw: HfFileSystemResolvedPath(
            "datasets/fictional/fixture", "archive.tar"
        ),
    )
    monkeypatch.setattr(fs, "info", lambda *a, **kw: {"size": len(raw), "type": "file"})
    calls = []

    def fetch_range(file, start, end):
        assert isinstance(file, HfFileSystemFile)
        calls.append((start, min(end, len(raw))))
        return raw[start:end]

    monkeypatch.setattr(HfFileSystemFile, "_fetch_range", fetch_range)
    catalog = Mock(client_config={})
    catalog.get_client.return_value = client
    parent = File(source=client.uri, path="archive.tar", size=len(raw))
    parent._set_stream(catalog, caching_enabled=False)
    member = next(process_tar(parent))
    member._set_stream(catalog, caching_enabled=False)
    fetched = []
    for block_size in (None, len(data)):
        calls.clear()
        with member.open(block_size=block_size) as stream:
            assert stream.read() == data
        fetched.append(sum(end - start for start, end in calls))
    assert 0 < fetched[0] <= fetched[1] < len(raw) // 2


def test_versioned_gcs_resolved_read_ahead(tmp_path, monkeypatch):
    data = b"01234567" * (128 * 1024)
    client = GCSClient(
        "bucket",
        {"token": "anon", "default_block_size": 512 * 1024, "version_aware": True},
        Cache(str(tmp_path / "cache"), str(tmp_path / "tmp")),
    )
    fs = client.fs
    monkeypatch.setattr(fs, "info", lambda *a, **kw: {"size": len(data)})
    open_file = fs.open

    def open_readahead(path, **kwargs):
        return open_file(path, cache_type="readahead", **kwargs)

    monkeypatch.setattr(fs, "open", open_readahead)
    monkeypatch.setattr(
        GCSFile, "_fetch_range", lambda file, start, end: data[start:end]
    )
    file = File(source=client.uri, path="archive.tar", version="123")
    with client.open_object(file, max_block_size=4096) as stream:
        opened = stream.fileobj
        assert isinstance(opened, GCSFile)
        assert opened.path == "bucket/archive.tar#123"
        assert opened.generation == "123"
        assert opened.blocksize == opened.cache.blocksize == 4096
        stream.seek(len(data) - 17)
        assert stream.read() == data[-17:]
    assert opened.closed
