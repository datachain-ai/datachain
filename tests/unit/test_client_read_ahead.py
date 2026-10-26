import io
import tarfile
from unittest.mock import Mock

import pytest
from gcsfs.core import GCSFile
from huggingface_hub import hf_file_system
from huggingface_hub.hf_file_system import HfFileSystemFile

from datachain.cache import Cache
from datachain.client.gcs import GCSClient
from datachain.client.hf import HfClient
from datachain.lib.file import File
from datachain.lib.tar import build_tar_member, process_tar


def _tar_bytes(data):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, payload in (
            ("prefix.bin", b"p" * 1024),
            ("member.bin", data),
            ("padding.bin", bytes(4 * 1024 * 1024)),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _tar_member(client, raw, version=""):
    catalog = Mock(client_config={})
    catalog.get_client.return_value = client
    parent = File(source=client.uri, path="archive.tar", size=len(raw), version=version)
    parent._set_stream(catalog, caching_enabled=False)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        member = build_tar_member(parent, archive.getmember("member.bin"))
    member._set_stream(catalog, caching_enabled=False)
    return parent, member


def test_tar_member_huggingface_read_ahead(tmp_path, monkeypatch):
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
    resolved_path_type = getattr(
        hf_file_system,
        "HfFileSystemResolvedRepositoryPath",
        hf_file_system.HfFileSystemResolvedPath,
    )
    monkeypatch.setattr(
        fs,
        "resolve_path",
        lambda *a, **kw: resolved_path_type(
            repo_type="dataset",
            repo_id="fictional/fixture",
            revision="main",
            path_in_repo="archive.tar",
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
            assert stream.read() == b""
        fetched.append(sum(end - start for start, end in calls))
    assert 0 < fetched[0] <= fetched[1] < len(raw) // 2


@pytest.mark.parametrize("block_size", [None, 256 * 1024])
def test_tar_member_versioned_gcs_read_ahead(tmp_path, monkeypatch, block_size):
    data = b"m" * (100 * 1024)
    raw = _tar_bytes(data)
    client = GCSClient(
        "bucket",
        {"token": "anon", "default_block_size": 2 * 1024 * 1024, "version_aware": True},
        Cache(str(tmp_path / "cache"), str(tmp_path / "tmp")),
    )
    fs = client.fs
    monkeypatch.setattr(fs, "info", lambda *a, **kw: {"size": len(raw)})
    open_file = fs.open

    def open_readahead(path, **kwargs):
        return open_file(path, cache_type="readahead", **kwargs)

    monkeypatch.setattr(fs, "open", open_readahead)
    calls = []

    def fetch_range(file, start, end):
        assert isinstance(file, GCSFile)
        assert file.path == "bucket/archive.tar#123"
        assert file.generation == "123"
        calls.append((start, min(end, len(raw))))
        return raw[start:end]

    monkeypatch.setattr(GCSFile, "_fetch_range", fetch_range)
    _, member = _tar_member(client, raw, version="123")
    with member.open(block_size=block_size) as stream:
        assert stream.read() == data
        assert stream.read() == b""

    assert calls[0][0] == member.location[0]["offset"] > 0
    fetched = sum(end - start for start, end in calls)
    expected_block = len(data) if block_size is None else block_size
    assert len(data) <= fetched <= len(data) + expected_block + io.DEFAULT_BUFFER_SIZE
    if block_size is not None:
        assert fetched >= block_size
