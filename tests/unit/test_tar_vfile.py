import io
import tarfile
from types import SimpleNamespace

import fsspec
import pytest
from fsspec.spec import AbstractBufferedFile

from datachain.lib.file import PARTIAL_READ_BLOCK_SIZE, File, VFileError
from datachain.lib.tar import build_tar_member


class RangeFile(AbstractBufferedFile):
    def __init__(self, backend, block_size):
        self.backend = backend
        super().__init__(
            fs=fsspec.filesystem("file"),
            path=str(backend.path),
            mode="rb",
            size=backend.path.stat().st_size,
            block_size=block_size,
            cache_type="readahead",
        )

    def _fetch_range(self, start, end):
        with self.backend.path.open("rb") as stream:
            stream.seek(start)
            data = stream.read(end - start)
        self.backend.fetched_bytes += len(data)
        return data


class RangeBackend:
    def __init__(self, path, cached):
        self.path = path
        self.cached = cached
        self.fetched_bytes = 0
        self.handles = []

    def open_object(self, file, use_cache=True, cb=None, block_size=None):
        if self.cached and use_cache:
            stream = self.path.open("rb")
        else:
            stream = RangeFile(
                self, 50 * 1024 * 1024 if block_size is None else block_size
            )
        self.handles.append(stream)
        return stream


@pytest.fixture
def make_member(tmp_path):
    def make(contents, cached=False):
        path = tmp_path / "archive.tar"
        with tarfile.open(path, "w") as archive:
            info = tarfile.TarInfo("member.bin")
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))
            padding = tarfile.TarInfo("padding.bin")
            padding.size = 4 * 1024 * 1024
            archive.addfile(padding, io.BytesIO(b"p" * padding.size))
        with tarfile.open(path) as archive:
            info = archive.getmember("member.bin")

        backend = RangeBackend(path, cached)
        catalog = SimpleNamespace(
            client_config={}, get_client=lambda *args, **kwargs: backend
        )
        parent = File(source="file:///", path=path.name, size=path.stat().st_size)
        member = build_tar_member(parent, info)
        member._set_stream(catalog, caching_enabled=False)
        return member, backend

    return make


@pytest.mark.parametrize("size", [0, 1, 100 * 1024, 2 * PARTIAL_READ_BLOCK_SIZE])
def test_tar_member_default_read_ahead(make_member, size):
    contents = b"m" * size
    member, backend = make_member(contents)

    with member.open() as stream:
        assert stream.read() == contents

    if size:
        assert (
            size
            <= backend.fetched_bytes
            <= (size + min(size, PARTIAL_READ_BLOCK_SIZE) + io.DEFAULT_BUFFER_SIZE)
        )
    else:
        assert backend.fetched_bytes == 0
    assert all(stream.closed for stream in backend.handles)


@pytest.mark.parametrize("block_size", [0, 32 * 1024, 2 * PARTIAL_READ_BLOCK_SIZE])
def test_tar_member_explicit_read_ahead(make_member, block_size):
    contents = b"m" * (100 * 1024)
    member, backend = make_member(contents)

    with member.open(block_size=block_size) as stream:
        assert stream.read(256) == contents[:256]

    assert backend.fetched_bytes == block_size + 256
    assert all(stream.closed for stream in backend.handles)


def test_tar_member_explicit_none_read_ahead(make_member):
    member, backend = make_member(b"m" * 1024)

    with member.open(block_size=None) as stream:
        assert stream.read(1) == b"m"

    assert backend.fetched_bytes <= 1025
    assert all(stream.closed for stream in backend.handles)


def test_tar_member_cached_parent(make_member):
    contents = b"cached member"
    member, backend = make_member(contents, cached=True)

    with member.open() as stream:
        assert stream.read() == contents

    assert backend.fetched_bytes == 0
    assert all(stream.closed for stream in backend.handles)


def test_tar_member_nested_tar_seek(make_member):
    inner = io.BytesIO()
    with tarfile.open(fileobj=inner, mode="w") as archive:
        info = tarfile.TarInfo("inner.bin")
        info.size = len(b"nested")
        archive.addfile(info, io.BytesIO(b"nested"))
    member, backend = make_member(inner.getvalue())

    with member.open() as stream, tarfile.open(fileobj=stream) as archive:
        with archive.extractfile("inner.bin") as extracted:
            assert extracted.read() == b"nested"

    assert all(stream.closed for stream in backend.handles)


def test_tar_member_truncated_parent(make_member):
    contents = b"m" * 1024
    member, backend = make_member(contents)
    member.location[0]["size"] = len(contents) + 1
    with backend.path.open("r+b") as stream:
        stream.truncate(member.location[0]["offset"] + len(contents))

    with member.open() as stream, pytest.raises(RuntimeError, match="unexpected end"):
        stream.read()

    assert all(stream.closed for stream in backend.handles)


def test_tar_member_multiple_locations(make_member):
    member, backend = make_member(b"member")
    member.location.append(member.location[0].copy())

    with pytest.raises(VFileError, match="multiple 'location's"):
        with member.open():
            pass

    assert not backend.handles
