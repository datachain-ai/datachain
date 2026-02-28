import json
import os
import posixpath
from pathlib import Path
from unittest.mock import Mock

import pytest
from fsspec.implementations.local import LocalFileSystem
from PIL import Image

from datachain.catalog import Catalog
from datachain.fs.utils import path_to_fsspec_uri
from datachain.lib.file import (
    Audio,
    File,
    FileError,
    ImageFile,
    TextFile,
    VFileError,
    resolve,
)


@pytest.fixture
def create_file(catalog: Catalog):
    def _create_file(
        source: str,
        caching_enabled: bool = False,
        *,
        path: str = "dir1/dir2/test.txt",
        etag: str = "ed779276108738fdb2179ccabf9680d9",
    ) -> File:
        file = File(
            path=path,
            source=source,
            etag=etag,
        )
        file._set_stream(catalog, caching_enabled)
        return file

    return _create_file


def test_file_stem():
    s = File(path=".file.jpg.txt")
    assert s.get_file_stem() == ".file.jpg"


def test_file_ext():
    s = File(path=".file.jpg.txt")
    assert s.get_file_ext() == "txt"


def test_file_suffix():
    s = File(path=".file.jpg.txt")
    assert s.get_file_suffix() == ".txt"


def test_cache_get_path(catalog: Catalog):
    stream = File(path="test.txt1", source="s3://mybkt")
    stream._set_stream(catalog)

    data = b"some data is heRe"
    catalog.cache.store_data(stream, data)

    path = stream.get_local_path()
    assert path is not None

    with open(path, mode="rb") as f:
        assert f.read() == data


def test_export_unsupported_placement_raises(create_file, tmp_path):
    file = create_file("s3://mybkt")

    with pytest.raises(ValueError, match="Unsupported file export placement"):
        file.export(tmp_path, placement="wrong")  # type: ignore[arg-type]


def test_save_requires_catalog(create_file, tmp_path):
    file = create_file("s3://mybkt")
    object.__setattr__(file, "_catalog", None)
    with pytest.raises(RuntimeError, match="Cannot save file: catalog is not set"):
        file.save(tmp_path / "out.txt")


def test_export_requires_catalog(create_file, tmp_path):
    file = create_file("s3://mybkt")
    object.__setattr__(file, "_catalog", None)
    with pytest.raises(RuntimeError, match="Cannot export file: catalog is not set"):
        file.export(tmp_path / "output", use_cache=False)


def _output_dot(_tmp_path: Path) -> str:
    return "."


def _output_dotdot(_tmp_path: Path) -> str:
    return ".."


def _output_relative(_tmp_path: Path) -> str:
    return "relout"


def _output_empty(_tmp_path: Path) -> str:
    return ""


def _output_absolute(tmp_path: Path) -> Path:
    return tmp_path / "absolute"


@pytest.mark.parametrize(
    "source,expected_fullpath_prefix",
    [
        ("s3://mybkt", "mybkt"),
        ("gs://mybkt", "mybkt"),
        ("az://container", "container"),
        ("file://bucket", None),
        ("file:///", None),
    ],
)
@pytest.mark.parametrize(
    "output_factory",
    [
        _output_dot,
        _output_empty,
        _output_relative,
        _output_absolute,
        _output_dotdot,
    ],
    ids=["output=.", "output=empty", "output=relative", "output=absolute", "output=.."],
)
def test_export_placements_build_expected_destination_for_each_source(
    create_file,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    expected_fullpath_prefix: str | None,
    output_factory,
):
    file = create_file(source)

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    output = output_factory(tmp_path)

    saved: list[str] = []

    monkeypatch.setattr(
        File,
        "save",
        lambda _self, destination, client_config=None: saved.append(destination),
    )

    file.export(output, placement="filename", use_cache=False)
    file.export(output, placement="etag", use_cache=False)
    file.export(output, placement="filepath", use_cache=False)
    file.export(output, placement="fullpath", use_cache=False)

    # export() resolves output to absolute (os.path.abspath) and normalizes
    # backslashes to forward slashes on Windows, so build the expected prefix
    # the same way.
    expected_prefix = os.path.abspath(os.fspath(output)).replace("\\", "/")

    expected_fullpath_suffix = (
        f"{expected_fullpath_prefix}/dir1/dir2/test.txt"
        if expected_fullpath_prefix
        else "dir1/dir2/test.txt"
    )

    assert saved == [
        posixpath.join(expected_prefix, "test.txt"),
        posixpath.join(expected_prefix, "ed779276108738fdb2179ccabf9680d9.txt"),
        posixpath.join(expected_prefix, "dir1/dir2/test.txt"),
        posixpath.join(expected_prefix, expected_fullpath_suffix),
    ]


def test_export_rejects_traversal_path(create_file, tmp_path):
    file = create_file("s3://mybkt", path="../escape.txt", etag="etag")

    output = tmp_path / "output"
    with pytest.raises(FileError, match=r"must not contain"):
        file.export(output, placement="filepath", use_cache=False)


def test_export_rejects_traversal(create_file, tmp_path):
    file = create_file("s3://mybkt", path="..", etag="etag")

    output = tmp_path / "output"
    with pytest.raises(FileError, match=r"must not contain"):
        file.export(output, placement="filename", use_cache=False)


def test_export_rejects_empty_segments(create_file, tmp_path):
    file = create_file("s3://mybkt", path="dir//file.txt", etag="etag")

    output = tmp_path / "output"
    with pytest.raises(FileError, match="must not contain empty segments"):
        file.export(output, placement="filepath", use_cache=False)


def test_export_local_output_allows_literal_percent_encoded_traversal(
    create_file, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    payload = "..%2f..%2fescape.txt"  # literal percent-encoding; must not be decoded
    file = create_file("s3://mybkt")
    object.__setattr__(file, "path", payload)

    output = tmp_path / "output"
    saved: list[str] = []

    monkeypatch.setattr(
        File,
        "save",
        lambda _self, destination, client_config=None: saved.append(destination),
    )

    file.export(output, placement="filepath", use_cache=False)
    file.export(output, placement="fullpath", use_cache=False)

    for dst in saved:
        assert Path(dst).resolve().is_relative_to(output.resolve())


@pytest.mark.parametrize(
    "source,path,is_error,expected",
    [
        ("", "", True, r"must not be empty"),
        ("", ".", True, r"must not contain"),
        ("", "..", True, r"must not contain"),
        ("", "/abs/file.txt", True, r"must not be absolute"),
        ("", "file#hash.txt", False, "file#hash.txt"),
        ("", "./dir/../file#hash.txt", True, r"must not contain"),
        ("", "../escape.txt", True, r"must not contain"),
        ("", "dir//file.txt", True, r"must not contain empty segments"),
        ("", "dir/", True, r"must not be a directory"),
        # file:// source with drive letter — valid on both platforms.
        # Path validation fires first regardless of OS.
        ("file:///C:/bucket", "", True, r"must not be empty"),
        ("file:///C:/bucket", ".", True, r"must not contain"),
        ("file:///C:/bucket", "..", True, r"must not contain"),
        ("file:///C:/bucket", "/abs/file.txt", True, r"must not be absolute"),
        ("file:///C:/bucket", "./dir/../file.txt", True, r"must not contain"),
        pytest.param(
            "file:///bucket",
            "file#hash.txt",
            os.name == "nt",
            r"drive letter" if os.name == "nt" else "file:///bucket/file#hash.txt",
            id="get_uri-file:///bucket-file#hash.txt",
        ),
        pytest.param(
            "file:///bucket",
            "dir/file#hash.txt",
            os.name == "nt",
            r"drive letter" if os.name == "nt" else "file:///bucket/dir/file#hash.txt",
            id="get_uri-file:///bucket-dir/file#hash.txt",
        ),
        # Windows-style file:// URIs with drive letters (valid on all platforms).
        ("file:///C:/data", "file.txt", False, "file:///C:/data/file.txt"),
        (
            "file:///D:/path/to/dir",
            "sub/out.csv",
            False,
            "file:///D:/path/to/dir/sub/out.csv",
        ),
        ("s3://mybkt", "", True, r"path must not be empty"),
        ("s3://mybkt", ".", True, r"must not contain"),
        ("s3://mybkt", "..", True, r"must not contain"),
        ("s3://mybkt", "/abs/file.txt", False, "s3://mybkt//abs/file.txt"),
        ("s3://mybkt", "dir/file#frag.txt", False, "s3://mybkt/dir/file#frag.txt"),
        ("gs://mybkt", "", True, r"path must not be empty"),
        ("gs://mybkt", ".", True, r"must not contain"),
        ("gs://mybkt", "..", True, r"must not contain"),
        ("gs://mybkt", "/abs/file.txt", False, "gs://mybkt//abs/file.txt"),
        ("gs://mybkt", "dir/file#frag.txt", False, "gs://mybkt/dir/file#frag.txt"),
        (
            "az://container",
            "",
            True,
            r"path must not be empty",
        ),
        (
            "az://container",
            ".",
            True,
            r"must not contain",
        ),
        (
            "az://container",
            "..",
            True,
            r"must not contain",
        ),
        (
            "az://container",
            "/abs/file.txt",
            False,
            "az://container//abs/file.txt",
        ),
        (
            "az://container",
            "dir/file#frag.txt",
            False,
            "az://container/dir/file#frag.txt",
        ),
    ],
)
def test_get_uri_contract(
    source: str,
    path: str,
    is_error: bool,
    expected: str,
):
    file = File(path=path, source=source)

    if is_error:
        with pytest.deprecated_call():
            with pytest.raises(FileError, match=expected):
                file.get_uri()
        return

    with pytest.deprecated_call():
        uri = file.get_uri()

    assert uri == expected


@pytest.mark.parametrize(
    "source,path,is_error,expected",
    [
        ("", "", True, r"must not be empty"),
        ("", ".", True, r"must not contain"),
        ("", "..", True, r"must not contain"),
        ("", "/abs/file.txt", True, r"must not be absolute"),
        ("", "file.txt", False, "file.txt"),
        ("", "../escape.txt", True, r"must not contain"),
        ("", "dir/", True, r"must not be a directory"),
        # file:// source with drive letter — valid on both platforms.
        # Path validation fires first regardless of OS.
        ("file:///C:/bucket", "", True, r"must not be empty"),
        ("file:///C:/bucket", ".", True, r"must not contain"),
        ("file:///C:/bucket", "..", True, r"must not contain"),
        ("file:///C:/bucket", "/abs/file.txt", True, r"must not be absolute"),
        ("file:///C:/bucket", "./dir/../file.txt", True, r"must not contain"),
        pytest.param(
            "file:///bucket",
            "file#hash.txt",
            os.name == "nt",
            r"drive letter" if os.name == "nt" else "/bucket/file#hash.txt",
            id="file:///bucket-file#hash.txt",
        ),
        # Windows-style file:// URIs with drive letters (valid on all platforms).
        # On Unix, _strip_protocol keeps the leading '/' before the drive letter.
        pytest.param(
            "file:///C:/data",
            "file.txt",
            False,
            "C:/data/file.txt" if os.name == "nt" else "/C:/data/file.txt",
            id="drive-C-data-file",
        ),
        pytest.param(
            "file:///D:/path/to/dir",
            "sub/out.csv",
            False,
            "D:/path/to/dir/sub/out.csv"
            if os.name == "nt"
            else "/D:/path/to/dir/sub/out.csv",
            id="drive-D-nested",
        ),
        pytest.param(
            "file:///C:/",
            "readme.md",
            False,
            "C:/readme.md" if os.name == "nt" else "/C:/readme.md",
            id="drive-C-root",
        ),
        ("s3://mybkt", "", True, r"path must not be empty"),
        ("s3://mybkt", ".", True, r"must not contain"),
        ("s3://mybkt", "..", True, r"must not contain"),
        ("s3://mybkt", "/abs/file.txt", False, "s3://mybkt//abs/file.txt"),
        ("s3://mybkt", "dir/file#frag.txt", False, "s3://mybkt/dir/file#frag.txt"),
        ("gs://mybkt", "", True, r"path must not be empty"),
        ("gs://mybkt", ".", True, r"must not contain"),
        ("gs://mybkt", "..", True, r"must not contain"),
        ("gs://mybkt", "/abs/file.txt", False, "gs://mybkt//abs/file.txt"),
        ("gs://mybkt", "dir/file#frag.txt", False, "gs://mybkt/dir/file#frag.txt"),
        ("az://container", "", True, r"path must not be empty"),
        ("az://container", ".", True, r"must not contain"),
        ("az://container", "..", True, r"must not contain"),
        ("az://container", "/abs/file.txt", False, "az://container//abs/file.txt"),
        (
            "az://container",
            "dir/file#frag.txt",
            False,
            "az://container/dir/file#frag.txt",
        ),
    ],
)
def test_get_fs_path_contract(
    source: str,
    path: str,
    is_error: bool,
    expected: str,
):
    file = File(path=path, source=source)

    if is_error:
        with pytest.raises(FileError, match=expected):
            file.get_fs_path()
        return

    fs_path = file.get_fs_path()

    assert fs_path.replace("\\", "/") == expected


def test_read_binary_data(tmp_path, catalog: Catalog):
    file_name = "myfile"
    data = b"some\x00data\x00is\x48\x65\x6c\x57\x6f\x72\x6c\x64\xff\xffheRe"

    file_path = tmp_path / file_name
    with open(file_path, "wb") as fd:
        fd.write(data)

    file = File(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, False)
    assert file.read() == data


def test_read_binary_data_as_text(tmp_path, catalog: Catalog):
    file_name = "myfile43.txt"
    data = b"some\x00data\x00is\x48\x65\x6c\x57\x6f\x72\x6c\x64\xff\xffheRe"

    file_path = tmp_path / file_name
    with open(file_path, "wb") as fd:
        fd.write(data)

    file = TextFile(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, False)
    try:
        x = file.read()
    except UnicodeDecodeError:  # Unix
        pass
    else:  # Windows
        assert x != data


def test_read_text_data(tmp_path, catalog: Catalog):
    file_name = "myfile"
    data = "this is a TexT data..."

    file_path = tmp_path / file_name
    with open(file_path, "w") as fd:
        fd.write(data)

    file = TextFile(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, True)
    assert file.read() == data


def test_read_file_as_text(tmp_path, catalog: Catalog):
    file_name = "myfile"
    data = "this is a TexT data..."

    file_path = tmp_path / file_name
    with open(file_path, "w") as fd:
        fd.write(data)

    file = File(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, True)
    assert file.as_text_file().read() == data
    assert file.as_text_file().as_text_file().read() == data


def test_save_binary_data(tmp_path, catalog: Catalog):
    file1_name = "myfile1"
    file2_name = "myfile2"
    data = b"some\x00data\x00is\x48\x65\x6c\x57\x6f\x72\x6c\x64\xff\xffheRe"

    with open(tmp_path / file1_name, "wb") as fd:
        fd.write(data)

    file1 = File(path=file1_name, source=f"file://{tmp_path}")
    file1._set_stream(catalog, False)

    file1.save(tmp_path / file2_name)

    file2 = File(path=file2_name, source=f"file://{tmp_path}")
    file2._set_stream(catalog, False)
    assert file2.read() == data


def test_save_accepts_file_fsspec_uri_destination(tmp_path, catalog: Catalog):
    data = b"uri destination test"
    (tmp_path / "src.bin").write_bytes(data)
    dest = tmp_path / "dest.bin"
    file = File(path="src.bin", source=f"file://{tmp_path}")
    file._set_stream(catalog, False)
    file.save(path_to_fsspec_uri(str(dest)))
    assert dest.read_bytes() == data


def test_save_rfc_encoded_uri_writes_to_literal_path(tmp_path, catalog: Catalog):
    # Path.as_uri() encodes the space as %20; fsspec doesn't decode it,
    # so save() writes to the literal "dir%20name" dir, not "dir name".
    space_dir = tmp_path / "dir name"
    space_dir.mkdir()
    (tmp_path / "src.bin").write_bytes(b"data")
    file = File(path="src.bin", source=f"file://{tmp_path}")
    file._set_stream(catalog, False)
    file.save((space_dir / "out.bin").as_uri())
    assert not (space_dir / "out.bin").exists()
    assert (tmp_path / "dir%20name" / "out.bin").exists()


def test_save_text_data(tmp_path, catalog: Catalog):
    file1_name = "myfile1.txt"
    file2_name = "myfile2.txt"
    data = "some text"

    with open(tmp_path / file1_name, "w") as fd:
        fd.write(data)

    file1 = TextFile(path=file1_name, source=f"file://{tmp_path}")
    file1._set_stream(catalog, False)

    file1.save(tmp_path / file2_name)

    file2 = TextFile(path=file2_name, source=f"file://{tmp_path}")
    file2._set_stream(catalog, False)
    assert file2.read() == data


def test_save_image_data(tmp_path, catalog: Catalog):
    from tests.utils import images_equal

    file1_name = "myfile1.jpg"
    file2_name = "myfile2.jpg"

    image = Image.new(mode="RGB", size=(64, 64))
    image.save(tmp_path / file1_name)

    file1 = ImageFile(path=file1_name, source=f"file://{tmp_path}")
    file1._set_stream(catalog, False)

    file1.save(tmp_path / file2_name)

    file2 = ImageFile(path=file2_name, source=f"file://{tmp_path}")
    file2._set_stream(catalog, False)
    assert images_equal(image, file2.read())


def test_cache_get_path_without_cache():
    stream = File(path="test.txt1", source="s3://mybkt")
    with pytest.raises(RuntimeError):
        stream.get_local_path()


def test_json_from_string():
    d = {"e": 12}

    file = File(path="something", location=d)
    assert file.location == d

    file = File(path="something", location=None)
    assert file.location is None

    file = File(path="something", location="")
    assert file.location is None

    file = File(path="something", location=json.dumps(d))
    assert file.location == d

    with pytest.raises(ValueError):
        File(path="something", location="{not a json}")


def test_file_info_jsons():
    file = File(path="something", location="")
    assert file.location is None

    d = {"e": 12}
    file = File(path="something", location=json.dumps(d))
    assert file.location == d


def test_get_fs(catalog):
    file = File(path="dir/file", source="file:///")
    file._catalog = catalog
    assert isinstance(file.get_fs(), LocalFileSystem)


def test_open_mode(tmp_path, catalog: Catalog):
    file_name = "myfile"
    data = "this is a TexT data..."

    file_path = tmp_path / file_name
    with open(file_path, "w") as fd:
        fd.write(data)

    file = File(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, True)
    with file.open(mode="r") as stream:
        assert stream.read() == data


def test_read_length(tmp_path, catalog):
    file_name = "myfile"
    data = b"some\x00data\x00is\x48\x65\x6c\x57\x6f\x72\x6c\x64\xff\xffheRe"

    file_path = tmp_path / file_name
    with open(file_path, "wb") as fd:
        fd.write(data)

    file = File(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, False)
    assert file.read(length=4) == data[:4]


def test_read_bytes(tmp_path, catalog):
    file_name = "myfile"
    data = b"some\x00data\x00is\x48\x65\x6c\x57\x6f\x72\x6c\x64\xff\xffheRe"

    file_path = tmp_path / file_name
    with open(file_path, "wb") as fd:
        fd.write(data)

    file = File(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, False)
    assert file.read_bytes() == data


def test_read_text(tmp_path, catalog):
    file_name = "myfile"
    data = "this is a TexT data..."

    file_path = tmp_path / file_name
    with open(file_path, "w") as fd:
        fd.write(data)

    file = File(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, True)
    assert file.read_text() == data


def test_resolve_unsupported_protocol():
    mock_catalog = Mock()
    mock_catalog.get_client.side_effect = NotImplementedError("Unsupported protocol")

    file = File(source="unsupported://example.com", path="test.txt")
    file._catalog = mock_catalog

    with pytest.raises(RuntimeError) as exc_info:
        file.resolve()

    assert (
        str(exc_info.value)
        == "Unsupported protocol for file source: unsupported://example.com"
    )


def test_file_resolve_no_catalog():
    file = File(path="test.txt", source="s3://mybucket")
    with pytest.raises(RuntimeError, match="Cannot resolve file: catalog is not set"):
        file.resolve()


def test_file_resolve_virtual_file_raises():
    location = [
        {
            "vtype": "tar",
            "parent": {"source": "s3://bucket", "path": "archive.tar"},
            "offset": 0,
            "size": 10,
        }
    ]
    file = File(source="s3://bucket", path="entry.txt", location=location)
    file._catalog = Mock()
    with pytest.raises(VFileError, match="Resolving a virtual file is not supported"):
        file.resolve()


@pytest.mark.parametrize("bad_path", ["dir/", "a/../b"])
def test_file_resolve_invalid_path_raises(bad_path, catalog):
    # validate_file_path raises ValueError for bad paths; resolve() must not swallow it.
    file = File(source="file:///tmp", path=bad_path)
    file._set_stream(catalog)
    with pytest.raises(ValueError):
        file.resolve()


def test_file_resolve_sets_catalog(tmp_path, catalog):
    # https://github.com/datachain-ai/datachain/pull/1393
    file_name = "myfile"
    file_path = tmp_path / file_name
    file_path.write_text("this is a TexT data...")

    file = File(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, True)

    file_path.write_text("new data")
    new_file = file.resolve()

    assert new_file._catalog is catalog
    new_file.ensure_cached()
    assert new_file.read_text() == "new data"


def test_resolve_function():
    mock_file = Mock(spec=File)
    mock_file.resolve.return_value = "resolved_file"

    with pytest.deprecated_call(match=r"resolve\(\) is deprecated"):
        result = resolve(mock_file)

    assert result == "resolved_file"
    mock_file.resolve.assert_called_once()


def test_get_local_path(tmp_path, catalog):
    file_name = "myfile"
    data = b"some\x00data\x00is\x48\x65\x6c\x57\x6f\x72\x6c\x64\xff\xffheRe"

    file_path = tmp_path / file_name
    with open(file_path, "wb") as fd:
        fd.write(data)

    file = File(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog)

    assert file.get_local_path() is None
    file.ensure_cached()
    assert file.get_local_path() is not None


@pytest.mark.parametrize("use_cache", (True, False))
def test_export_with_symlink(tmp_path, catalog, use_cache):
    path = tmp_path / "myfile.txt"
    path.write_text("some text")

    file = File(path=path.name, source=tmp_path.as_uri())
    file._set_stream(catalog, use_cache)

    file.export(tmp_path / "dir", link_type="symlink", use_cache=use_cache)
    assert (tmp_path / "dir" / "myfile.txt").is_symlink()

    if use_cache:
        cached_path = file.get_local_path()
        assert cached_path is not None
        dst = Path(cached_path)
    else:
        dst = path

    assert (tmp_path / "dir" / "myfile.txt").resolve() == dst


def test_export_filename_percent_encoded_traversal_writes_outside_output(
    tmp_path, catalog
):
    payload = "..%2fescaped.txt"  # unquote -> ../escaped.txt
    (tmp_path / payload).write_text("poc")

    file = File(path=payload, source=tmp_path.as_uri())
    file._set_stream(catalog, False)

    output = tmp_path / "output"
    file.export(output, placement="filename", use_cache=False)

    # Expected-safe behavior: exported files must stay within the output dir.
    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.parametrize(
    "path,expected",
    [
        ("", ""),
        (".", "."),
        ("dir/file.txt", "dir/file.txt"),
        ("../dir/file.txt", "../dir/file.txt"),
        ("/dir/file.txt", "/dir/file.txt"),
    ],
)
def test_path_validation(path, expected):
    assert File(path=path, source="file:///").path == expected


def test_file_rebase_method():
    """Test File.rebase() method"""
    file = File(source="s3://bucket", path="data/audio/file.wav")

    # Basic rebase
    result = file.rebase("s3://bucket/data/audio", "s3://output-bucket/waveforms")
    assert result == "s3://output-bucket/waveforms/file.wav"

    # With suffix and extension
    result = file.rebase(
        "s3://bucket/data/audio",
        "s3://output-bucket/processed",
        suffix="_ch1",
        extension="npy",
    )
    assert result == "s3://output-bucket/processed/file_ch1.npy"


@pytest.mark.parametrize(
    "source,old_base,new_base,expected",
    [
        pytest.param(
            "file:///data/audio",
            "file:///data/audio",
            "/output/processed",
            "/output/processed/folder/file.mp3",
            id="unix",
            marks=pytest.mark.skipif(
                os.name == "nt", reason="No drive letter - rejected on Windows"
            ),
        ),
        pytest.param(
            "file:///C:/data/audio",
            "C:/data/audio",
            "C:/output/processed",
            "C:/output/processed/folder/file.mp3",
            id="windows",
            # old_base uses plain path form because _split_scheme("file:///C:/…")
            # yields "/C:/…" with a leading slash that won't match the native
            # path returned by get_fs_path() on Windows (C:\…).
            marks=pytest.mark.skipif(
                os.name != "nt", reason="Windows drive-letter path"
            ),
        ),
    ],
)
def test_file_rebase_local_path(source, old_base, new_base, expected):
    """Test File.rebase() with local file paths"""
    file = File(source=source, path="folder/file.mp3")
    result = file.rebase(old_base, new_base)
    assert result == expected


def test_audio_get_channel_name():
    # Test known channel configurations
    assert Audio.get_channel_name(1, 0) == "Mono"
    assert Audio.get_channel_name(2, 0) == "Left"
    assert Audio.get_channel_name(2, 1) == "Right"
    assert Audio.get_channel_name(4, 2) == "Y"  # Ambisonics
    assert Audio.get_channel_name(6, 3) == "LFE"  # 5.1 surround
    assert Audio.get_channel_name(8, 7) == "SR"  # 7.1 surround

    # Test fallback for unknown configurations
    assert Audio.get_channel_name(-1, 0) == "Ch1"
    assert Audio.get_channel_name(3, 0) == "Ch1"
    assert Audio.get_channel_name(5, 4) == "Ch5"
    assert Audio.get_channel_name(10, 9) == "Ch10"

    # Test out of range indices
    assert Audio.get_channel_name(2, 5) == "Ch6"
    assert Audio.get_channel_name(1, 1) == "Ch2"
