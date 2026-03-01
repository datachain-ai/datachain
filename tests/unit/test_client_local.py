import os
from pathlib import Path

import pytest

from datachain.client.local import FileClient
from datachain.fs.utils import path_to_fsspec_uri
from datachain.lib.file import File, FileError


def test_split_url_directory_preserves_leaf(tmp_path):
    uri = path_to_fsspec_uri(str(tmp_path))
    bucket, rel = FileClient.split_url(uri)

    # For directories, the parent directory becomes the bucket and the leaf
    # directory becomes the relative path to keep downstream listings stable.
    assert Path(bucket) == tmp_path.parent
    assert rel == tmp_path.name


def test_split_url_file_in_directory(tmp_path):
    file_path = tmp_path / "sub" / "file.bin"
    file_path.parent.mkdir(parents=True)
    uri = path_to_fsspec_uri(str(file_path))

    bucket, rel = FileClient.split_url(uri)

    # The bucket should be the directory containing the file;
    # rel should be the filename.
    assert Path(bucket) == file_path.parent
    assert rel == "file.bin"


def test_split_url_accepts_plain_directory_path(tmp_path):
    bucket, rel = FileClient.split_url(str(tmp_path))

    assert Path(bucket) == tmp_path.parent
    assert rel == tmp_path.name


def test_split_url_accepts_plain_file_path(tmp_path):
    file_path = tmp_path / "leaf.txt"
    file_path.write_text("data")

    bucket, rel = FileClient.split_url(str(file_path))

    assert Path(bucket) == file_path.parent
    assert rel == "leaf.txt"


def test_path_to_fsspec_uri_preserves_trailing_slash(tmp_path):
    dir_path = tmp_path / "trail"
    dir_path.mkdir()

    uri = path_to_fsspec_uri(f"{dir_path}{os.sep}")

    base_uri = path_to_fsspec_uri(str(dir_path))

    # Trailing separator in the input should keep a trailing slash in the URI.
    assert uri.endswith("/")
    assert uri[:-1] == base_uri


def test_path_to_fsspec_uri_passes_through_existing_uri(tmp_path):
    uri = path_to_fsspec_uri(str(tmp_path))

    assert path_to_fsspec_uri(uri) == uri


def test_path_to_fsspec_uri_keeps_chars_literal(tmp_path):
    base = tmp_path / "dir #% percent"
    base.mkdir(parents=True)

    uri = path_to_fsspec_uri(str(base))
    assert uri.startswith("file://")
    assert " " in uri
    assert "#" in uri
    assert "%" in uri


def test_split_url_does_not_decode_percent_escapes(tmp_path):
    # If the filename literally contains percent-escapes, split_url must not
    # decode them (e.g. %2f -> '/').
    file_path = tmp_path / "file%2fescape%23hash.txt"
    file_path.write_text("x", encoding="utf-8")

    uri = path_to_fsspec_uri(str(file_path))
    bucket, rel = FileClient.split_url(uri)

    assert Path(bucket) == tmp_path
    assert rel == file_path.name


@pytest.mark.parametrize(
    "url,expected_unix,expected_nt",
    [
        # Backslash: literal char on Unix, converted to / by fsspec on Windows
        (
            "file:///tmp/data/foo\\bar",
            ("/tmp/data", "foo\\bar"),  # noqa: S108
            ("C:/tmp/data/foo", "bar"),
        ),
        # Colon: preserved literally on both platforms
        (
            "file:///tmp/data/HH:MM:SS.txt",
            ("/tmp/data", "HH:MM:SS.txt"),  # noqa: S108
            ("C:/tmp/data", "HH:MM:SS.txt"),
        ),
        # Both backslash and colon
        (
            "file:///tmp/data/a\\b:c",
            ("/tmp/data", "a\\b:c"),  # noqa: S108
            ("C:/tmp/data/a", "b:c"),
        ),
        # Multiple backslashes become extra path segments on Windows
        (
            "file:///tmp/data/a\\b\\c",
            ("/tmp/data", "a\\b\\c"),  # noqa: S108
            ("C:/tmp/data/a/b", "c"),
        ),
    ],
)
def test_split_url_preserves_backslash_and_colon(url, expected_unix, expected_nt):
    """split_url uses forward-slash splitting only.

    On Unix, backslashes and colons are valid filename characters and are
    preserved literally.  On Windows, fsspec's make_path_posix converts
    backslashes to forward slashes *before* split_url splits, so the result
    has extra path segments instead.
    """
    bucket, rel = FileClient.split_url(url)
    if os.name == "nt":
        assert (bucket, rel) == expected_nt
    else:
        assert (bucket, rel) == expected_unix


@pytest.mark.parametrize(
    "path,expected,raises",
    [
        ("", None, "must not be empty"),
        ("/", None, "must not be a directory"),
        (".", None, "must not contain"),
        ("dir/..", None, "must not contain"),
        ("dir/file.txt", "dir/file.txt", None),
        ("dir//file.txt", None, "must not contain empty segments"),
        ("./dir/file.txt", None, "must not contain"),
        ("dir/./file.txt", None, "must not contain"),
        ("dir/../file.txt", None, "must not contain"),
        ("dir/foo/../file.txt", None, "must not contain"),
        ("./dir/./foo/.././../file.txt", None, "must not contain"),
        ("./dir", None, "must not contain"),
        ("dir/.", None, "must not contain"),
        ("./dir/.", None, "must not contain"),
        ("/dir", None, "must not be absolute"),
        ("/dir/file.txt", None, "must not be absolute"),
        ("/dir/../file.txt", None, "must not contain"),
        ("..", None, "must not contain"),
        ("../file.txt", None, "must not contain"),
        ("dir/../../file.txt", None, "must not contain"),
    ],
)
def test_get_fs_path_validates_local_paths(path, expected, raises):
    source = "file:///C:/tmp" if os.name == "nt" else "file:///tmp"
    file = File(path=path, source=source)
    if raises:
        with pytest.raises(FileError, match=raises):
            file.get_fs_path()
    else:
        assert file.get_fs_path().endswith(expected)


@pytest.mark.skipif(os.name == "nt", reason="Backslash is a separator on Windows")
def test_validate_relpath_unix_backslash_in_filename_accepted():
    FileClient.validate_file_path("dir\\file.txt")


@pytest.mark.skipif(os.name == "nt", reason="Backslash is a separator on Windows")
def test_validate_relpath_unix_trailing_backslash_accepted():
    # Trailing backslash is NOT a directory indicator on Unix — it's part
    # of the filename.
    FileClient.validate_file_path("dir\\")


@pytest.mark.skipif(os.name == "nt", reason="Backslash is a separator on Windows")
def test_validate_relpath_unix_colon_in_filename_accepted():
    # Colons are legal on Unix; "C:foo" is a valid relative filename.
    FileClient.validate_file_path("C:foo.txt")


@pytest.mark.skipif(os.name != "nt", reason="Windows-only separator semantics")
def test_validate_relpath_win_trailing_backslash_rejected():
    with pytest.raises(ValueError, match="must not be a directory"):
        FileClient.validate_file_path("dir\\")


@pytest.mark.skipif(os.name != "nt", reason="Windows-only separator semantics")
def test_validate_relpath_win_absolute_backslash_rejected():
    with pytest.raises(ValueError, match="must not be absolute"):
        FileClient.validate_file_path("\\dir\\file.txt")


@pytest.mark.skipif(os.name != "nt", reason="Windows-only separator semantics")
def test_validate_relpath_win_drive_letter_rejected():
    with pytest.raises(ValueError, match="must not be absolute"):
        FileClient.validate_file_path("C:\\secret\\file.txt")


@pytest.mark.skipif(os.name != "nt", reason="Windows-only separator semantics")
def test_validate_relpath_win_drive_relative_rejected():
    with pytest.raises(ValueError, match="must not be absolute"):
        FileClient.validate_file_path("D:file.txt")


@pytest.mark.skipif(os.name != "nt", reason="Windows-only separator semantics")
def test_validate_relpath_win_dot_segment_via_backslash_rejected():
    with pytest.raises(ValueError, match="must not contain"):
        FileClient.validate_file_path("dir\\..\\file.txt")


@pytest.mark.skipif(os.name != "nt", reason="Windows-only separator semantics")
def test_validate_relpath_win_empty_segment_via_backslash_rejected():
    with pytest.raises(ValueError, match="must not contain empty segments"):
        FileClient.validate_file_path("dir\\\\file.txt")


@pytest.mark.parametrize(
    "source,expected",
    [
        ("file:///C:/data", True),
        ("file:///D:/path/to/dir", True),
        ("file:///c:/lowercase", True),
        ("file:///bucket", False),
        ("file:///data/dir", False),
        ("file:////network/share", False),
        ("file:///123/numeric", False),
    ],
)
def test_has_drive_letter(source, expected):
    assert FileClient._has_drive_letter(source) is expected


def test_get_fs_path_with_drive_letter_uri():
    file = File(source="file:///C:/data", path="sub/file.txt")
    result = file.get_fs_path()

    # get_fs_path returns a bare OS path for local files
    if os.name == "nt":
        assert result == "C:/data/sub/file.txt"
    else:
        assert result == "/C:/data/sub/file.txt"


def test_get_fs_path_drive_letter_root():
    file = File(source="file:///D:/", path="readme.md")
    result = file.get_fs_path()
    if os.name == "nt":
        assert result == "D:/readme.md"
    else:
        assert result == "/D:/readme.md"


def test_get_fs_path_nested_drive_letter():
    file = File(source="file:///C:/Users/me/projects", path="repo/data.csv")
    result = file.get_fs_path()
    if os.name == "nt":
        assert result == "C:/Users/me/projects/repo/data.csv"
    else:
        assert result == "/C:/Users/me/projects/repo/data.csv"


def test_file_at_with_relative_path(tmp_path, monkeypatch, test_session):
    monkeypatch.chdir(tmp_path)

    file = File.at("sub/file.txt", session=test_session)

    expected_source = path_to_fsspec_uri(str(tmp_path / "sub"))
    assert file.path == "file.txt"
    assert file.source == expected_source


def test_file_at_with_absolute_path(tmp_path, test_session):
    abs_path = tmp_path / "deep" / "nested" / "file.txt"

    file = File.at(str(abs_path), session=test_session)

    expected_source = path_to_fsspec_uri(str(abs_path.parent))
    assert file.path == "file.txt"
    assert file.source == expected_source


def test_file_at_with_file_uri(tmp_path, test_session):
    abs_path = tmp_path / "data" / "file.txt"
    file_uri = path_to_fsspec_uri(str(abs_path))

    file = File.at(file_uri, session=test_session)

    expected_source = path_to_fsspec_uri(str(abs_path.parent))
    assert file.path == "file.txt"
    assert file.source == expected_source


def test_file_at_relative_and_absolute_agree(tmp_path, monkeypatch, test_session):
    monkeypatch.chdir(tmp_path)

    from_rel = File.at("dir/file.bin", session=test_session)
    from_abs = File.at(str(tmp_path / "dir" / "file.bin"), session=test_session)
    from_uri = File.at(
        path_to_fsspec_uri(str(tmp_path / "dir" / "file.bin")), session=test_session
    )

    assert from_rel.source == from_abs.source == from_uri.source
    assert from_rel.path == from_abs.path == from_uri.path == "file.bin"


def test_file_at_source_is_file_uri(tmp_path, test_session):
    file = File.at(str(tmp_path / "file.txt"), session=test_session)

    assert file.source.startswith("file:///")


def test_file_at_path_inside_symlinked_directory(tmp_path, test_session):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir)

    file = File.at(str(link_dir / "file.txt"), session=test_session)

    assert "link" in file.source
    assert "real" not in file.source
    assert file.path == "file.txt"


def test_file_at_path_is_symlink_to_file(tmp_path, test_session):
    real_file = tmp_path / "src.txt"
    real_file.write_text("data")
    link_file = tmp_path / "link.txt"
    link_file.symlink_to(real_file)

    file = File.at(str(link_file), session=test_session)

    assert file.path == "link.txt"
    assert "src" not in file.path


def test_storage_uri_preserves_symlink_name(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir)

    uri = str(FileClient.storage_uri(str(link_dir)))
    assert "link" in uri
    assert "real" not in uri


def test_split_url_preserves_symlink_name(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir)

    _, rel = FileClient.split_url(str(link_dir))
    assert rel == "link"


def test_get_uri_preserves_symlink_root(tmp_path, catalog):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir)

    client = FileClient.from_source(str(link_dir), catalog.cache)
    uri = client.get_uri("file.txt")
    assert "link" in uri
    assert "real" not in uri


def test_file_save_destination_preserves_symlink(tmp_path, catalog):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir)

    (tmp_path / "src.txt").write_bytes(b"hello")
    file = File(path="src.txt", source=f"file://{tmp_path}")
    file._set_stream(catalog, False)

    file.save(str(link_dir / "out.txt"))

    assert (link_dir / "out.txt").read_bytes() == b"hello"
