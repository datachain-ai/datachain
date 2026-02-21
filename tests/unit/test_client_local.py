import os
from pathlib import Path

import pytest

from datachain.client.local import FileClient
from datachain.lib.file import File, FileError


def test_split_url_directory_preserves_leaf(tmp_path):
    uri = FileClient.path_to_uri(str(tmp_path))
    bucket, rel = FileClient.split_url(uri)

    # For directories, the parent directory becomes the bucket and the leaf
    # directory becomes the relative path to keep downstream listings stable.
    assert Path(bucket) == tmp_path.parent
    assert rel == tmp_path.name


def test_split_url_file_in_directory(tmp_path):
    file_path = tmp_path / "sub" / "file.bin"
    file_path.parent.mkdir(parents=True)
    uri = FileClient.path_to_uri(str(file_path))

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


def test_path_to_uri_preserves_trailing_slash(tmp_path):
    dir_path = tmp_path / "trail"
    dir_path.mkdir()

    uri = FileClient.path_to_uri(f"{dir_path}{os.sep}")

    base_uri = FileClient.path_to_uri(str(dir_path))

    # Trailing separator in the input should keep a trailing slash in the URI.
    assert uri.endswith("/")
    assert uri[:-1] == base_uri


def test_path_to_uri_idempotent_for_file_uri(tmp_path):
    uri = FileClient.path_to_uri(str(tmp_path))

    assert FileClient.path_to_uri(uri) == uri


def test_path_to_uri_does_not_percent_encode_spaces_and_hash(tmp_path):
    base = tmp_path / "dir #% percent"
    base.mkdir(parents=True)

    uri = FileClient.path_to_uri(str(base))
    assert uri.startswith("file://")
    assert " " in uri
    assert "#" in uri
    assert "%" in uri


def test_split_url_does_not_decode_percent_escapes(tmp_path):
    # If the filename literally contains percent-escapes, split_url must not
    # decode them (e.g. %2f -> '/').
    file_path = tmp_path / "file%2fescape%23hash.txt"
    file_path.write_text("x", encoding="utf-8")

    uri = FileClient.path_to_uri(str(file_path))
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
        ("/dir/../file.txt", None, "must not be absolute"),
        ("..", None, "must not contain"),
        ("../file.txt", None, "must not contain"),
        ("dir/../../file.txt", None, "must not contain"),
    ],
)
def test_rel_path_for_file_normalizes_and_validates(path, expected, raises):
    file = File(path=path, source="file:///tmp")
    if raises:
        with pytest.raises(FileError, match=raises):
            FileClient.rel_path_for_file(file)
    else:
        assert FileClient.rel_path_for_file(file) == expected


def test_is_path_in_rejects_dotdot(tmp_path):
    output = tmp_path / "out"
    output.mkdir()

    dst = (output / ".." / "escape.txt").as_posix()
    assert not FileClient.is_path_in(output, dst)


def test_is_path_in_rejects_directory_destination(tmp_path):
    output = tmp_path / "out"
    output.mkdir()

    assert not FileClient.is_path_in(output, output.as_posix())


def test_is_path_in_rejects_escaping_destination(tmp_path):
    output = tmp_path / "out"
    output.mkdir()

    # Absolute path outside output, without '..' segments.
    dst = (tmp_path / "escape.txt").as_posix()
    assert not FileClient.is_path_in(output, dst)


def test_is_path_in_accepts_contained_destination(tmp_path):
    output = tmp_path / "out"
    output.mkdir()

    dst = (output / "sub" / "file.txt").as_posix()
    assert FileClient.is_path_in(output, dst)


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


def test_full_path_for_file_with_drive_letter_uri():
    file = File(source="file:///C:/data", path="sub/file.txt")
    result = FileClient.full_path_for_file(file)

    # On Unix _strip_protocol keeps the leading '/', on Windows it's removed.
    if os.name == "nt":
        assert result.replace("\\", "/") == "C:/data/sub/file.txt"
    else:
        assert result == "/C:/data/sub/file.txt"


def test_full_path_for_file_drive_letter_root():
    file = File(source="file:///D:/", path="readme.md")
    result = FileClient.full_path_for_file(file)
    if os.name == "nt":
        assert result.replace("\\", "/") == "D:/readme.md"
    else:
        assert result == "/D:/readme.md"


def test_full_path_for_file_nested_drive_letter():
    file = File(source="file:///C:/Users/me/projects", path="repo/data.csv")
    result = FileClient.full_path_for_file(file)
    if os.name == "nt":
        assert result.replace("\\", "/") == "C:/Users/me/projects/repo/data.csv"
    else:
        assert result == "/C:/Users/me/projects/repo/data.csv"
