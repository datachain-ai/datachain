import os
from pathlib import Path

from datachain.client.local import FileClient


def test_split_url_directory_preserves_leaf(tmp_path):
    uri = tmp_path.as_uri()
    bucket, rel = FileClient.split_url(uri)

    # For directories, the parent directory becomes the bucket and the leaf
    # directory becomes the relative path to keep downstream listings stable.
    assert Path(bucket) == tmp_path.parent
    assert rel == tmp_path.name


def test_split_url_file_in_directory(tmp_path):
    file_path = tmp_path / "sub" / "file.bin"
    file_path.parent.mkdir(parents=True)
    uri = file_path.as_uri()

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

    # Trailing separator in the input should keep a trailing slash in the URI.
    assert uri.endswith("/")
    assert uri[:-1] == dir_path.resolve().as_uri()


def test_path_to_uri_idempotent_for_file_uri(tmp_path):
    uri = tmp_path.as_uri()

    assert FileClient.path_to_uri(uri) == uri
