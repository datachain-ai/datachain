import io

import pytest

import datachain as dc
from datachain.lib.file import File, FileError
from datachain.query import C
from datachain.utils import TIME_ZERO


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
def test_get_path_cloud(cloud_test_catalog):
    file = File(path="dir/file", source="s3://")
    file._set_stream(catalog=cloud_test_catalog.catalog)
    assert file.get_fs_path().strip("/") == "s3:///dir/file"


@pytest.mark.parametrize("caching_enabled", [True, False])
def test_resolve_file(cloud_test_catalog, caching_enabled):
    ctc = cloud_test_catalog

    chain = dc.read_storage(ctc.src_uri, session=ctc.session)
    for orig_file in chain.to_values("file"):
        file = File(
            source=orig_file.source,
            path=orig_file.path,
        )
        file._set_stream(catalog=ctc.catalog, caching_enabled=caching_enabled)
        resolved_file = file.resolve()
        assert orig_file == resolved_file

        file.ensure_cached()


def test_resolve_file_no_exist(cloud_test_catalog):
    ctc = cloud_test_catalog

    non_existent_file = File(source=ctc.src_uri, path="non_existent_file.txt")
    non_existent_file._set_stream(catalog=ctc.catalog)
    resolved_non_existent = non_existent_file.resolve()
    assert resolved_non_existent.size == 0
    assert resolved_non_existent.etag == ""
    assert resolved_non_existent.last_modified == TIME_ZERO


@pytest.mark.parametrize("path", ["", ".", "..", "/", "dir/../../file.txt"])
def test_resolve_file_wrong_path(cloud_test_catalog, path):
    ctc = cloud_test_catalog

    wrong_file = File(source=ctc.src_uri, path=path)
    wrong_file._set_stream(catalog=ctc.catalog)
    resolved_wrong = wrong_file.resolve()
    assert resolved_wrong.size == 0
    assert resolved_wrong.etag == ""
    assert resolved_wrong.last_modified == TIME_ZERO


@pytest.mark.parametrize("caching_enabled", [True, False])
@pytest.mark.parametrize("path", ["", ".", "..", "/", "dir/../../file.txt"])
def test_cache_file_wrong_path(cloud_test_catalog, path, caching_enabled):
    ctc = cloud_test_catalog

    wrong_file = File(source=ctc.src_uri, path=path)
    wrong_file._set_stream(catalog=ctc.catalog, caching_enabled=caching_enabled)
    with pytest.raises(FileError):
        wrong_file.ensure_cached()


def test_upload(cloud_test_catalog):
    ctc = cloud_test_catalog

    src_uri = ctc.src_uri
    filename = "image_1.jpg"
    dest = f"{src_uri}/upload-test-images"
    catalog = ctc.catalog

    img_bytes = b"bytes"

    f = File.upload(img_bytes, f"{dest}/{filename}", catalog)

    client = catalog.get_client(src_uri)
    source, rel_path = client.split_url(f"{dest}/{filename}")

    assert f.path == rel_path
    assert f.source == client.get_uri(source)
    assert f.read() == img_bytes

    client.fs.rm(dest, recursive=True)


def test_open_write_binary(cloud_test_catalog):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    src_uri = ctc.src_uri
    data = b"hello via open()"
    file_path = f"{src_uri}/test-open-write-bytes.bin"

    file = File.at(file_path, ctc.session)
    with file.open("wb") as f:
        f.write(data)

    assert file.size == len(data)
    assert file.read() == data

    # Query storage for exactly that relative path.
    # Metadata already refreshed by open() write path.
    rel_path = file.path
    chain = dc.read_storage(src_uri, session=ctc.session).filter(
        C("file.path") == rel_path
    )
    results = list(chain.to_values("file"))
    assert len(results) == 1
    match = results[0]
    for field_name in File.model_fields:
        if field_name == "last_modified":
            # Allow up to 1s difference across backends
            # (some backends don't keep microsecond precision, we keep it simple here)
            assert match.last_modified.timestamp() == pytest.approx(
                file.last_modified.timestamp(), abs=1
            )
        else:
            assert getattr(match, field_name) == getattr(file, field_name), (
                f"Mismatch in field '{field_name}'"
            )

    catalog.get_client(src_uri).fs.rm(file_path)


def test_open_write_text(cloud_test_catalog):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    src_uri = ctc.src_uri
    file_path = f"{src_uri}/test-open-write-text.txt"
    # Unicode content to exercise non-default (utf-16) encoding round trip
    content = "Привет Мир\nSecond line"

    file = File.at(file_path, ctc.session)
    with file.open("w", encoding="utf-16-le") as f:
        written_chars = f.write(content)

    assert written_chars == len(content)
    assert file.read_text(encoding="utf-16-le") == content

    # Compute expected byte size using identical TextIOWrapper logic
    buf = io.BytesIO()
    tw = io.TextIOWrapper(buf, encoding="utf-16-le")
    tw.write(content)
    tw.flush()
    expected_size = len(buf.getvalue())
    tw.close()
    assert file.size == expected_size

    catalog.get_client(src_uri).fs.rm(file_path)
