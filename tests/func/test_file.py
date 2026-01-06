import io
from pathlib import Path

import pytest

import datachain as dc
from datachain.lib.file import File, FileError
from datachain.query import C
from datachain.utils import TIME_ZERO


def test_file_pickle_with_catalog(tmp_dir, test_session_tmpfile):
    """Test that File objects with catalog can be pickled for parallel processing.

    File objects created with File.at() have a _catalog attached which contains
    SQLAlchemy connections, weakrefs, etc that cannot be pickled. The __getstate__
    method must exclude _catalog to allow pickling.
    """
    import cloudpickle

    # Create a real file using File.at + open (this attaches _catalog)
    file = File.at(tmp_dir / "test.txt", session=test_session_tmpfile)
    with file.open("w") as f:
        f.write("hello world")

    assert file._catalog is not None, "File should have catalog attached"

    # This would fail with "cannot pickle 'sqlite3.Connection' object"
    # if __getstate__ doesn't exclude _catalog
    data = cloudpickle.dumps(file)
    restored = cloudpickle.loads(data)

    # Catalog should be None after unpickling (will be re-set by worker)
    assert restored._catalog is None
    # But other attributes should be preserved
    assert restored.path == file.path
    assert restored.source == file.source


def test_file_serialized_in_udf(tmp_dir, test_session_tmpfile):
    # A captured File created via File.at() (has _catalog) to force pickling in closure
    captured = File.at(tmp_dir / "captured.txt", session=test_session_tmpfile)
    with captured.open("w") as f:
        f.write("captured")

    chain = (
        dc.read_values(id=range(100), session=test_session_tmpfile)
        .settings(parallel=True)
        .map(path=lambda id: captured.path)
        .persist()
    )

    results = sorted(chain.to_values("path"))
    assert len(results) == 100


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


def test_file_at_accepts_pathlike(tmp_dir, test_session_tmpfile, monkeypatch):
    monkeypatch.chdir(tmp_dir)

    as_str = File.at("rel.bin", session=test_session_tmpfile)
    as_path = File.at(Path("rel.bin"), session=test_session_tmpfile)

    assert as_str.path == as_path.path == "rel.bin"
    expected_source = tmp_dir.as_uri()

    assert as_str.source == as_path.source == expected_source


def test_file_at_rejects_directory_uri(tmp_dir, test_session_tmpfile, monkeypatch):
    monkeypatch.chdir(tmp_dir)

    dir_uri = f"{tmp_dir}/"

    with pytest.raises(ValueError):
        File.at(dir_uri, session=test_session_tmpfile)


def test_read_storage_returns_same_source_and_path(
    tmp_dir, test_session_tmpfile, monkeypatch
):
    monkeypatch.chdir(tmp_dir)
    import os

    print(os.getcwd())

    file_obj = File.at("rel.bin", session=test_session_tmpfile)
    with file_obj.open("wb") as f:
        f.write(b"hi")

    import os

    print(os.getcwd())

    listed = dc.read_storage(tmp_dir, session=test_session_tmpfile).to_values("file")
    assert len(listed) == 1
    listed_file = listed[0]
    print(listed_file.path)
    print(listed_file.source)
    # print pwd
    import os

    print(os.getcwd())
    assert listed_file.path == "rel.bin"
    assert listed_file.source == file_obj.source

    assert (tmp_dir / "rel.bin").read_bytes() == b"hi"
    assert file_obj.size == len(b"hi")
    assert file_obj.read() == b"hi"


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("version_aware", [True], indirect=True)
def test_write_version_capture(cloud_test_catalog, cloud_type):
    """Test version capture when writes interleave.

    By manually closing the fsspec handle inside File.open(), we commit
    to storage. The fsspec close() is idempotent so the context __exit__
    won't fail. The sequence becomes:
    1. f1.write() + f1.close() → commits V1
    2. f2.write() + f2.close() → commits V2
    3. f2's File.open() __exit__ → extract_version(f2), get_file_info
    4. f1's File.open() __exit__ → extract_version(f1), get_file_info

    S3: version_id on handle survives close() → f1 gets V1
    GCS/Azure: No version captured → get_file_info returns V2 for f1 (race!)
    """
    ctc = cloud_test_catalog
    src_uri = ctc.src_uri
    file_path = f"{src_uri}/test-write-version.bin"
    client = ctc.catalog.get_client(src_uri)

    file1 = File.at(file_path, ctc.session)
    file2 = File.at(file_path, ctc.session)

    with file1.open("wb") as f1:
        f1.write(b"content version 1")
        f1.close()  # Commit V1 to storage (close is idempotent)

        with file2.open("wb") as f2:
            f2.write(b"content version 2")
            f2.close()  # Commit V2 to storage
        # f2's File.open() __exit__ runs here
    # f1's File.open() __exit__ runs here (after V2 exists!)

    assert file1.version, "file1 should have a version"
    assert file2.version, "file2 should have a version"

    if cloud_type == "s3":
        # S3 captures version_id from handle - survives close()
        assert file1.version != file2.version, (
            "S3: each write captures its own version_id"
        )
    else:
        # GCS/Azure: No version on handle - get_file_info returns latest
        # Update it when it is fixed in those backends.
        assert file1.version == file2.version, (
            f"{cloud_type}: file1 sees V2 due to get_file_info race"
        )

    client.fs.rm(file_path)


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [True], indirect=True)
def test_empty_upload_captures_version(cloud_test_catalog_upload, cloud_type):
    ctc = cloud_test_catalog_upload
    catalog = ctc.catalog
    src_uri = ctc.src_uri

    upload_path = f"{src_uri}/empty-upload.bin"
    uploaded = File.upload(b"", upload_path, catalog)

    assert uploaded.size == 0
    assert uploaded.version
    assert uploaded.read() == b""


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [True], indirect=True)
def test_empty_open_write_captures_version(cloud_test_catalog_upload, cloud_type):
    ctc = cloud_test_catalog_upload
    src_uri = ctc.src_uri

    open_path = f"{src_uri}/empty-open.bin"
    opened = File.at(open_path, ctc.session)
    with opened.open("wb"):
        pass

    assert opened.size == 0
    assert opened.version
    assert opened.read() == b""
