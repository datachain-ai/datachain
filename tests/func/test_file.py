import io
import tarfile
from pathlib import Path

import pytest

import datachain as dc
from datachain.lib.file import File, FileError
from datachain.lib.tar import process_tar
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
    file = File(path="dir/file", source="s3://bucket")
    file._set_stream(catalog=cloud_test_catalog.catalog)
    assert file.get_fs_path().strip("/") == "s3://bucket/dir/file"


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
    assert f.source == client.storage_uri(source)
    assert f.read() == img_bytes

    client.fs.rm(dest, recursive=True)


def test_tar_members_inherit_uri_encoded_local_source(tmp_dir, test_session_tmpfile):
    base = tmp_dir / "dir #% percent"
    base.mkdir(parents=True)

    tar_path = base / "archive.tar"
    data = b"payload"
    with tarfile.open(tar_path, mode="w") as tf:
        info = tarfile.TarInfo("member.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    chain = dc.read_storage(base, session=test_session_tmpfile)
    tar_file = chain.filter(C("file.path") == "archive.tar").to_values("file")[0]
    members = chain.filter(C("file.path") == "archive.tar").gen(file=process_tar)
    member = members.to_values("file")[0]

    assert tar_file.source.startswith("file://")
    assert " " in tar_file.source
    assert "#" in tar_file.source
    assert "%" in tar_file.source
    assert member.source == tar_file.source
    assert member.path == "archive.tar/member.txt"
    assert member.name == "member.txt"


@pytest.mark.parametrize(
    "dirname",
    [
        "dir%percent",
        "dir #% combo",
        "v1.0-release",
        "user@host",
        "100%done",
    ],
    ids=lambda d: d.replace(" ", "_"),
)
def test_read_storage_special_chars_in_local_path(
    dirname, tmp_dir, test_session_tmpfile
):
    """Regression: special chars (%, #, ., @, space) in a local directory name
    must not break listing dataset creation.  The original bug was that raw '%'
    flowed into the SQL table name and was misinterpreted by pysqlite."""
    base = tmp_dir / dirname
    base.mkdir(parents=True)
    (base / "data.txt").write_text("hello")

    files = dc.read_storage(base, session=test_session_tmpfile).to_values("file")
    assert len(files) == 1
    assert files[0].name == "data.txt"
    assert files[0].source.startswith("file://")


def test_to_storage_tar_member_filepath_keeps_percent_encoded_traversal_literal(
    tmp_dir, test_session_tmpfile
):
    payload = "..%2f..%2fescaped.txt"  # unquote -> ../../escaped.txt
    tar_path = tmp_dir / "malicious.tar"

    data = b"poc"
    with tarfile.open(tar_path, mode="w") as tf:
        info = tarfile.TarInfo(payload)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    chain = dc.read_storage(tmp_dir, session=test_session_tmpfile)
    tar_members = chain.filter(C("file.path").glob("*.tar")).gen(file=process_tar)

    member_path = f"malicious.tar/{payload}"
    member = tar_members.filter(C("file.path") == member_path)

    # Ensure the test actually selects only the malicious extracted member.
    assert member.to_values("file.path") == [member_path]

    output = tmp_dir / "output"
    assert not (tmp_dir / "escaped.txt").exists()
    member.to_storage(output, placement="filepath")

    exported_member = output / "malicious.tar" / payload

    # Expected-safe behavior: exporting the selected tar member creates a
    # directory for the parent archive path and writes the member payload under
    # its literal encoded name, without decoding %2f into path traversal.
    assert (output / "malicious.tar").is_dir()
    assert exported_member.read_bytes() == data
    assert not (tmp_dir / "escaped.txt").exists()


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


def test_read_storage_preserves_relative_file_at_source_and_path(
    tmp_dir, test_session_tmpfile, monkeypatch
):
    monkeypatch.chdir(tmp_dir)

    file_obj = File.at("rel.bin", session=test_session_tmpfile)
    with file_obj.open("wb") as f:
        f.write(b"hi")

    listed = dc.read_storage(tmp_dir, session=test_session_tmpfile).to_values("file")
    assert len(listed) == 1
    listed_file = listed[0]
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
    GCS (gcsfs>=2026.2.0): generation on handle survives close() → f1 gets V1
    Azure: No version captured → get_file_info returns V2 for f1 (race!)
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

    if cloud_type in ("s3", "gs"):
        # S3 captures version_id from handle - survives close()
        # GCS (gcsfs>=2026.2.0) now captures generation correctly
        assert file1.version != file2.version, (
            f"{cloud_type}: each write captures its own version"
        )
    else:
        # Azure: No version on handle - get_file_info returns latest
        # Update it when it is fixed in that backend.
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


def test_file_save_roundtrip(cloud_test_catalog_upload, version_aware):
    ctc = cloud_test_catalog_upload
    data = b"round-trip via File.save()"

    src = File.upload(data, f"{ctc.src_uri}/save-src.bin", ctc.catalog)
    dest = src.save(f"{ctc.src_uri}/save-dest.bin", client_config=ctc.client_config)

    assert dest.read() == data
    if version_aware:
        assert dest.version, "versioned backend must capture a version on save"


# --- write-metadata (content type / disposition / custom metadata) ---

CT = "application/pdf"
CD = 'attachment; filename="report.pdf"'
CC = "max-age=3600"
META = {"origin": "datachain-test"}
WRITE_CLOUD_TYPES = ["s3", "gs", "azure"]


def _read_object_meta(client, cloud_type, full_path):
    fs = client.fs
    if cloud_type == "s3":
        bucket, key, _ = fs.split_path(full_path)
        head = fs.call_s3("head_object", Bucket=bucket, Key=key)
        return {
            "content_type": head.get("ContentType"),
            "content_disposition": head.get("ContentDisposition"),
            "cache_control": head.get("CacheControl"),
            "metadata": head.get("Metadata") or {},
        }
    if cloud_type == "gs":
        info = fs.info(full_path, refresh=True)
        return {
            "content_type": info.get("contentType"),
            "content_disposition": info.get("contentDisposition"),
            "cache_control": info.get("cacheControl"),
            "metadata": info.get("metadata") or {},
        }
    info = fs.info(full_path)  # azure
    cs = info.get("content_settings") or {}
    md = {k: v for k, v in (info.get("metadata") or {}).items() if k != "is_directory"}
    return {
        "content_type": cs.get("content_type"),
        "content_disposition": cs.get("content_disposition"),
        "cache_control": cs.get("cache_control"),
        "metadata": md,
    }


def _cd_readback(cloud_type, streamed):
    # fake-gcs-server drops fixed metadata (content-disposition) on resumable
    # (streaming) uploads; real GCS persists it, as do S3/Azure on both paths.
    return not (cloud_type == "gs" and streamed)


def _metadata_readback(cloud_type):
    # moto does not echo user metadata via head_object.
    return cloud_type != "s3"


def _cache_control_readback(cloud_type):
    # fake-gcs-server does not persist cacheControl (real GCS does).
    return cloud_type != "gs"


@pytest.mark.parametrize("cloud_type", WRITE_CLOUD_TYPES, indirect=True)
def test_upload_sets_object_metadata(cloud_test_catalog_upload, cloud_type):
    ctc = cloud_test_catalog_upload
    catalog = ctc.catalog
    path = f"{ctc.src_uri}/wm/upload.bin"

    f = File.upload(
        b"data",
        path,
        catalog,
        content_type=CT,
        content_disposition=CD,
        cache_control=CC,
        metadata=META,
    )

    client = catalog.get_client(ctc.src_uri)
    meta = _read_object_meta(client, cloud_type, client.get_uri(f.path))
    assert meta["content_type"] == CT
    assert meta["content_disposition"] == CD
    if _cache_control_readback(cloud_type):
        assert meta["cache_control"] == CC
    if _metadata_readback(cloud_type):
        assert meta["metadata"] == META
    assert f.read() == b"data"


@pytest.mark.parametrize("cloud_type", WRITE_CLOUD_TYPES, indirect=True)
def test_save_sets_object_metadata(cloud_test_catalog_upload, cloud_type):
    ctc = cloud_test_catalog_upload
    catalog = ctc.catalog
    src = File.upload(b"payload", f"{ctc.src_uri}/wm/save-src.bin", catalog)

    dest = src.save(
        f"{ctc.src_uri}/wm/save-dest.bin",
        client_config=ctc.client_config,
        content_type=CT,
        content_disposition=CD,
        metadata=META,
    )

    client = catalog.get_client(ctc.src_uri)
    meta = _read_object_meta(client, cloud_type, client.get_uri(dest.path))
    assert meta["content_type"] == CT
    if _cd_readback(cloud_type, streamed=True):
        assert meta["content_disposition"] == CD
    if _metadata_readback(cloud_type):
        assert meta["metadata"] == META
    assert dest.read() == b"payload"


@pytest.mark.parametrize("cloud_type", WRITE_CLOUD_TYPES, indirect=True)
def test_open_write_sets_object_metadata(cloud_test_catalog_upload, cloud_type):
    ctc = cloud_test_catalog_upload
    file = File.at(f"{ctc.src_uri}/wm/open.bin", ctc.session)
    with file.open("wb", content_type=CT, content_disposition=CD, metadata=META) as f:
        f.write(b"streamed")

    client = ctc.catalog.get_client(ctc.src_uri)
    meta = _read_object_meta(client, cloud_type, client.get_uri(file.path))
    assert meta["content_type"] == CT
    if _cd_readback(cloud_type, streamed=True):
        assert meta["content_disposition"] == CD
    if _metadata_readback(cloud_type):
        assert meta["metadata"] == META
    assert file.read() == b"streamed"


@pytest.mark.parametrize("cloud_type", WRITE_CLOUD_TYPES, indirect=True)
def test_export_sets_object_metadata(cloud_test_catalog_upload, cloud_type):
    ctc = cloud_test_catalog_upload
    catalog = ctc.catalog
    src = File.upload(b"exp", f"{ctc.src_uri}/wm/exp-src.bin", catalog)

    src.export(
        f"{ctc.src_uri}/wm/exported",
        placement="filename",
        client_config=ctc.client_config,
        content_type=CT,
        content_disposition=CD,
    )

    client = catalog.get_client(ctc.src_uri)
    full = client.get_uri("wm/exported/exp-src.bin")
    meta = _read_object_meta(client, cloud_type, full)
    assert meta["content_type"] == CT
    if _cd_readback(cloud_type, streamed=True):
        assert meta["content_disposition"] == CD


@pytest.mark.parametrize("cloud_type", WRITE_CLOUD_TYPES, indirect=True)
def test_text_file_save_sets_object_metadata(cloud_test_catalog_upload, cloud_type):
    from datachain.lib.file import TextFile

    ctc = cloud_test_catalog_upload
    catalog = ctc.catalog
    src = File.upload(b"hello text", f"{ctc.src_uri}/wm/text-src.txt", catalog)
    tf = TextFile(**src.model_dump())
    tf._set_stream(catalog)

    dest = tf.save(
        f"{ctc.src_uri}/wm/text-dest.txt",
        client_config=ctc.client_config,
        content_type="text/plain",
        content_disposition=CD,
    )

    client = catalog.get_client(ctc.src_uri)
    meta = _read_object_meta(client, cloud_type, client.get_uri(dest.path))
    assert meta["content_type"] == "text/plain"
    if _cd_readback(cloud_type, streamed=True):
        assert meta["content_disposition"] == CD


@pytest.mark.parametrize("cloud_type", WRITE_CLOUD_TYPES, indirect=True)
def test_image_file_save_sets_object_metadata(cloud_test_catalog_upload, cloud_type):
    from io import BytesIO

    from PIL import Image as PilImage

    from datachain.lib.file import ImageFile

    ctc = cloud_test_catalog_upload
    catalog = ctc.catalog

    buf = BytesIO()
    PilImage.new("RGB", (4, 4), color="red").save(buf, format="PNG")
    src = File.upload(buf.getvalue(), f"{ctc.src_uri}/wm/img-src.png", catalog)
    img = ImageFile(**src.model_dump())
    img._set_stream(catalog)

    dest = img.save(
        f"{ctc.src_uri}/wm/img-dest.png",
        client_config=ctc.client_config,
        content_type="image/png",
        content_disposition=CD,
        metadata=META,
    )

    client = catalog.get_client(ctc.src_uri)
    meta = _read_object_meta(client, cloud_type, client.get_uri(dest.path))
    assert meta["content_type"] == "image/png"
    # ImageFile.save uploads bytes (pipe path), so gcs disposition persists too.
    assert meta["content_disposition"] == CD
    if _metadata_readback(cloud_type):
        assert meta["metadata"] == META


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
def test_upload_write_options_applied_on_s3(cloud_test_catalog_upload, cloud_type):
    ctc = cloud_test_catalog_upload
    catalog = ctc.catalog

    f = File.upload(
        b"raw",
        f"{ctc.src_uri}/wm/wo.bin",
        catalog,
        write_options={"ContentDisposition": CD},
    )

    client = catalog.get_client(ctc.src_uri)
    meta = _read_object_meta(client, cloud_type, client.get_uri(f.path))
    assert meta["content_disposition"] == CD


@pytest.mark.parametrize("cloud_type", ["gs", "azure"], indirect=True)
def test_write_options_rejected_on_gcs_azure(cloud_test_catalog_upload, cloud_type):
    ctc = cloud_test_catalog_upload
    with pytest.raises(NotImplementedError, match="write_options"):
        File.upload(
            b"x",
            f"{ctc.src_uri}/wm/wo-reject.bin",
            ctc.catalog,
            write_options={"foo": "bar"},
        )


@pytest.mark.parametrize("cloud_type", ["azure"], indirect=True)
def test_open_append_azure(cloud_test_catalog_upload):
    ctc = cloud_test_catalog_upload
    path = f"{ctc.src_uri}/wm/append.txt"
    file = File.at(path, ctc.session)
    with file.open("ab") as f:
        f.write(b"AB")
    with file.open("ab") as f:
        f.write(b"CD")
    assert File.at(path, ctc.session).read() == b"ABCD"


@pytest.mark.parametrize("cloud_type", ["azure"], indirect=True)
def test_open_append_azure_rejects_write_metadata(cloud_test_catalog_upload):
    ctc = cloud_test_catalog_upload
    file = File.at(f"{ctc.src_uri}/wm/append-meta.txt", ctc.session)
    with pytest.raises(NotImplementedError, match="append"):
        with file.open("ab", content_type="text/plain"):
            pass
