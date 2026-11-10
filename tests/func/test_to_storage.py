import os
import posixpath
import re
import signal
import subprocess
import sys
import textwrap
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import cast
from urllib.parse import urlparse

import pytest
from PIL import Image

import datachain as dc
from datachain import DataModel, Session
from datachain.lib.file import (
    AudioFile,
    AudioFragment,
    File,
    FileError,
    ImageFile,
    TextFile,
    VideoFile,
    VideoFragment,
)
from datachain.lib.tar import process_tar
from datachain.lib.utils import DataChainParamsError
from tests.conftest import get_cloud_test_catalog, make_cloud_server
from tests.utils import images_equal, make_tar, skip_if_not_sqlite, tree_from_path

python_exc = sys.executable or "python3"


@pytest.mark.parametrize("same", [False, True])
def test_file_save_checks_full_destination_path(
    test_session, tmp_path, monkeypatch, same
):
    source = tmp_path / "source"
    other = tmp_path / "other"
    source.mkdir()
    other.mkdir()
    (source / "a.bin").write_bytes(b"source bytes")
    (other / "a.bin").write_bytes(b"old bytes")
    file = File.at(source / "a.bin", session=test_session)
    monkeypatch.chdir(other if same else source)
    destination = (source if same else other) / "a.bin"
    saved = file.save(str(destination))
    assert destination.read_bytes() == b"source bytes"
    assert (source / "a.bin").read_bytes() == b"source bytes"
    assert saved.read() == b"source bytes"


@pytest.mark.parametrize("file_cls", [File, TextFile, ImageFile, VideoFile, AudioFile])
def test_to_storage_copies_bytes_as_stored(test_session, tmp_path, file_cls):
    (tmp_path / "a.jpg").write_bytes(b"not decoded")
    dc.read_values(
        file=[file_cls(path="a.jpg", source=tmp_path.as_uri())], session=test_session
    ).to_storage(tmp_path / "out", placement="filename")
    assert (tmp_path / "out" / "a.jpg").read_bytes() == b"not decoded"


@pytest.mark.parametrize("link_type", ["copy", "symlink"])
def test_to_storage_relative_source_preserves_bytes(
    test_session, tmp_path, monkeypatch, link_type
):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "a.jpg"
    source.write_bytes(b"original bytes")
    dc.read_values(file=[ImageFile(path=source.name)], session=test_session).settings(
        cache=False, prefetch=0
    ).to_storage(tmp_path, placement="filename", link_type=link_type)
    assert source.read_bytes() == b"original bytes"


@pytest.mark.parametrize("alias", ["same", "hardlink", "symlink"])
@pytest.mark.parametrize("cached_archive", [False, True])
def test_virtual_image_export_preserves_backing_archive(
    test_session, tmp_path, alias, cached_archive
):
    image = BytesIO()
    Image.new("RGB", (1, 1)).save(image, format="JPEG")
    archive = tmp_path / "archive.tar"
    original = make_tar({archive.name: image.getvalue()})
    archive.write_bytes(original)
    archive_file = File.at(archive, session=test_session)
    if cached_archive:
        archive_file.ensure_cached()
    member = next(process_tar(archive_file))
    output = tmp_path if alias == "same" else tmp_path / "out"
    if alias != "same":
        output.mkdir()
        if alias == "symlink":
            if os.name == "nt":
                pytest.skip("Creating symlinks needs privileges on Windows")
            (output / archive.name).symlink_to(archive)
        else:
            os.link(archive, output / archive.name)
    with pytest.raises(FileError, match="backing file"):
        dc.read_values(
            file=[ImageFile(**member.model_dump())], session=test_session
        ).settings(cache=False, prefetch=0).to_storage(output, placement="filename")
    assert archive.read_bytes() == original
    assert (output / archive.name).read_bytes() == original


@pytest.mark.parametrize("file_cls", [File, ImageFile])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("alias", ["same", "hardlink", "symlink"])
@pytest.mark.parametrize("link_type", ["copy", "symlink"])
def test_to_storage_over_source_preserves_bytes(
    test_session, tmp_path, file_cls, cache, alias, link_type
):
    source = tmp_path / "a.jpg"
    source.write_bytes(b"original bytes")
    output = tmp_path if alias == "same" else tmp_path / "out"
    if alias != "same":
        output.mkdir()
        if alias == "symlink":
            if os.name == "nt":
                pytest.skip("Creating symlinks needs privileges on Windows")
            (output / source.name).symlink_to(source)
        else:
            os.link(source, output / source.name)
    dc.read_values(
        file=[file_cls(path=source.name, source=tmp_path.as_uri())],
        session=test_session,
    ).settings(cache=cache).to_storage(
        output, placement="filename", link_type=link_type
    )
    assert source.read_bytes() == b"original bytes"
    assert (output / source.name).read_bytes() == b"original bytes"


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("alias", ["same", "hardlink"])
@pytest.mark.parametrize("operation", ["save", "copy", "symlink"])
def test_export_over_source_uses_the_saved_file_contents(
    test_session, tmp_path, cache, alias, operation
):
    source = tmp_path / "input.bin"
    source.write_bytes(b"old")
    (file,) = dc.read_storage(source.as_uri(), session=test_session).to_values("file")
    if cache:
        file.ensure_cached()
    output = tmp_path if alias == "same" else tmp_path / "out"
    output.mkdir(exist_ok=True)
    destination = output / source.name
    if alias == "hardlink":
        os.link(source, destination)
    mtime = source.stat().st_mtime + 1
    source.write_bytes(b"new")
    os.utime(source, (mtime, mtime))

    def export():
        if operation == "save":
            file.save(str(destination))
        else:
            file.export(
                output, placement="filename", use_cache=cache, link_type=operation
            )

    if cache:
        export()
        assert destination.read_bytes() == b"old"
    else:
        with pytest.raises(FileNotFoundError, match="changed on the source"):
            export()
        assert destination.read_bytes() == b"new"


@pytest.mark.parametrize("cache", [False, True])
def test_remote_export_links_cached_bytes_or_falls_back_to_copy(
    test_session, http_server, tmp_path, cache
):
    if cache and os.name == "nt":
        pytest.skip("Creating symlinks needs privileges on Windows")
    (tmp_path / "source.bin").write_bytes(b"original")
    file = File.at(f"{http_server}/source.bin", session=test_session)
    output = tmp_path / "out"

    file.export(output, placement="filename", use_cache=cache, link_type="symlink")

    assert (output / "source.bin").read_bytes() == b"original"
    assert (output / "source.bin").is_symlink() == cache


def test_to_storage_reexports_cached_file_after_source_is_deleted(
    test_session, tmp_path
):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "a.bin"
    path.write_bytes(b"original bytes")
    chain = (
        dc.read_storage(source.as_uri(), session=test_session)
        .settings(cache=True)
        .persist()
    )
    output = tmp_path / "out"
    chain.to_storage(output, placement="filename")
    path.unlink()

    chain.to_storage(output, placement="filename")

    assert (output / "a.bin").read_bytes() == b"original bytes"


@pytest.mark.parametrize("placement", ["fullpath", "filename", "filepath"])
@pytest.mark.parametrize("use_map", [True, False])
@pytest.mark.parametrize("use_cache", [True, False])
@pytest.mark.parametrize("file_type", ["", "binary", "text"])
@pytest.mark.parametrize("num_threads", [0, 2])
@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
def test_to_storage(
    tmp_dir,
    cloud_test_catalog,
    test_session,
    placement,
    use_map,
    use_cache,
    file_type,
    num_threads,
):
    call_count = 0

    def mapper(file_path):
        nonlocal call_count
        call_count += 1
        return len(file_path)

    ctc = cloud_test_catalog
    df = dc.read_storage(ctc.src_uri, type=file_type, session=test_session)
    if use_map:
        (
            df.settings(cache=use_cache)
            .map(mapper, params=["file.path"], output={"path_len": int})
            .map(res=lambda file: file.export(tmp_dir / "output", placement=placement))
            .exec()
        )
    else:
        (
            df.settings(cache=use_cache)
            .map(mapper, params=["file.path"], output={"path_len": int})
            .to_storage(
                tmp_dir / "output", placement=placement, num_threads=num_threads
            )
        )

    expected = {
        "description": "Cats and Dogs",
        "cat1": "meow",
        "cat2": "mrow",
        "dog1": "woof",
        "dog2": "arf",
        "dog3": "bark",
        "dog4": "ruff",
    }

    def _expected_destination_rel(file_obj: File, placement: str) -> Path:
        rel_path = PurePosixPath(file_obj.path).as_posix()

        if placement == "filename":
            return Path(file_obj.name)
        if placement == "filepath":
            return Path(rel_path)
        if placement == "fullpath":
            parsed = urlparse(file_obj.source)
            full_rel = rel_path
            if parsed.scheme and parsed.scheme != "file":
                full_rel = posixpath.join(parsed.netloc, rel_path)
            return Path(full_rel)
        raise AssertionError(f"Unsupported placement: {placement}")

    output_root = tmp_dir / "output"
    for file_record in df.to_values("file"):
        file_obj = cast("File", file_record)
        destination_rel = _expected_destination_rel(file_obj, placement)

        with (output_root / destination_rel).open() as f:
            assert f.read() == expected[file_obj.name]

    assert call_count == len(expected)


@pytest.mark.parametrize("use_cache", [True, False])
def test_export_images_files(test_session, tmp_dir, tmp_path, use_cache):
    images = [
        {"name": "img1.jpg", "data": Image.new(mode="RGB", size=(64, 64))},
        {"name": "img2.jpg", "data": Image.new(mode="RGB", size=(128, 128))},
    ]

    for img in images:
        img["data"].save(tmp_path / img["name"])

    dc.read_values(
        file=[ImageFile(path=img["name"], source=tmp_path.as_uri()) for img in images],
        session=test_session,
    ).settings(cache=use_cache).to_storage(tmp_dir / "output", placement="filename")

    for img in images:
        with Image.open(tmp_dir / "output" / img["name"]) as exported_img:
            assert images_equal(img["data"], exported_img)


def test_to_storage_skips_none_optional_file(test_session, tmp_dir, tmp_path):
    """to_storage on an Optional[File] signal skips absent (None) rows instead of
    crashing with AttributeError: 'NoneType' object has no attribute 'export'."""
    (tmp_path / "a.txt").write_text("hi")
    dc.read_values(
        file=[File(path="a.txt", source=tmp_path.as_uri()), None],
        output={"file": File | None},
        session=test_session,
    ).to_storage(tmp_dir / "output", placement="filename")

    assert (tmp_dir / "output" / "a.txt").read_text() == "hi"


def test_read_storage_multiple_file_uris_and_directory(test_session, tmp_dir, tmp_path):
    images = [
        {"name": "img1.jpg", "data": Image.new(mode="RGB", size=(64, 64))},
        {"name": "img2.jpg", "data": Image.new(mode="RGB", size=(128, 128))},
    ]

    for img in images:
        img["data"].save(tmp_path / img["name"])

    dc.read_storage(
        [
            f"file://{tmp_path}/img1.jpg",
            f"file://{tmp_path}/img2.jpg",
        ],
        session=test_session,
        anon=True,
        update=True,
    ).to_storage(tmp_dir / "output", placement="filename")

    for img in images:
        with Image.open(tmp_dir / "output" / img["name"]) as exported_img:
            assert images_equal(img["data"], exported_img)

    chain = dc.read_storage(
        [
            f"file://{tmp_path}/img1.jpg",
            f"file://{tmp_path}/img2.jpg",
            f"file://{tmp_dir}/output/",
        ],
        session=test_session,
    )
    assert chain.count() == 4

    chain = dc.read_storage([f"file://{tmp_dir}/output/"], session=test_session)
    assert chain.count() == 2


def test_to_storage_from_path_object(test_session, tmp_dir, tmp_path):
    images = [
        {"name": "img1.jpg", "data": Image.new(mode="RGB", size=(64, 64))},
        {"name": "img2.jpg", "data": Image.new(mode="RGB", size=(128, 128))},
    ]

    for img in images:
        img["data"].save(tmp_path / img["name"])

    dc.read_storage(tmp_path).to_storage(tmp_dir / "output", placement="filename")

    for img in images:
        with Image.open(tmp_dir / "output" / img["name"]) as exported_img:
            assert images_equal(img["data"], exported_img)


def test_to_storage_relative_path(test_session, tmp_path, monkeypatch):
    images = [
        {"name": "img1.jpg", "data": Image.new(mode="RGB", size=(64, 64))},
        {"name": "img2.jpg", "data": Image.new(mode="RGB", size=(128, 128))},
    ]

    for img in images:
        img["data"].save(tmp_path / img["name"])

    work_dir = tmp_path / "cwd"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    dc.read_values(
        file=[
            ImageFile(path=img["name"], source=f"file://{tmp_path}") for img in images
        ],
        session=test_session,
    ).to_storage("output", placement="filename")

    for img in images:
        with Image.open(Path("output") / img["name"]) as exported_img:
            assert images_equal(img["data"], exported_img)


def test_to_storage_files_filename_placement_not_unique_files(tmp_dir, test_session):
    data = b"some\x00data\x00is\x48\x65\x6c\x57\x6f\x72\x6c\x64\xff\xffheRe"
    bucket_name = "mybucket"
    files = ["dir1/a.json", "dir1/dir2/a.json"]

    bucket_dir = tmp_dir / bucket_name
    bucket_dir.mkdir(parents=True)
    for file_path in files:
        file_path = bucket_dir / file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as fd:
            fd.write(data)

    df = dc.read_storage((tmp_dir / bucket_name).as_uri(), session=test_session)
    with pytest.raises(ValueError):
        df.to_storage(tmp_dir / "output", placement="filename")


@skip_if_not_sqlite
@pytest.mark.parametrize(
    "num_threads",
    [1, 5],
    ids=["single_thread", "multiple_threads"],
)
def test_to_storage_keyboard_interrupt_does_not_lock_cleanup(
    tmp_path, catalog_tmpfile, num_threads
):
    script = tmp_path / "test_to_storage_interrupt.py"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for idx in range(100):
        (input_dir / f"{idx}.txt").write_text(f"file-{idx}")

    script.write_text(
        textwrap.dedent(f"""
            import signal
            import threading
            import time

            import datachain as dc
            from datachain.lib.file import File

            original_export = File.export
            signaled = threading.Event()

            def patched_export(self, *args, **kwargs):
                if not signaled.is_set():
                    signaled.set()

                    def interrupt():
                        time.sleep(0.2)
                        signal.raise_signal(signal.SIGINT)

                    threading.Thread(
                        target=interrupt,
                        daemon=True,
                    ).start()
                    time.sleep(1)
                return original_export(self, *args, **kwargs)

            File.export = patched_export

            dc.read_storage({str(input_dir)!r}).to_storage(
                {str(output_dir)!r},
                placement="filename",
                num_threads={num_threads},
            )
        """),
    )

    result = subprocess.run(  # noqa: S603
        [python_exc, str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "DATACHAIN__METASTORE": catalog_tmpfile.metastore.serialize(),
            "DATACHAIN__WAREHOUSE": catalog_tmpfile.warehouse.serialize(),
        },
    )

    interrupt_exit_codes = (-signal.SIGINT, 130)
    if sys.platform == "win32":
        interrupt_exit_codes = (3221225786, 130)

    assert result.returncode in interrupt_exit_codes
    assert "KeyboardInterrupt" in result.stderr
    assert "database is locked" not in result.stderr
    assert "Error in sys.excepthook" not in result.stderr
    assert "Exception ignored in atexit callback" not in result.stderr


def test_cross_cloud_transfer(
    request,
    tmp_upath_factory,
    tree,
    tmp_path,
    metastore,
    warehouse,
):
    disabled_remotes = request.config.getoption("--disable-remotes") or []

    if any(remote in disabled_remotes for remote in ["azure", "gs", "all"]):
        pytest.skip("Skipping all tests for azure, gs or all remotes")

    azure_path = tmp_upath_factory.mktemp("azure", version_aware=False)
    azure_server = make_cloud_server(azure_path, "azure", tree)

    gcloud_path = tmp_upath_factory.mktemp("gs", version_aware=False)
    gcloud_server = make_cloud_server(gcloud_path, "gs", tree)

    azure_catalog = get_cloud_test_catalog(azure_server, tmp_path, metastore, warehouse)
    gcloud_catalog = get_cloud_test_catalog(
        gcloud_server, tmp_path, metastore, warehouse
    )

    test_filename = "image_1.jpg"
    test_content = b"bytes"

    source_dir = f"{azure_catalog.src_uri}/source-test-images"
    source_file = f"{source_dir}/{test_filename}"

    dest_dir = f"{gcloud_catalog.src_uri}/destination-test-images"
    dest_file = f"{dest_dir}/{test_filename}"

    azure_client = azure_catalog.catalog.get_client(source_file)
    gcloud_client = gcloud_catalog.catalog.get_client(dest_file)

    try:
        with azure_client.fs.open(source_file, "wb") as f:
            f.write(test_content)

        combined_config = azure_server.client_config | gcloud_server.client_config
        with Session("testSession", client_config=combined_config):
            datachain = dc.read_storage(source_dir)
            datachain.to_storage(dest_dir, placement="filename")

        with gcloud_client.fs.open(dest_file, "rb") as f:
            assert f.read() == test_content

    finally:
        try:
            azure_client.fs.rm(source_dir, recursive=True)
            gcloud_client.fs.rm(dest_dir, recursive=True)
        except FileNotFoundError:
            pass


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
def test_to_storage_skips_empty_segments_locally(cloud_test_catalog, tmp_dir, caplog):
    ctc = cloud_test_catalog
    client = ctc.catalog.get_client(ctc.src_uri)
    prefix = "empty-seg"
    keys = {"r/b.txt": b"real", "r//b.txt": b"odd", "x//alone.txt": b"alone"}
    for key, data in keys.items():
        client.fs.pipe_file(f"{client.name}/{prefix}/{key}", data)
    client.fs.invalidate_cache()
    out = tmp_dir / "out"
    try:
        with caplog.at_level("WARNING", logger="datachain"):
            dc.read_storage(f"{ctc.src_uri}/{prefix}", session=ctc.session).to_storage(
                out, placement="filepath"
            )
    finally:
        client.fs.rm(f"{client.name}/{prefix}", recursive=True)
        client.fs.invalidate_cache()

    assert tree_from_path(out) == {prefix: {"r": {"b.txt": "real"}}}
    for key in ("r//b.txt", "x//alone.txt"):
        assert f"File '{prefix}/{key}' not exported: path has '//'" in caplog.text


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
def test_to_storage_keeps_empty_segments_in_cloud(cloud_test_catalog):
    ctc = cloud_test_catalog
    client = ctc.catalog.get_client(ctc.src_uri)
    src, dst = "empty-seg-src", "empty-seg-dst"
    client.fs.pipe_file(f"{client.name}/{src}/x//a.txt", b"a")
    client.fs.invalidate_cache()
    try:
        dc.read_storage(f"{ctc.src_uri}/{src}", session=ctc.session).to_storage(
            f"{ctc.src_uri}/{dst}", placement="filepath"
        )
        client.fs.invalidate_cache()
        assert client.fs.cat_file(f"{client.name}/{dst}/{src}/x//a.txt") == b"a"
    finally:
        client.fs.rm(f"{client.name}/{src}", recursive=True)
        client.fs.rm(f"{client.name}/{dst}", recursive=True)
        client.fs.invalidate_cache()


@pytest.mark.parametrize("placement", ["filename", "etag"])
@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
def test_to_storage_empty_segments_with_flat_placement(cloud_test_catalog, placement):
    ctc = cloud_test_catalog
    client = ctc.catalog.get_client(ctc.src_uri)
    key = "flat-export/d//e.txt"
    client.fs.pipe_file(f"{client.name}/{key}", b"content")
    client.fs.invalidate_cache()
    out = ctc.working_dir / "out"
    try:
        dc.read_storage(f"{ctc.src_uri}/flat-export/", session=ctc.session).to_storage(
            out, placement=placement
        )
        assert [p.read_bytes() for p in out.iterdir()] == [b"content"]
    finally:
        client.fs.rm(f"{client.name}/{key}")
        client.fs.invalidate_cache()


class FilePair(DataModel):
    file: File
    n: int


class MediaHolder(DataModel):
    media: File | str


class FragmentHolder(DataModel):
    frag: VideoFragment


class NestedFragmentHolder(DataModel):
    holder: FragmentHolder


@pytest.mark.parametrize(
    "signal,msg",
    [
        (
            "frag",
            (
                "'frag' is VideoFragment, not a file; use signal='frag.video', "
                "or frag.save(output) in map()"
            ),
        ),
        ("pair", "'pair' is FilePair, not a file; use signal='pair.file'"),
        ("files", "'files' is list, not a file"),
        ("n", "'n' is int, not a file"),
        ("mixed", "'mixed' is int | str, not a file"),
    ],
)
def test_to_storage_rejects_a_signal_without_files(test_session, tmp_path, signal, msg):
    file = VideoFile(source=tmp_path.as_uri(), path="v.mp4")
    chain = dc.read_values(
        frag=[VideoFragment(video=file, start=0, end=1)],
        pair=[FilePair(file=file, n=1)],
        files=[[file]],
        n=[1],
        mixed=[1],
        output={
            "frag": VideoFragment,
            "pair": FilePair,
            "files": list[File],
            "n": int,
            "mixed": int | str,
        },
        session=test_session,
    )
    with pytest.raises(DataChainParamsError, match=re.escape(msg) + "$"):
        chain.to_storage(tmp_path / "out", signal=signal)


@pytest.mark.parametrize("num_threads", [1, 2])
@pytest.mark.parametrize("case", ["empty", "absent", "nonfiles", "mixed"])
def test_to_storage_validates_union_values(test_session, tmp_path, case, num_threads):
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.txt").write_text("content")
    file = File(source=source.as_uri(), path="a.txt")
    values = {
        "empty": [],
        "absent": [None],
        "nonfiles": ["plain", None],
        "mixed": ["plain", file, None],
    }[case]
    chain = dc.read_values(
        value=values, output={"value": File | str | None}, session=test_session
    )
    output = tmp_path / "out"
    if case == "nonfiles":
        with pytest.raises(DataChainParamsError, match="'value' contains no files"):
            chain.to_storage(output, signal="value", num_threads=num_threads)
        assert not output.exists()
    else:
        chain.to_storage(output, signal="value", num_threads=num_threads)
        files = [p for p in output.rglob("*") if p.is_file()]
        assert [p.read_text() for p in files] == (
            ["content"] if case == "mixed" else []
        )


@pytest.mark.parametrize("duplicate", [False, True])
def test_to_storage_filename_handles_file_subclass_union(
    test_session, tmp_path, duplicate
):
    (tmp_path / "left").mkdir()
    (tmp_path / "right").mkdir()
    (tmp_path / "left" / "a.bin").write_bytes(b"image")
    second = "a.bin" if duplicate else "b.bin"
    (tmp_path / "right" / second).write_bytes(b"video")
    chain = dc.read_values(
        value=[
            ImageFile(source=tmp_path.as_uri(), path="left/a.bin"),
            VideoFile(source=tmp_path.as_uri(), path=f"right/{second}"),
        ],
        output={"value": ImageFile | VideoFile},
        session=test_session,
    )
    output = tmp_path / "out"
    if duplicate:
        with pytest.raises(ValueError, match="Files with the same name found"):
            chain.to_storage(output, signal="value", placement="filename")
        assert not output.exists()
    else:
        chain.to_storage(output, signal="value", placement="filename")
        assert {p.name: p.read_bytes() for p in output.iterdir()} == {
            "a.bin": b"image",
            "b.bin": b"video",
        }


def test_to_storage_hint_includes_all_fragment_union_arms(test_session, tmp_path):
    chain = dc.read_values(
        frag=[VideoFragment(video=VideoFile(path="video.mp4"), start=0, end=1)],
        output={"frag": VideoFragment | AudioFragment},
        session=test_session,
    )
    with pytest.raises(DataChainParamsError) as error:
        chain.to_storage(tmp_path / "out", signal="frag")
    assert "signal='frag.VideoFragment.video'" in str(error.value)
    assert "signal='frag.AudioFragment.audio'" in str(error.value)


@pytest.mark.parametrize("signal", ["pair.file", "pair__file"])
@pytest.mark.parametrize("placement", ["filename", "filepath"])
def test_to_storage_normalizes_nested_signal(test_session, tmp_path, signal, placement):
    (tmp_path / "source.txt").write_text("content")
    chain = dc.read_values(
        pair=[FilePair(file=File(source=tmp_path.as_uri(), path="source.txt"), n=1)],
        session=test_session,
    )
    chain.to_storage(tmp_path / "out", signal=signal, placement=placement)
    assert (tmp_path / "out" / "source.txt").read_text() == "content"


def test_to_storage_hint_uses_nested_union_root(test_session, tmp_path):
    chain = dc.read_values(item=[MediaHolder(media="plain")], session=test_session)
    with pytest.raises(DataChainParamsError) as error:
        chain.to_storage(tmp_path / "out", signal="item")
    assert "signal='item.media'" in str(error.value)
    assert "item.media.File" not in str(error.value)
    with pytest.raises(DataChainParamsError, match="'item.media' contains no files"):
        chain.to_storage(tmp_path / "out", signal="item.media")


def test_to_storage_hint_normalizes_readable_union_arm(test_session, tmp_path):
    chain = dc.read_values(
        item=[
            FragmentHolder(
                frag=VideoFragment(video=VideoFile(path="v.mp4"), start=0, end=1)
            )
        ],
        output={"item": FragmentHolder | int},
        session=test_session,
    )
    with pytest.raises(DataChainParamsError) as error:
        chain.to_storage(tmp_path / "out", signal="item.frag")
    assert "signal='item.FragmentHolder.frag.video'" in str(error.value)


def test_to_storage_normalizes_mixed_delimiters_before_union_arm(
    test_session, tmp_path
):
    frag = VideoFragment(video=VideoFile(path="v.mp4"), start=0, end=1)
    chain = dc.read_values(
        item=[NestedFragmentHolder(holder=FragmentHolder(frag=frag))],
        output={"item": NestedFragmentHolder | int},
        session=test_session,
    )
    with pytest.raises(DataChainParamsError) as error:
        chain.to_storage(tmp_path / "out", signal="item__holder.frag")
    assert "signal='item.NestedFragmentHolder.holder.frag.video'" in str(error.value)
