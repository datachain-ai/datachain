import os
import posixpath
import subprocess
import sys
import textwrap
from pathlib import Path, PurePosixPath
from typing import cast
from urllib.parse import urlparse

import pytest
from PIL import Image

import datachain as dc
from datachain import Session
from datachain.lib.file import File, ImageFile
from tests.conftest import get_cloud_test_catalog, make_cloud_server
from tests.utils import images_equal, skip_if_not_sqlite

python_exc = sys.executable or "python3"


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
    call_count = {"count": 0}

    def mapper(file_path):
        call_count["count"] += 1
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

    assert call_count["count"] == len(expected)


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


def test_to_storage_relative_path(test_session, tmp_path):
    images = [
        {"name": "img1.jpg", "data": Image.new(mode="RGB", size=(64, 64))},
        {"name": "img2.jpg", "data": Image.new(mode="RGB", size=(128, 128))},
    ]

    for img in images:
        img["data"].save(tmp_path / img["name"])

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
def test_to_storage_keyboard_interrupt_does_not_lock_cleanup(tmp_path, catalog_tmpfile):
    script = tmp_path / "test_to_storage_interrupt.py"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for idx in range(3):
        (input_dir / f"{idx}.txt").write_text(f"file-{idx}")

    script.write_text(
        textwrap.dedent(f"""
            import os
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
                        os.kill(os.getpid(), signal.SIGINT)

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
                num_threads=1,
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

    assert result.returncode == 130
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
