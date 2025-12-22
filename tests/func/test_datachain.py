import json
import math
import os
import posixpath
import re
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import cast
from unittest.mock import Mock, patch
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import datachain as dc
from datachain import DataModel, func
from datachain.dataset import DatasetDependencyType
from datachain.lib.data_model import compute_model_fingerprint
from datachain.lib.file import File, ImageFile
from datachain.lib.listing import LISTING_TTL, is_listing_dataset, parse_listing_uri
from datachain.lib.tar import process_tar
from datachain.query.dataset import QueryStep
from tests.utils import (
    ANY_VALUE,
    TARRED_TREE,
    df_equal,
    images_equal,
    skip_if_not_sqlite,
    sorted_dicts,
    text_embedding,
)

DF_DATA = {
    "first_name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "age": [25, 30, 35, 40, 45],
    "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
}


def _get_listing_datasets(session):
    return sorted(
        [
            f"{ds.name}@v{ds.version}"
            for ds in dc.datasets(
                column="dataset", session=session, include_listing=True
            ).to_values("dataset")
            if is_listing_dataset(ds.name)
        ]
    )


@pytest.mark.parametrize("anon", [True, False])
def test_catalog_anon(tmp_dir, catalog, anon):
    chain = dc.read_storage(tmp_dir.as_uri(), anon=anon)
    assert chain.session.catalog.client_config.get("anon", False) is anon


def test_read_storage_client_config(tmp_dir):
    chain = dc.read_storage(tmp_dir.as_uri())
    assert chain.session.catalog.client_config == {}  # Default client config is set.

    chain = dc.read_storage(tmp_dir.as_uri(), client_config={"anon": True})
    assert chain.session.catalog.client_config == {
        "anon": True
    }  # New client config is set.


def test_read_storage(cloud_test_catalog):
    ctc = cloud_test_catalog
    chain = dc.read_storage(ctc.src_uri, session=ctc.session)
    assert chain.count() == 7


def test_read_storage_non_recursive(cloud_test_catalog):
    ctc = cloud_test_catalog
    chain = dc.read_storage(f"{ctc.src_uri}/dogs", session=ctc.session, recursive=False)
    assert chain.count() == 3


def test_read_storage_glob(cloud_test_catalog):
    ctc = cloud_test_catalog
    chain = dc.read_storage(f"{ctc.src_uri}/dogs/*", session=ctc.session)
    assert chain.count() == 3


def test_read_storage_as_image(cloud_test_catalog):
    ctc = cloud_test_catalog
    chain = dc.read_storage(ctc.src_uri, session=ctc.session, type="image")
    for im in chain.to_values("file"):
        assert isinstance(im, ImageFile)


def test_read_storage_reindex(tmp_dir, test_session):
    tmp_dir = tmp_dir / "parquets"
    path = tmp_dir.as_uri()
    os.mkdir(tmp_dir)

    pd.DataFrame({"name": ["Alice", "Bob"]}).to_parquet(tmp_dir / "test1.parquet")
    assert dc.read_storage(path, session=test_session).count() == 1

    pd.DataFrame({"name": ["Charlie", "David"]}).to_parquet(tmp_dir / "test2.parquet")
    assert dc.read_storage(path, session=test_session).count() == 1
    assert dc.read_storage(path, session=test_session, update=True).count() == 2


def test_read_storage_reindex_expired(tmp_dir, test_session):
    tmp_dir = tmp_dir / "parquets"
    os.mkdir(tmp_dir)
    uri = tmp_dir.as_uri()

    lst_ds_name = parse_listing_uri(uri)[0]

    pd.DataFrame({"name": ["Alice", "Bob"]}).to_parquet(tmp_dir / "test1.parquet")
    assert dc.read_storage(uri, session=test_session).count() == 1
    pd.DataFrame({"name": ["Charlie", "David"]}).to_parquet(tmp_dir / "test2.parquet")
    # mark dataset as expired
    test_session.catalog.metastore.update_dataset_version(
        test_session.catalog.get_dataset(lst_ds_name),
        "1.0.0",
        finished_at=datetime.now(timezone.utc) - timedelta(seconds=LISTING_TTL + 20),
    )

    # listing was updated because listing dataset was expired
    assert dc.read_storage(uri, session=test_session).count() == 2


@pytest.mark.parametrize(
    "cloud_type",
    ["s3", "azure", "gs"],
    indirect=True,
)
def test_read_storage_partials(cloud_test_catalog):
    ctc = cloud_test_catalog
    src_uri = ctc.src_uri
    session = ctc.session

    def _list_dataset_name(uri: str) -> str:
        name = parse_listing_uri(uri)[0]
        assert name
        return name

    dogs_uri = f"{src_uri}/dogs"
    dc.read_storage(dogs_uri, session=session).exec()
    assert _get_listing_datasets(session) == [
        f"{_list_dataset_name(dogs_uri)}@v1.0.0",
    ]

    dc.read_storage(f"{src_uri}/dogs/others", session=session)
    assert _get_listing_datasets(session) == [
        f"{_list_dataset_name(dogs_uri)}@v1.0.0",
    ]

    dc.read_storage(src_uri, session=session).exec()
    assert _get_listing_datasets(session) == sorted(
        [
            f"{_list_dataset_name(dogs_uri)}@v1.0.0",
            f"{_list_dataset_name(src_uri)}@v1.0.0",
        ]
    )

    dc.read_storage(f"{src_uri}/cats", session=session).exec()
    assert _get_listing_datasets(session) == sorted(
        [
            f"{_list_dataset_name(dogs_uri)}@v1.0.0",
            f"{_list_dataset_name(src_uri)}@v1.0.0",
        ]
    )


@pytest.mark.parametrize(
    "cloud_type",
    ["s3", "azure", "gs"],
    indirect=True,
)
def test_read_storage_partials_with_update(cloud_test_catalog):
    ctc = cloud_test_catalog
    src_uri = ctc.src_uri
    session = ctc.session

    def _list_dataset_name(uri: str) -> str:
        name = parse_listing_uri(uri)[0]
        assert name
        return name

    uri = f"{src_uri}/cats"
    dc.read_storage(uri, session=session).exec()
    assert _get_listing_datasets(session) == sorted(
        [
            f"{_list_dataset_name(uri)}@v1.0.0",
        ]
    )

    dc.read_storage(uri, session=session, update=True).exec()
    assert _get_listing_datasets(session) == sorted(
        [
            f"{_list_dataset_name(uri)}@v1.0.0",
            f"{_list_dataset_name(uri)}@v2.0.0",
        ]
    )


def test_read_storage_listing_happens_once(cloud_test_catalog, cloud_type):
    ctc = cloud_test_catalog
    uri = f"{ctc.src_uri}"
    ds_name = "cats_dogs"

    chain = dc.read_storage(uri, session=ctc.session)
    dc_cats = chain.filter(dc.C("file.path").glob("cats*"))
    dc_dogs = chain.filter(dc.C("file.path").glob("dogs*"))
    dc_cats.union(dc_dogs).save(ds_name)

    lst_ds_name = parse_listing_uri(uri)[0]
    assert _get_listing_datasets(ctc.session) == [f"{lst_ds_name}@v1.0.0"]


def test_read_storage_dependencies(cloud_test_catalog, cloud_type):
    ctc = cloud_test_catalog
    src_uri = ctc.src_uri
    uri = f"{src_uri}/cats"
    dep_name, _, _ = parse_listing_uri(uri)
    ds_name = "dep"
    dc.read_storage(uri, session=ctc.session).save(ds_name)
    dependencies = ctc.session.catalog.get_dataset_dependencies(ds_name, "1.0.0")
    assert len(dependencies) == 1
    assert dependencies[0].type == DatasetDependencyType.STORAGE
    assert dependencies[0].name == dep_name


def test_persist_not_affects_dependencies(tmp_dir, test_session):
    for i in range(4):
        (tmp_dir / f"file{i}.txt").write_text(f"file{i}")

    uri = tmp_dir.as_uri()
    dep_name, _, _ = parse_listing_uri(uri)
    chain = dc.read_storage(uri, session=test_session)  # .persist()
    # calling multiple persists to create temp datasets
    chain = chain.persist()
    chain = chain.persist()
    chain = chain.persist()
    chain.save("test-data")
    dependencies = test_session.catalog.get_dataset_dependencies("test-data", "1.0.0")

    assert len(dependencies) == 1
    assert dependencies[0].name == dep_name
    assert dependencies[0].type == DatasetDependencyType.STORAGE


@pytest.mark.parametrize("use_cache", [False, True])
def test_read_file(cloud_test_catalog, use_cache):
    ctc = cloud_test_catalog

    chain = dc.read_storage(ctc.src_uri, session=ctc.session)
    for file in chain.settings(cache=use_cache).to_values("file"):
        assert file.get_local_path() is None
        file.read()
        assert bool(file.get_local_path()) is use_cache


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
    mapper = Mock(side_effect=lambda file_path: len(file_path))

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

    assert mapper.call_count == len(expected)


@pytest.mark.parametrize("use_cache", [True, False])
def test_export_images_files(test_session, tmp_dir, tmp_path, use_cache):
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
    ).settings(cache=use_cache).to_storage(tmp_dir / "output", placement="filename")

    for img in images:
        with Image.open(tmp_dir / "output" / img["name"]) as exported_img:
            assert images_equal(img["data"], exported_img)


@pytest.mark.parametrize("use_cache", [True, False])
def test_read_storage_multiple_uris_files(test_session, tmp_dir, tmp_path, use_cache):
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
        ]
    )
    assert chain.count() == 4

    chain = dc.read_storage([f"file://{tmp_dir}/output/"])
    assert chain.count() == 2


@pytest.mark.parametrize(
    "cloud_type",
    ["s3", "azure", "gs"],
    indirect=True,
)
def test_read_storage_multiple_uris_cache(cloud_test_catalog):
    ctc = cloud_test_catalog
    src_uri = ctc.src_uri
    session = ctc.session

    with pytest.raises(ValueError):
        dc.read_storage([])  # No URIs provided

    with patch(
        "datachain.lib.dc.storage.get_listing", wraps=dc.lib.listing.get_listing
    ) as mock_get_listing:
        chain = dc.read_storage(
            [
                f"{src_uri}/cats",
                f"{src_uri}/dogs",
                f"{src_uri}/cats/cat*",
                f"{src_uri}/dogs/dog*",
            ],
            session=session,
            update=True,
        ).exec()

        assert chain.count() == 11

        files = chain.to_values("file")
        assert {f.name for f in files} == {
            "cat1",
            "cat2",
            "dog1",
            "dog2",
            "dog3",
            "dog4",
        }

        # Verify read_records was called exactly twice
        assert mock_get_listing.call_count == 4  # TODO FIX THIS


def test_read_storage_path_object(test_session, tmp_dir, tmp_path):
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

    # create bucket dir with duplicate file names
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


def test_show(capsys, test_session):
    first_name = ["Alice", "Bob", "Charlie"]
    dc.read_values(
        first_name=first_name,
        age=[40, 30, None],
        city=[
            "Houston",
            "Los Angeles",
            None,
        ],
        session=test_session,
    ).order_by("first_name").show()
    captured = capsys.readouterr()
    normalized_output = re.sub(r"\s+", " ", captured.out)
    assert "first_name age city" in normalized_output
    for i in range(3):
        assert f"{i} {first_name[i]}" in normalized_output


@skip_if_not_sqlite
def test_show_preserves_none(capsys, test_session):
    chain = dc.read_values(
        score=[1, None],
        ts=[
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            None,
        ],
        session=test_session,
    )

    chain.show()

    captured = capsys.readouterr().out
    assert "NaN" not in captured
    assert "NaT" not in captured
    assert captured.count("None") >= 2


def test_show_without_temp_datasets(capsys, test_session):
    dc.read_values(
        key=[1, 2, 3, 4], session=test_session
    ).persist()  # creates temp dataset
    dc.datasets().show()
    captured = capsys.readouterr()
    normalized_output = re.sub(r"\s+", " ", captured.out)
    print(normalized_output)
    assert "Empty result" in normalized_output


def test_class_method_deprecated(capsys, test_session):
    with pytest.warns(DeprecationWarning):
        dc.DataChain.from_values(key=["a", "b", "c"], session=test_session)


def test_save(test_session):
    chain = dc.read_values(key=["a", "b", "c"])
    chain.save(
        name="new_name",
        version="1.0.0",
        description="new description",
        attrs=["new_label", "old_label"],
    )

    ds = test_session.catalog.get_dataset("new_name")
    assert ds.name == "new_name"
    assert ds.description == "new description"
    assert ds.attrs == ["new_label", "old_label"]

    chain.save(
        name="new_name",
        description="updated description",
        attrs=["new_label", "old_label", "new_label2"],
    )
    ds = test_session.catalog.get_dataset("new_name")
    assert ds.name == "new_name"
    assert ds.description == "updated description"
    assert ds.attrs == ["new_label", "old_label", "new_label2"]


def test_show_nested_empty(capsys, test_session):
    files = [
        File(size=s, path=p) for p, s in zip(list("abcde"), range(5), strict=False)
    ]
    dc.read_values(file=files, session=test_session).limit(0).show()

    captured = capsys.readouterr()
    normalized_output = re.sub(r"\s+", " ", captured.out)
    assert "Empty result" in normalized_output
    assert "('file', 'path')" in normalized_output


def test_show_empty(capsys, test_session):
    first_name = ["Alice", "Bob", "Charlie"]
    dc.read_values(first_name=first_name, session=test_session).limit(0).show()

    captured = capsys.readouterr()
    normalized_output = re.sub(r"\s+", " ", captured.out)
    assert "Empty result" in normalized_output
    assert "Columns: ['first_name']" in normalized_output


def test_show_limit(capsys, test_session):
    first_name = ["Alice", "Bob", "Charlie"]
    dc.read_values(
        first_name=first_name,
        age=[40, 30, None],
        city=[
            "Houston",
            "Los Angeles",
            None,
        ],
        session=test_session,
    ).limit(1).show()
    captured = capsys.readouterr()
    new_line_count = captured.out.count("\n")
    assert new_line_count == 2


def test_show_transpose(capsys, test_session):
    first_name = ["Alice", "Bob", "Charlie"]
    last_name = ["A", "B", "C"]
    dc.read_values(
        first_name=first_name,
        last_name=last_name,
        session=test_session,
    ).order_by("first_name", "last_name").show(transpose=True)
    captured = capsys.readouterr()
    stripped_output = re.sub(r"\s+", " ", captured.out)
    assert " ".join(first_name) in stripped_output
    assert " ".join(last_name) in stripped_output


def test_show_truncate(capsys, test_session):
    client = ["Alice A", "Bob B", "Charles C"]
    details = [
        "This is a very long piece of text that would not fit in the default output "
        "because pandas will truncate the column",
        "Gives good tips",
        "Not very nice",
    ]

    chain = dc.read_values(
        client=client,
        details=details,
        session=test_session,
    )

    chain.show()
    captured = capsys.readouterr()
    normalized_output = re.sub(r"\s+", " ", captured.out)
    assert f"{client[0]} {details[0][:10]}" in normalized_output
    assert details[0] not in normalized_output
    for i in [1, 2]:
        assert f"{client[i]} {details[i]}" in normalized_output


def test_show_no_truncate(capsys, test_session):
    client = ["Alice A", "Bob B", "Charles C"]
    details = [
        "This is a very long piece of text that would not fit in the default output "
        "because pandas will truncate the column",
        "Gives good tips",
        "Not very nice",
    ]

    chain = dc.read_values(
        client=client,
        details=details,
        session=test_session,
    )

    chain.show(truncate=False)
    captured = capsys.readouterr()
    normalized_output = re.sub(r"\s+", " ", captured.out)
    for i in range(3):
        assert client[i] in normalized_output
        assert details[i] in normalized_output


@pytest.mark.parametrize("ordered_by", ["letter", "number"])
def test_show_ordered(capsys, test_session, ordered_by):
    numbers = [6, 2, 3, 1, 5, 7, 4]
    letters = ["u", "y", "x", "z", "v", "t", "w"]

    dc.read_values(number=numbers, letter=letters, session=test_session).order_by(
        ordered_by
    ).show()

    captured = capsys.readouterr()
    normalized_lines = [
        re.sub(r"\s+", " ", line).strip() for line in captured.out.strip().split("\n")
    ]

    ordered_entries = sorted(
        zip(numbers, letters, strict=False),
        key=lambda x: x[0 if ordered_by == "number" else 1],
    )

    assert normalized_lines[0].strip() == "number letter"
    for i, line in enumerate(normalized_lines[1:]):
        number, letter = ordered_entries[i]
        assert line == f"{i} {number} {letter}"


def test_read_storage_dataset_stats(tmp_dir, test_session):
    for i in range(4):
        (tmp_dir / f"file{i}.txt").write_text(f"file{i}")

    chain = dc.read_storage(tmp_dir.as_uri(), session=test_session).save("test-data")
    version = test_session.catalog.get_dataset(chain.name).get_version(chain.version)
    assert version.num_objects == 4
    assert version.size == 20


def test_read_storage_check_rows(tmp_dir, test_session):
    stats = {}
    for i in range(4):
        file = tmp_dir / f"{i}.txt"
        file.write_text(f"file{i}")
        stats[file.name] = file.stat()

    chain = dc.read_storage(tmp_dir.as_uri(), session=test_session).save("test-data")

    for file in chain.to_values("file"):
        assert isinstance(file, File)
        stat = stats[file.name]
        assert file == File(
            source=Path(tmp_dir).as_uri(),
            path=file.path,
            size=stat.st_size,
            version="",
            etag=stat.st_mtime.hex(),
            is_latest=True,
            last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            location=None,
        )


@pytest.mark.parametrize("processes", [False, 2, True])
@pytest.mark.xdist_group(name="tmpfile")
def test_parallel(processes, test_session_tmpfile):
    prefix = "t & "
    vals = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

    res = list(
        dc.read_values(key=vals, session=test_session_tmpfile)
        .settings(parallel=processes)
        .map(res=lambda key: prefix + key)
        .order_by("res")
        .to_values("res")
    )

    assert res == [prefix + v for v in vals]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_avoid_recalculation_after_save(cloud_test_catalog, monkeypatch):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

    calls = 0

    def name_len(path):
        nonlocal calls
        calls += 1
        return (len(path),)

    uri = cloud_test_catalog.src_uri
    session = cloud_test_catalog.session
    ds = (
        dc.read_storage(uri, session=session)
        .filter(dc.C("file.path").glob("*/dog1"))
        .map(name_len, params=["file.path"], output={"name_len": int})
    )
    ds2 = ds.save("ds1")

    assert ds2._query.steps == []
    assert ds2._query.dependencies == set()
    assert isinstance(ds2._query.starting_step, QueryStep)
    ds2.save("ds2")
    assert calls == 1  # UDF should be called only once


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_number_with_order_by_name_len_desc_and_name_asc(cloud_test_catalog):
    session = cloud_test_catalog.session

    path = cloud_test_catalog.src_uri
    ds_name = uuid.uuid4().hex

    def name_len(path):
        return (len(posixpath.basename(path)),)

    dc.read_storage(path, session=session).map(
        name_len, params=["file.path"], output={"name_len": int}
    ).order_by("name_len", descending=True).order_by("file.path").save(ds_name)

    assert dc.read_dataset(name=ds_name, session=session).to_list(
        "sys.id", "file.path"
    ) == [
        (1, "description"),
        (2, "cats/cat1"),
        (3, "cats/cat2"),
        (4, "dogs/dog1"),
        (5, "dogs/dog2"),
        (6, "dogs/dog3"),
        (7, "dogs/others/dog4"),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_number_with_order_by_before_map(cloud_test_catalog):
    session = cloud_test_catalog.session

    path = cloud_test_catalog.src_uri
    ds_name = uuid.uuid4().hex

    def name_len(path):
        return (len(posixpath.basename(path)),)

    dc.read_storage(path, session=session).order_by("file.path").map(
        name_len, params=["file.path"], output={"name_len": int}
    ).save(ds_name)

    # we should preserve order in final result based on order by which was added
    # before add_signals
    assert dc.read_dataset(name=ds_name, session=session).to_list(
        "sys.id", "file.path"
    ) == [
        (1, "cats/cat1"),
        (2, "cats/cat2"),
        (3, "description"),
        (4, "dogs/dog1"),
        (5, "dogs/dog2"),
        (6, "dogs/dog3"),
        (7, "dogs/others/dog4"),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_gen_parallel(cloud_test_catalog_tmpfile):
    session = cloud_test_catalog_tmpfile.session

    def func(file) -> Iterator[tuple[str]]:
        for i in range(5):
            yield (f"{file.path}_{i}",)

    chain = (
        dc.read_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .settings(parallel=True)
        .gen(gen=func, params=["file"], output={"val": str})
        .order_by("val")
    )
    assert chain.to_values("val") == [
        "cats/cat1_0",
        "cats/cat1_1",
        "cats/cat1_2",
        "cats/cat1_3",
        "cats/cat1_4",
        "cats/cat2_0",
        "cats/cat2_1",
        "cats/cat2_2",
        "cats/cat2_3",
        "cats/cat2_4",
        "description_0",
        "description_1",
        "description_2",
        "description_3",
        "description_4",
        "dogs/dog1_0",
        "dogs/dog1_1",
        "dogs/dog1_2",
        "dogs/dog1_3",
        "dogs/dog1_4",
        "dogs/dog2_0",
        "dogs/dog2_1",
        "dogs/dog2_2",
        "dogs/dog2_3",
        "dogs/dog2_4",
        "dogs/dog3_0",
        "dogs/dog3_1",
        "dogs/dog3_2",
        "dogs/dog3_3",
        "dogs/dog3_4",
        "dogs/others/dog4_0",
        "dogs/others/dog4_1",
        "dogs/others/dog4_2",
        "dogs/others/dog4_3",
        "dogs/others/dog4_4",
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_gen_with_new_columns_numpy(cloud_test_catalog, dogs_dataset):
    session = cloud_test_catalog.session

    def gen_numpy():
        for _ in range(10):
            yield (
                np.int32(11),
                np.int64(12),
                np.float32(0.5),
                np.float64(0.5),
                np.int32(13),
                np.array([[0.5], [0.5]], dtype=np.float32),
                np.array([0.5, 0.5], dtype=np.float32),
                np.array([0.5, 0.5], dtype=np.float64),
                np.array([14, 15], dtype=np.int32),
                np.array([], dtype=np.float32),
            )

    dc.read_storage(cloud_test_catalog.src_uri, session=session).gen(
        subobject=gen_numpy,
        output={
            "int_col_32": int,
            "int_col_64": int,
            "float_col_32": float,
            "float_col_64": float,
            "int_float_col_32": float,
            "array_col_nested": list[list[float]],
            "array_col_32": list[float],
            "array_col_64": list[float],
            "array_int_float_col_32": list[float],
            "array_empty_col_32": list[float],
        },
    ).save("dogs_with_rows_and_signals")

    chain = dc.read_dataset(name="dogs_with_rows_and_signals", session=session)
    for r in chain.to_iter(
        "int_col_32",
        "int_col_64",
        "float_col_32",
        "float_col_64",
        "int_float_col_32",
        "array_col_nested",
        "array_col_32",
        "array_col_64",
        "array_int_float_col_32",
        "array_empty_col_32",
    ):
        assert r == (
            11,
            12,
            0.5,
            0.5,
            13.0,
            [[0.5], [0.5]],
            [0.5, 0.5],
            [0.5, 0.5],
            [14.0, 15.0],
            [],
        )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_gen_with_new_columns_wrong_type(cloud_test_catalog, dogs_dataset):
    session = cloud_test_catalog.session

    def gen_func():
        yield (0.5)

    with pytest.raises(ValueError):
        dc.read_storage(cloud_test_catalog.src_uri, session=session).gen(
            new_val=gen_func, output={"new_val": int}
        ).show()


def test_similarity_search(cloud_test_catalog):
    session = cloud_test_catalog.session
    src_uri = cloud_test_catalog.src_uri

    def calc_emb(file):
        text = file.read().decode("utf-8")
        return text_embedding(text)

    target_embedding = (
        dc.read_storage(src_uri, session=session)
        .filter(dc.C("file.path").glob("*description"))
        .order_by("file.path")
        .limit(1)
        .map(embedding=calc_emb, output={"embedding": list[float]})
    ).to_values("embedding")[0]

    chain = (
        dc.read_storage(src_uri, session=session)
        .map(embedding=calc_emb, output={"embedding": list[float]})
        .mutate(
            cos_dist=func.cosine_distance("embedding", target_embedding),
            eucl_dist=func.euclidean_distance("embedding", target_embedding),
        )
        .order_by("file.path")
    )
    count = chain.count()
    assert count == 7

    expected = [
        ("cats/cat1", 0.8508677010357059, 1.9078358385397216),
        ("cats/cat2", 0.8508677010357059, 1.9078358385397216),
        ("description", 0.0, 0.0),
        ("dogs/dog1", 0.7875133863812602, 1.8750659656122843),
        ("dogs/dog2", 0.7356502722055684, 1.775619888314893),
        ("dogs/dog3", 0.7695916496857775, 1.8344983482620636),
        ("dogs/others/dog4", 0.9789704524691446, 2.0531542018152322),
    ]

    for (p1, c1, e1), (p2, c2, e2) in zip(
        chain.to_iter("file.path", "cos_dist", "eucl_dist"), expected, strict=False
    ):
        assert p1.endswith(p2)
        assert math.isclose(c1, c2, abs_tol=1e-5)
        assert math.isclose(e1, e2, abs_tol=1e-5)


@pytest.mark.parametrize("tree", [TARRED_TREE], indirect=True)
def test_process_and_open_tar(cloud_test_catalog, cloud_type):
    ctc = cloud_test_catalog
    chain = (
        dc.read_storage(ctc.src_uri, session=ctc.session)
        .settings(cache=True, prefetch=2)
        .gen(file=process_tar)
        .map(content=lambda file: str(file.read(), encoding="utf-8"))
    )
    assert chain.count() == 7

    assert {
        (content, file.path) for file, content in chain.to_iter("file", "content")
    } == {
        ("meow", "animals.tar/cats/cat1"),
        ("mrow", "animals.tar/cats/cat2"),
        ("Cats and Dogs", "animals.tar/description"),
        ("woof", "animals.tar/dogs/dog1"),
        ("arf", "animals.tar/dogs/dog2"),
        ("bark", "animals.tar/dogs/dog3"),
        ("ruff", "animals.tar/dogs/others/dog4"),
    }


def test_group_by_signals(cloud_test_catalog):
    from datachain import func

    session = cloud_test_catalog.session
    src_uri = cloud_test_catalog.src_uri

    class FileInfo(DataModel):
        path: str = ""
        name: str = ""

    def file_info(file: File) -> FileInfo:
        full_path = file.source.rstrip("/") + "/" + file.path
        rel_path = posixpath.relpath(full_path, src_uri)
        path_parts = rel_path.split("/", 1)
        return FileInfo(
            path=path_parts[0] if len(path_parts) > 1 else "",
            name=path_parts[1] if len(path_parts) > 1 else path_parts[0],
        )

    ds = (
        dc.read_storage(src_uri, session=session)
        .map(file_info, params=["file"], output={"file_info": FileInfo})
        .group_by(
            cnt=func.count(),
            sum=func.sum("file.size"),
            value=func.any_value("file.size"),
            partition_by="file_info.path",
        )
        .save("my-ds")
    )

    fp_file_info_path = compute_model_fingerprint(FileInfo, {"path": None})
    fi_partial_base = f"FileInfoPartial_{fp_file_info_path[:10]}"
    fi_partial_name = f"{fi_partial_base}@v1"

    assert ds.signals_schema.serialize() == {
        "_custom_types": {
            fi_partial_name: {
                "bases": [
                    (
                        fi_partial_base,
                        "datachain.lib.signal_schema",
                        fi_partial_name,
                    ),
                    ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                    ("BaseModel", "pydantic.main", None),
                    ("object", "builtins", None),
                ],
                "fields": {"path": "str"},
                "hidden_fields": [],
                "name": fi_partial_name,
                "schema_version": 2,
                "partial_fingerprint": fp_file_info_path,
            }
        },
        "file_info": fi_partial_name,
        "cnt": "int",
        "sum": "int",
        "value": "int",
    }
    assert sorted_dicts(ds.to_records(), "file_info__path") == sorted_dicts(
        [
            {"file_info__path": "", "cnt": 1, "sum": 13, "value": ANY_VALUE(13)},
            {"file_info__path": "cats", "cnt": 2, "sum": 8, "value": ANY_VALUE(4)},
            {"file_info__path": "dogs", "cnt": 4, "sum": 15, "value": ANY_VALUE(3, 4)},
        ],
        "file_info__path",
    )


def test_group_by_signals_same_model(cloud_test_catalog):
    from datachain import func

    session = cloud_test_catalog.session
    src_uri = cloud_test_catalog.src_uri

    class FileInfo(DataModel):
        path: str = ""
        name: str = ""

    def file_info(file: File) -> FileInfo:
        full_path = file.source.rstrip("/") + "/" + file.path
        rel_path = posixpath.relpath(full_path, src_uri)
        path_parts = rel_path.split("/", 1)
        return FileInfo(
            path=path_parts[0] if len(path_parts) > 1 else "",
            name=path_parts[1] if len(path_parts) > 1 else path_parts[0],
        )

    ds = (
        dc.read_storage(src_uri, session=session)
        .map(f1=file_info)
        .map(f2=file_info)
        .group_by(
            cnt=func.count(),
            sum=func.sum("file.size"),
            partition_by=("f1.name", "f2.path"),
        )
        .save("my-ds")
    )

    fp_file_info_name = compute_model_fingerprint(FileInfo, {"name": None})
    fp_file_info_path = compute_model_fingerprint(FileInfo, {"path": None})
    fi_name_base = f"FileInfoPartial_{fp_file_info_name[:10]}"
    fi_name_full = f"{fi_name_base}@v1"
    fi_path_base = f"FileInfoPartial_{fp_file_info_path[:10]}"
    fi_path_full = f"{fi_path_base}@v1"

    assert ds.signals_schema.serialize() == {
        "_custom_types": {
            fi_name_full: {
                "bases": [
                    (fi_name_base, "datachain.lib.signal_schema", fi_name_full),
                    ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                    ("BaseModel", "pydantic.main", None),
                    ("object", "builtins", None),
                ],
                "fields": {"name": "str"},
                "hidden_fields": [],
                "name": fi_name_full,
                "schema_version": 2,
                "partial_fingerprint": fp_file_info_name,
            },
            fi_path_full: {
                "bases": [
                    (fi_path_base, "datachain.lib.signal_schema", fi_path_full),
                    ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                    ("BaseModel", "pydantic.main", None),
                    ("object", "builtins", None),
                ],
                "fields": {"path": "str"},
                "hidden_fields": [],
                "name": fi_path_full,
                "schema_version": 2,
                "partial_fingerprint": fp_file_info_path,
            },
        },
        "f1": fi_name_full,
        "f2": fi_path_full,
        "cnt": "int",
        "sum": "int",
    }
    assert sorted_dicts(ds.to_records(), "f1__name", "f2__path") == sorted_dicts(
        [
            {"f1__name": "cat1", "f2__path": "cats", "cnt": 1, "sum": 4},
            {"f1__name": "cat2", "f2__path": "cats", "cnt": 1, "sum": 4},
            {"f1__name": "description", "f2__path": "", "cnt": 1, "sum": 13},
            {"f1__name": "dog1", "f2__path": "dogs", "cnt": 1, "sum": 4},
            {"f1__name": "dog2", "f2__path": "dogs", "cnt": 1, "sum": 3},
            {"f1__name": "dog3", "f2__path": "dogs", "cnt": 1, "sum": 4},
            {"f1__name": "others/dog4", "f2__path": "dogs", "cnt": 1, "sum": 4},
        ],
        "f1__name",
        "f2__path",
    )


def test_group_by_signals_nested(cloud_test_catalog):
    from datachain import func

    session = cloud_test_catalog.session
    src_uri = cloud_test_catalog.src_uri

    class FileName(DataModel):
        name: str = ""
        ext: str = ""

    class FileInfo(DataModel):
        path: str = ""
        name: FileName

    def file_info(file: File) -> FileInfo:
        full_path = file.source.rstrip("/") + "/" + file.path
        rel_path = posixpath.relpath(full_path, src_uri)
        path_parts = rel_path.split("/", 1)
        file_name = path_parts[1] if len(path_parts) > 1 else path_parts[0]
        return FileInfo(
            path=path_parts[0] if len(path_parts) > 1 else "",
            name=FileName(
                name=file_name,
                ext=posixpath.splitext(file_name)[1].lstrip("."),
            ),
        )

    ds = (
        dc.read_storage(src_uri, session=session)
        .map(f1=file_info)
        .map(f2=file_info)
        .group_by(
            cnt=func.count(),
            sum=func.sum("file.size"),
            partition_by=("f1.name.name", "f2.path"),
        )
        .save("my-ds")
    )

    fp_file_info_name = compute_model_fingerprint(FileInfo, {"name": {"name": None}})
    fp_file_info_path = compute_model_fingerprint(FileInfo, {"path": None})
    fp_file_name = compute_model_fingerprint(FileName, {"name": None})

    fi_name_base = f"FileInfoPartial_{fp_file_info_name[:10]}"
    fi_name_full = f"{fi_name_base}@v1"
    fi_path_base = f"FileInfoPartial_{fp_file_info_path[:10]}"
    fi_path_full = f"{fi_path_base}@v1"
    fname_base = f"FileNamePartial_{fp_file_name[:10]}"
    fname_full = f"{fname_base}@v1"

    assert ds.signals_schema.serialize() == {
        "_custom_types": {
            fname_full: {
                "bases": [
                    (fname_base, "datachain.lib.signal_schema", fname_full),
                    ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                    ("BaseModel", "pydantic.main", None),
                    ("object", "builtins", None),
                ],
                "fields": {"name": "str"},
                "hidden_fields": [],
                "name": fname_full,
                "schema_version": 2,
                "partial_fingerprint": fp_file_name,
            },
            fi_name_full: {
                "bases": [
                    (fi_name_base, "datachain.lib.signal_schema", fi_name_full),
                    ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                    ("BaseModel", "pydantic.main", None),
                    ("object", "builtins", None),
                ],
                "fields": {"name": fname_full},
                "hidden_fields": [],
                "name": fi_name_full,
                "schema_version": 2,
                "partial_fingerprint": fp_file_info_name,
            },
            fi_path_full: {
                "bases": [
                    (fi_path_base, "datachain.lib.signal_schema", fi_path_full),
                    ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                    ("BaseModel", "pydantic.main", None),
                    ("object", "builtins", None),
                ],
                "fields": {"path": "str"},
                "hidden_fields": [],
                "name": fi_path_full,
                "schema_version": 2,
                "partial_fingerprint": fp_file_info_path,
            },
        },
        "f1": fi_name_full,
        "f2": fi_path_full,
        "cnt": "int",
        "sum": "int",
    }
    assert sorted_dicts(ds.to_records(), "f1__name__name", "f2__path") == sorted_dicts(
        [
            {"f1__name__name": "cat1", "f2__path": "cats", "cnt": 1, "sum": 4},
            {"f1__name__name": "cat2", "f2__path": "cats", "cnt": 1, "sum": 4},
            {"f1__name__name": "description", "f2__path": "", "cnt": 1, "sum": 13},
            {"f1__name__name": "dog1", "f2__path": "dogs", "cnt": 1, "sum": 4},
            {"f1__name__name": "dog2", "f2__path": "dogs", "cnt": 1, "sum": 3},
            {"f1__name__name": "dog3", "f2__path": "dogs", "cnt": 1, "sum": 4},
            {"f1__name__name": "others/dog4", "f2__path": "dogs", "cnt": 1, "sum": 4},
        ],
        "f1__name__name",
        "f2__path",
    )


def test_group_by_known_signals(cloud_test_catalog):
    from datachain import func
    from datachain.model import BBox

    session = cloud_test_catalog.session
    src_uri = cloud_test_catalog.src_uri

    def process(file: File) -> BBox:
        return BBox(title=file.path.split("/")[0], coords=[10, 20, 80, 90])

    ds = (
        dc.read_storage(src_uri, session=session)
        .map(box=process)
        .group_by(
            cnt=func.count(),
            value=func.any_value("box.coords"),
            partition_by="box.title",
        )
        .save("my-ds")
    )

    fp_bbox_title = compute_model_fingerprint(BBox, {"title": None})
    bbox_base = f"BBoxPartial_{fp_bbox_title[:10]}"
    bbox_full = f"{bbox_base}@v1"

    assert ds.signals_schema.serialize() == {
        "_custom_types": {
            bbox_full: {
                "bases": [
                    (bbox_base, "datachain.lib.signal_schema", bbox_full),
                    ("DataModel", "datachain.lib.data_model", "DataModel@v1"),
                    ("BaseModel", "pydantic.main", None),
                    ("object", "builtins", None),
                ],
                "fields": {"title": "str"},
                "hidden_fields": [],
                "name": bbox_full,
                "schema_version": 2,
                "partial_fingerprint": fp_bbox_title,
            }
        },
        "box": bbox_full,
        "cnt": "int",
        "value": "list[int]",
    }
    assert sorted_dicts(ds.to_records(), "box__title") == sorted_dicts(
        [
            {"box__title": "cats", "cnt": 2, "value": [10, 20, 80, 90]},
            {"box__title": "description", "cnt": 1, "value": [10, 20, 80, 90]},
            {"box__title": "dogs", "cnt": 4, "value": [10, 20, 80, 90]},
        ],
        "box__title",
    )


def test_group_by_func(cloud_test_catalog):
    from datachain import func

    session = cloud_test_catalog.session
    src_uri = cloud_test_catalog.src_uri

    ds = (
        dc.read_storage(src_uri, session=session)
        .group_by(
            cnt=func.count(),
            sum=func.sum("file.size"),
            partition_by=func.path.parent("file.path").label("file_dir"),
        )
        .save("my-ds")
    )

    assert ds.signals_schema.serialize() == {
        "file_dir": "str",
        "cnt": "int",
        "sum": "int",
    }
    assert sorted_dicts(ds.to_records(), "file_dir") == sorted_dicts(
        [
            {"file_dir": "", "cnt": 1, "sum": 13},
            {"file_dir": "cats", "cnt": 2, "sum": 8},
            {"file_dir": "dogs", "cnt": 3, "sum": 11},
            {"file_dir": "dogs/others", "cnt": 1, "sum": 4},
        ],
        "file_dir",
    )


@pytest.mark.parametrize("partition_by", ["file_info.path", "file_info__path"])
@pytest.mark.parametrize("order_by", ["file_info.name", "file_info__name"])
def test_window_signals(cloud_test_catalog, partition_by, order_by):
    from datachain import func

    session = cloud_test_catalog.session
    src_uri = cloud_test_catalog.src_uri

    class FileInfo(DataModel):
        path: str = ""
        name: str = ""

    def file_info(file: File) -> FileInfo:
        full_path = file.source.rstrip("/") + "/" + file.path
        rel_path = posixpath.relpath(full_path, src_uri)
        path_parts = rel_path.split("/", 1)
        return FileInfo(
            path=path_parts[0] if len(path_parts) > 1 else "",
            name=path_parts[1] if len(path_parts) > 1 else path_parts[0],
        )

    window = func.window(partition_by=partition_by, order_by=order_by, desc=True)

    ds = (
        dc.read_storage(src_uri, session=session)
        .map(file_info, params=["file"], output={"file_info": FileInfo})
        .mutate(row_number=func.row_number().over(window))
        .save("my-ds")
    )

    results = {}
    for r in ds.to_records():
        filename = (
            r["file_info__path"] + "/" + r["file_info__name"]
            if r["file_info__path"]
            else r["file_info__name"]
        )
        results[filename] = r["row_number"]

    assert results == {
        "cats/cat2": 1,
        "cats/cat1": 2,
        "description": 1,
        "dogs/others/dog4": 1,
        "dogs/dog3": 2,
        "dogs/dog2": 3,
        "dogs/dog1": 4,
    }


def test_window_signals_random(cloud_test_catalog):
    from datachain import func

    session = cloud_test_catalog.session
    src_uri = cloud_test_catalog.src_uri

    class FileInfo(DataModel):
        path: str = ""
        name: str = ""

    def file_info(file: File) -> FileInfo:
        full_path = file.source.rstrip("/") + "/" + file.path
        rel_path = posixpath.relpath(full_path, src_uri)
        path_parts = rel_path.split("/", 1)
        return FileInfo(
            path=path_parts[0] if len(path_parts) > 1 else "",
            name=path_parts[1] if len(path_parts) > 1 else path_parts[0],
        )

    window = func.window(partition_by="file_info.path", order_by="sys.rand")

    ds = (
        dc.read_storage(src_uri, session=session)
        .map(file_info, params=["file"], output={"file_info": FileInfo})
        .mutate(row_number=func.row_number().over(window))
        .filter(dc.C("row_number") < 3)
        .select_except("row_number")
        .save("my-ds")
    )

    results: dict[str, list[str]] = {}
    for r in ds.to_records():
        results.setdefault(r["file_info__path"], []).append(r["file_info__name"])

    assert results[""] == ["description"]
    assert sorted(results["cats"]) == sorted(["cat1", "cat2"])

    assert len(results["dogs"]) == 2
    all_dogs = ["dog1", "dog2", "dog3", "others/dog4"]
    for dog in results["dogs"]:
        assert dog in all_dogs
        all_dogs.remove(dog)
    assert len(all_dogs) == 2


def test_to_read_csv_remote(cloud_test_catalog_upload):
    ctc = cloud_test_catalog_upload
    path = f"{ctc.src_uri}/test.csv"

    df = pd.DataFrame(DF_DATA)
    dc_to = dc.read_pandas(df, session=ctc.session)
    dc_to.to_csv(path)

    dc_from = dc.read_csv(path, session=ctc.session)
    df1 = dc_from.select("first_name", "age", "city").to_pandas()
    assert df_equal(df1, df)


@pytest.mark.parametrize("chunk_size", (1000, 2))
@pytest.mark.parametrize("kwargs", ({}, {"compression": "gzip"}))
def test_to_read_parquet_remote(cloud_test_catalog_upload, chunk_size, kwargs):
    ctc = cloud_test_catalog_upload
    path = f"{ctc.src_uri}/test.parquet"

    df = pd.DataFrame(DF_DATA)
    dc_to = dc.read_pandas(df, session=ctc.session)
    dc_to.to_parquet(path, chunk_size=chunk_size, **kwargs)

    dc_from = dc.read_parquet(path, session=ctc.session)
    df1 = dc_from.select("first_name", "age", "city").to_pandas()

    assert df_equal(df1, df)


def test_to_read_parquet_partitioned_remote(cloud_test_catalog_upload):
    ctc = cloud_test_catalog_upload
    path = f"{ctc.src_uri}/parquets"

    df = pd.DataFrame(DF_DATA)
    dc_to = dc.read_pandas(df, session=ctc.session)
    dc_to.to_parquet(path, partition_cols=["first_name"], chunk_size=2)

    dc_from = dc.read_parquet(path, session=ctc.session)
    df1 = dc_from.select("first_name", "age", "city").to_pandas()
    df1 = df1.sort_values("first_name").reset_index(drop=True)
    assert df_equal(df1, df)


# These deprecation warnings occur in the datamodel-code-generator package.
@pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20")
def test_to_read_json(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    dc_to = dc.read_pandas(df, session=test_session)
    path = tmp_dir / "test.json"
    dc_to.order_by("first_name", "age").to_json(path)

    with open(path) as f:
        values = json.load(f)
    assert values == [
        {"first_name": n, "age": a, "city": c}
        for n, a, c in zip(
            DF_DATA["first_name"], DF_DATA["age"], DF_DATA["city"], strict=False
        )
    ]

    dc_from = dc.read_json(path.as_uri(), session=test_session)
    df1 = dc_from.select("json.first_name", "json.age", "json.city").to_pandas()
    df1 = df1["json"]
    assert df_equal(df1, df)


# These deprecation warnings occur in the datamodel-code-generator package.
@pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20")
def test_read_json_jmespath(tmp_dir, test_session):
    df = pd.DataFrame(DF_DATA)
    values = [
        {"first_name": n, "age": a, "city": c}
        for n, a, c in zip(
            DF_DATA["first_name"], DF_DATA["age"], DF_DATA["city"], strict=False
        )
    ]
    path = tmp_dir / "test.json"
    with open(path, "w") as f:
        json.dump({"author": "Test User", "version": 5, "values": values}, f)

    dc_from = dc.read_json(path, jmespath="values", session=test_session)
    df1 = dc_from.select("values.first_name", "values.age", "values.city").to_pandas()
    df1 = df1["values"]
    assert df_equal(df1, df)


# These deprecation warnings occur in the datamodel-code-generator package.
@pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20")
def test_to_read_json_remote(cloud_test_catalog_upload):
    ctc = cloud_test_catalog_upload
    path = f"{ctc.src_uri}/test.json"

    df = pd.DataFrame(DF_DATA)
    dc_to = dc.read_pandas(df, session=ctc.session)
    dc_to.to_json(path)

    dc_from = dc.read_json(path, session=ctc.session)
    df1 = dc_from.select("json.first_name", "json.age", "json.city").to_pandas()
    df1 = df1["json"]
    assert df_equal(df1, df)


# These deprecation warnings occur in the datamodel-code-generator package.
@pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20")
def test_to_read_jsonl_remote(cloud_test_catalog_upload):
    ctc = cloud_test_catalog_upload
    path = f"{ctc.src_uri}/test.jsonl"

    df = pd.DataFrame(DF_DATA)
    dc_to = dc.read_pandas(df, session=ctc.session)
    dc_to.to_jsonl(path)

    dc_from = dc.read_json(path, format="jsonl", session=ctc.session)
    df1 = dc_from.select("jsonl.first_name", "jsonl.age", "jsonl.city").to_pandas()
    df1 = df1["jsonl"]
    assert df_equal(df1, df)


def test_read_pandas_multiindex(test_session):
    # Create a DataFrame with MultiIndex columns
    header = pd.MultiIndex.from_tuples(
        [("A", "cat"), ("B", "dog"), ("B", "cat"), ("A", "dog")]
    )
    data = [[1, 2, 3, 4], [5, 6, 7, 8]]
    df = pd.DataFrame(data, columns=header)

    # Read the DataFrame into a DataChain
    chain = dc.read_pandas(df, session=test_session)

    # Check the resulting column names and data
    expected_columns = ["a_cat", "b_dog", "b_cat", "a_dog"]
    assert set(chain.schema.keys()) == set(expected_columns)

    expected_data = [
        {"a_cat": 1, "b_dog": 2, "b_cat": 3, "a_dog": 4},
        {"a_cat": 5, "b_dog": 6, "b_cat": 7, "a_dog": 8},
    ]
    assert sorted_dicts(chain.to_records(), *expected_columns) == sorted_dicts(
        expected_data, *expected_columns
    )


def test_datachain_functional_after_exceptions(test_session):
    def func(key: str) -> str:
        raise Exception("Test Error!")

    keys = ["a", "b", "c"]
    values = [3, 1, 2]
    chain = dc.read_values(key=keys, val=values, session=test_session)
    # Running a few times, since sessions closing and cleaning up
    # DB connections on errors. We need to make sure that it reconnects
    # if needed.
    for _ in range(4):
        with pytest.raises(Exception, match="Test Error!"):
            chain.map(res=func).exec()


@pytest.mark.parametrize("parallel", [1, 2])
def test_agg(catalog_tmpfile, parallel):
    from datachain import func

    session = catalog_tmpfile.session

    def process(files: list[str]) -> Iterator[tuple[str, int]]:
        yield str(PurePosixPath(files[0]).parent), len(files)

    ds = (
        dc.read_values(
            filename=(
                "cats/cat1",
                "cats/cat2",
                "dogs/dog1",
                "dogs/dog2",
                "dogs/dog3",
                "dogs/others/dog4",
            ),
            session=session,
        )
        .settings(parallel=parallel)
        .agg(
            process,
            params=["filename"],
            output={"parent": str, "count": int},
            partition_by=func.path.parent("filename"),
        )
        .save("my-ds")
    )

    assert sorted_dicts(ds.to_records(), "parent") == sorted_dicts(
        [
            {"parent": "cats", "count": 2},
            {"parent": "dogs", "count": 3},
            {"parent": "dogs/others", "count": 1},
        ],
        "parent",
    )


@pytest.mark.parametrize("parallel", [1, 2])
@pytest.mark.parametrize(
    "offset,limit,files",
    [
        (None, 1000, [f"file{i:02d}" for i in range(100)]),
        (None, 3, ["file00", "file01", "file02"]),
        (0, 3, ["file00", "file01", "file02"]),
        (97, 1000, ["file97", "file98", "file99"]),
        (1, 2, ["file01", "file02"]),
        (50, 3, ["file50", "file51", "file52"]),
        (None, 0, []),
        (50, 0, []),
    ],
)
def test_agg_offset_limit(catalog_tmpfile, parallel, offset, limit, files):
    def process(filename: list[str]) -> Iterator[tuple[str, int]]:
        yield filename[0], len(filename)

    ds = dc.read_values(
        filename=[f"file{i:02d}" for i in range(100)],
        value=list(range(100)),
        session=catalog_tmpfile.session,
    )
    # Read values in general doesn't guarantee order, so we need to order first
    ds = ds.order_by("filename")
    if offset is not None:
        ds = ds.offset(offset)
    if limit is not None:
        ds = ds.limit(limit)

    limited_filenames = ds.to_values("filename")
    assert set(limited_filenames) == set(files)

    ds = (
        ds.settings(parallel=parallel)
        .agg(
            process,
            output={"filename": str, "count": int},
            partition_by="filename",
        )
        .save("my-ds")
    )

    records = list(ds.to_records())
    assert len(records) == len(files)
    assert all(row["count"] == 1 for row in records)
    assert sorted(row["filename"] for row in records) == sorted(files)


@pytest.mark.parametrize("parallel", [1, 2])
@pytest.mark.parametrize("sample", [0, 1, 3, 10, 50, 100])
def test_agg_sample(catalog_tmpfile, parallel, sample):
    def process(filename: list[str]) -> Iterator[tuple[str, int]]:
        yield filename[0], len(filename)

    ds = (
        dc.read_values(
            filename=[f"file{i:02d}" for i in range(100)],
            session=catalog_tmpfile.session,
        )
        .sample(sample)
        .settings(parallel=parallel)
        .agg(
            process,
            output={"filename": str, "count": int},
            partition_by="filename",
        )
        .save("my-ds")
    )

    records = list(ds.to_records())
    assert len(records) == sample
    assert all(row["count"] == 1 for row in records)
