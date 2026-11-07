import os
import uuid
from typing import Any

import pytest
import regex as re
from PIL import Image

import datachain as dc
from datachain import DataModel, func
from datachain.error import DatasetNotFoundError, SchemaDriftError
from datachain.lib.dc import C
from datachain.lib.file import File, ImageFile
from datachain.lib.utils import DataChainParamsError
from datachain.toolkit import train_test_split


class DeltaKey(DataModel):
    a: int
    b: str


class ReorderedDeltaKey(DataModel):
    b: str
    a: int


class IncompatibleDeltaKey(DataModel):
    a: int
    c: str


class NullableDeltaKey(DataModel):
    a: int | None = None


def test_delta_update_with_missing_and_empty_model_keys_matches_full_recompute(
    test_session,
):
    def source(keys, values):
        dc.read_values(
            key=keys,
            value=values,
            output={"key": NullableDeltaKey | None, "value": int},
            session=test_session,
        ).save("nullable_model_source")

    def build(delta):
        return dc.read_dataset(
            "nullable_model_source", delta=delta, delta_on="key", session=test_session
        ).map(score=lambda value: value * 2, output=int)

    source([None, NullableDeltaKey(), NullableDeltaKey(a=1)], [10, 20, 30])
    first = build(True).save("nullable_model_result")
    source([None, NullableDeltaKey()], [100, 20])
    result = build(True).save("nullable_model_result")
    assert sorted(result.to_list("key", "score"), key=repr) == sorted(
        build(False).to_list("key", "score"), key=repr
    )
    assert sorted(result.to_list("key", "score"), key=repr) == sorted(
        [(None, 200), (NullableDeltaKey(), 40)], key=repr
    )
    assert first.count() == 3


def test_delta_update_with_source_column_named_like_status_column_keeps_its_values(
    test_session,
):
    def build():
        return dc.read_dataset(
            "status_source", delta=True, delta_on="id", session=test_session
        ).map(value=lambda id: id * 10, output=int)

    dc.read_values(id=[1, 2], DC_DELTA_STATUS=["a", "b"], session=test_session).save(
        "status_source"
    )
    build().save("status_result")
    dc.read_values(
        id=[1, 2, 3], DC_DELTA_STATUS=["a", "x", "c"], session=test_session
    ).save("status_source")
    result = build().save("status_result")
    assert sorted(result.to_list("id", "DC_DELTA_STATUS", "value")) == [
        (1, "a", 10),
        (2, "x", 20),
        (3, "c", 30),
    ]


def _delta_unsafe_required_message(method_name: str) -> str:
    return (
        f"Cannot use {method_name} with delta datasets - may cause inconsistency."
        " Set delta_unsafe=True on every participating delta source"
        " to allow this operation."
    )


def _get_short_ds_name(catalog, name, project_name, namespace_name) -> str:
    if project_name == catalog.metastore.default_project.name:
        return name
    if namespace_name == catalog.metastore.default_project.namespace.name:
        return f"{project_name}.{name}"
    return f"{namespace_name}.{project_name}.{name}"


def _get_dependencies(catalog, name, version) -> list[tuple[str, str]]:
    namespace_name, project_name, name = catalog.get_full_dataset_name(name)
    return sorted(
        [
            (_get_short_ds_name(catalog, d.name, d.project, d.namespace), d.version)
            for d in catalog.get_dataset_dependencies(
                name,
                version,
                project_name=project_name,
                namespace_name=namespace_name,
                indirect=False,
            )
        ]
    )


def test_delta_update_from_dataset(test_session, tmp_dir, tmp_path):
    catalog = test_session.catalog
    starting_ds_name = "starting_ds"
    ds_name = "delta_ds"

    images = [
        {"name": "img1.jpg", "data": Image.new(mode="RGB", size=(64, 64))},
        {"name": "img2.jpg", "data": Image.new(mode="RGB", size=(128, 128))},
        {"name": "img3.jpg", "data": Image.new(mode="RGB", size=(64, 64))},
        {"name": "img4.jpg", "data": Image.new(mode="RGB", size=(128, 128))},
    ]

    def create_image_dataset(ds_name, images):
        dc.read_values(
            file=[
                ImageFile(path=img["name"], source=f"file://{tmp_path}")
                for img in images
            ],
            session=test_session,
        ).save(ds_name)

    def create_delta_dataset(ds_name):
        dc.read_dataset(
            starting_ds_name,
            session=test_session,
            delta=True,
            delta_on=["file.source", "file.path"],
            delta_result_on=["file.source", "file.path"],
            delta_compare=["file.version", "file.etag"],
        ).save(ds_name)

    # first version of starting dataset
    create_image_dataset(starting_ds_name, images[:2])
    # first version of delta dataset
    create_delta_dataset(ds_name)
    assert _get_dependencies(catalog, ds_name, "1.0.0") == [(starting_ds_name, "1.0.0")]
    # second version of starting dataset
    create_image_dataset(starting_ds_name, images[2:])
    # second version of delta dataset
    create_delta_dataset(ds_name)
    assert _get_dependencies(catalog, ds_name, "1.0.1") == [(starting_ds_name, "1.0.1")]

    assert (dc.read_dataset(ds_name, version="1.0.0").order_by("file.path")).to_values(
        "file.path"
    ) == [
        "img1.jpg",
        "img2.jpg",
    ]

    assert (dc.read_dataset(ds_name, version="1.0.1").order_by("file.path")).to_values(
        "file.path"
    ) == [
        "img3.jpg",
        "img4.jpg",
    ]

    create_delta_dataset(ds_name)


def test_delta_update_respects_update_version(test_session):
    source_name = f"delta_uv_source_{uuid.uuid4().hex[:8]}"
    result_name = f"delta_uv_result_{uuid.uuid4().hex[:8]}"

    dc.read_values(id=[1, 2, 3], session=test_session).save(source_name)
    dc.read_dataset(source_name, session=test_session, delta=True, delta_on="id").save(
        result_name
    )

    dc.read_values(id=[1, 2, 3, 4], session=test_session).save(source_name)
    res = dc.read_dataset(
        source_name, session=test_session, delta=True, delta_on="id"
    ).save(result_name, update_version="major", description="new desc", attrs=["NLP"])

    assert res.version == "2.0.0"
    dataset = test_session.catalog.get_dataset(result_name)
    assert dataset.description == "new desc"
    assert dataset.attrs == ["NLP"]


def test_delta_falls_back_when_dependency_missing(test_session, no_studio_dataset):
    catalog = test_session.catalog

    source_ds = "delta_removed_dep_source"
    delta_ds = "delta_removed_dep_result"
    process_log: list[int] = []

    def record_processing(id: int) -> int:
        process_log.append(id)
        return id

    # Create first source dataset and initial delta version that depends on it
    dc.read_values(id=[1, 2], session=test_session).save(source_ds)
    dc.read_dataset(
        source_ds,
        session=test_session,
        delta=True,
        delta_on="id",
    ).map(processed_id=record_processing).save(delta_ds)

    assert _get_dependencies(catalog, delta_ds, "1.0.0") == [(source_ds, "1.0.0")]
    assert set(
        dc.read_dataset(delta_ds, version="1.0.0", session=test_session).to_values("id")
    ) == {1, 2}
    assert sorted(process_log[:2]) == [1, 2]

    dc.read_values(id=[1, 2, 10, 20, 30], session=test_session).save(source_ds)

    # Drop the previous version so it is clear the dependency targets 1.0.1
    dc.delete_dataset(source_ds, version="1.0.0", session=test_session)

    with pytest.raises(DatasetNotFoundError):
        dc.read_dataset(source_ds, session=test_session, version="1.0.0")

    # The dep row still points at the now-REMOVED v1.0.0 of the source
    # so the dependent's lineage view still resolves.
    assert _get_dependencies(catalog, delta_ds, "1.0.0") == [(source_ds, "1.0.0")]

    dc.read_dataset(
        source_ds,
        session=test_session,
        delta=True,
        delta_on="id",
    ).map(processed_id=record_processing).save(delta_ds)

    # Delta logic should fall back to rebuilding from scratch with the new dependency
    assert _get_dependencies(catalog, delta_ds, "1.0.1") == [(source_ds, "1.0.1")]
    assert set(
        dc.read_dataset(delta_ds, version="1.0.1", session=test_session).to_values("id")
    ) == {1, 2, 10, 20, 30}
    # Previous version remains intact and still reflects the original source dataset
    assert set(
        dc.read_dataset(delta_ds, version="1.0.0", session=test_session).to_values("id")
    ) == {1, 2}
    # Fallback rebuilds the dataset, so ids 1 and 2 appear twice across both runs.
    assert sorted(process_log[:2]) == [1, 2]
    assert sorted(process_log[2:]) == [1, 2, 10, 20, 30]


def test_delta_falls_back_when_all_result_versions_tombstoned(
    test_session, no_studio_dataset
):
    source_ds = "delta_tombstoned_result_source"
    delta_ds = "delta_tombstoned_result_target"

    dc.read_values(id=[1, 2], session=test_session).save(source_ds)
    dc.read_dataset(
        source_ds,
        session=test_session,
        delta=True,
        delta_on="id",
    ).save(delta_ds)

    # Tombstone every version of the delta result. The dataset record
    # stays in the metastore but has no live versions.
    dc.delete_dataset(delta_ds, force=True, session=test_session)

    # Extend the source and re-run delta. Without the fallback catching
    # DatasetVersionNotFoundError, dataset.latest_version would raise
    # because every result version is a tombstone.
    dc.read_values(id=[1, 2, 3, 4], session=test_session).save(source_ds)
    dc.read_dataset(
        source_ds,
        session=test_session,
        delta=True,
        delta_on="id",
    ).save(delta_ds)

    # Fresh version reserved past the tombstoned semver.
    assert set(dc.read_dataset(delta_ds, session=test_session).to_values("id")) == {
        1,
        2,
        3,
        4,
    }


@pytest.mark.parametrize("is_studio", [True])
def test_delta_returns_correct_dataset_on_no_changes(
    test_session, is_studio, studio_job
):
    catalog = test_session.catalog

    default_project_name = catalog.metastore.default_project.name
    default_namespace_name = catalog.metastore.default_project.namespace.name

    base_short = "same_name_base"
    delta_short = "same_name_delta"

    cases = [
        {"ns": default_namespace_name, "proj": default_project_name, "ids": [1, 2]},
        {"ns": default_namespace_name, "proj": "project_other", "ids": [10, 20, 30]},
        {"ns": "namespace_other", "proj": "project_other", "ids": [100, 200]},
    ]

    # First pass: create starting and delta datasets (v1)
    for case in cases:
        ns = case["ns"]
        proj = case["proj"]
        ids = case["ids"]

        starting_ds_name = _get_short_ds_name(catalog, base_short, proj, ns)
        delta_ds_name = _get_short_ds_name(catalog, delta_short, proj, ns)

        dc.read_values(id=ids, session=test_session).save(starting_ds_name)

        dc.read_dataset(
            starting_ds_name,
            session=test_session,
            delta=True,
            delta_on="id",
            delta_compare="id",
        ).save(delta_ds_name)

        assert _get_dependencies(catalog, delta_ds_name, "1.0.0") == [
            (starting_ds_name, "1.0.0")
        ]
        assert set(
            dc.read_dataset(delta_ds_name, version="1.0.0").to_values("id")
        ) == set(ids)

    # Second pass: re-save with no changes, ensure it returns the existing version
    for case in cases:
        ns = case["ns"]
        proj = case["proj"]
        ids = case["ids"]

        starting_ds_name = _get_short_ds_name(catalog, base_short, proj, ns)
        delta_ds_name = _get_short_ds_name(catalog, delta_short, proj, ns)

        res = dc.read_dataset(
            starting_ds_name,
            session=test_session,
            delta=True,
            delta_on="id",
            delta_compare="id",
        ).save(delta_ds_name)

        # Should return the dataset with the same contents (no changes)
        assert res.dataset is not None
        assert set(dc.read_dataset(delta_ds_name).to_values("id")) == set(ids)

        # Still no newer version available
        with pytest.raises(DatasetNotFoundError):
            dc.read_dataset(delta_ds_name, version="1.0.1")


def test_delta_no_change_records_reused_output_access(test_session, monkeypatch):
    catalog = test_session.catalog
    metastore = catalog.metastore
    source_name = "delta_access_source"
    output_name = "delta_access_output"

    dc.read_values(id=[1, 2], session=test_session).save(source_name)
    dc.read_dataset(
        source_name,
        session=test_session,
        delta=True,
        delta_on="id",
        delta_compare="id",
    ).save(output_name)

    accessed = []
    monkeypatch.setattr(
        metastore,
        "record_dataset_version_access",
        lambda dataset, version: accessed.append((dataset.name, version)),
    )

    result = dc.read_dataset(
        source_name,
        session=test_session,
        delta=True,
        delta_on="id",
        delta_compare="id",
    ).save(output_name)

    assert result.dataset is not None
    assert result.dataset.latest_version == "1.0.0"
    assert (source_name, "1.0.0") in accessed
    assert (output_name, "1.0.0") in accessed


def test_delta_replay_regenerates_system_columns(test_session):
    source_name = f"regen_source_{uuid.uuid4().hex[:8]}"
    result_name = f"regen_result_{uuid.uuid4().hex[:8]}"

    dc.read_values(
        measurement_id=[1, 2],
        err=["", ""],
        num=[1, 2],
        session=test_session,
    ).save(source_name)

    def build_chain(delta: bool):
        read_kwargs = {"session": test_session}
        if delta:
            read_kwargs.update({"delta": True, "delta_on": "measurement_id"})
        return (
            dc.read_dataset(source_name, **read_kwargs)
            .filter(C.err == "")
            .select_except("err")
            .map(double=lambda num: num * 2, output=int)
            .select_except("num")
        )

    build_chain(delta=False).save(result_name)

    build_chain(delta=True).save(result_name)

    assert set(
        dc.read_dataset(result_name, session=test_session).to_values("measurement_id")
    ) == {1, 2}


@pytest.mark.parametrize(
    (
        "initial_has_confidence",
        "updated_has_confidence",
        "scenario",
        "expected_detail",
    ),
    [
        pytest.param(
            False,
            True,
            "added",
            "new columns detected: item.confidence",
            id="column_added",
        ),
        pytest.param(
            True,
            False,
            "removed",
            "columns missing in updated data: item.confidence",
            id="column_removed",
        ),
    ],
)
def test_delta_update_fails_on_schema_change(
    test_session,
    initial_has_confidence: bool,
    updated_has_confidence: bool,
    scenario: str,
    expected_detail: str,
):
    source_name = f"delta_schema_change_source_{scenario}_{uuid.uuid4().hex[:8]}"
    result_name = f"delta_schema_change_result_{scenario}_{uuid.uuid4().hex[:8]}"

    dc.read_values(id=[1, 2], session=test_session).save(source_name)

    def build_chain(include_confidence: bool):
        class ItemBase(dc.DataModel):
            label: str
            score: int
            error: str

        class ItemWithConfidence(ItemBase):
            confidence: float

        item_model = ItemWithConfidence if include_confidence else ItemBase

        def mark_error(id: int) -> str:
            return "retry" if id == 2 else ""

        def build_item(id: int) -> ItemBase:
            data: dict[str, Any] = {
                "label": f"item-{id}",
                "score": id * 10,
                "error": mark_error(id),
            }
            if include_confidence:
                data["confidence"] = float(id)
            return item_model(**data)

        return dc.read_dataset(
            source_name,
            session=test_session,
            delta=True,
            delta_on="id",
            delta_retry="item.error",
        ).map(item=build_item, output=item_model)

    build_chain(include_confidence=initial_has_confidence).save(result_name)

    dc.read_values(id=[1, 2, 3], session=test_session).save(source_name)

    with pytest.raises(SchemaDriftError) as excinfo:
        build_chain(include_confidence=updated_has_confidence).save(result_name)

    msg = str(excinfo.value)
    assert "schema drift" in msg.lower()
    assert "item.confidence" in msg
    assert expected_detail in msg


def test_delta_retry_update_after_source_schema_change_raises_schema_drift(
    test_session,
):
    source_name = f"delta_retry_drift_source_{uuid.uuid4().hex[:8]}"
    result_name = f"delta_retry_drift_result_{uuid.uuid4().hex[:8]}"

    dc.read_values(
        id=[1, 2],
        err=["retry", ""],
        session=test_session,
    ).save(source_name)

    dc.read_dataset(
        source_name,
        session=test_session,
        delta=True,
        delta_on="id",
        delta_retry="err",
    ).save(result_name)

    dc.read_values(
        id=[1, 2, 3],
        err=["retry", "", ""],
        extra=["a", "b", "c"],
        session=test_session,
    ).save(source_name)

    with pytest.raises(SchemaDriftError) as excinfo:
        dc.read_dataset(
            source_name,
            session=test_session,
            delta=True,
            delta_on="id",
            delta_retry="err",
        ).save(result_name)

    msg = str(excinfo.value)
    assert "merging the delta output with the existing dataset version" in msg
    assert "new columns detected: extra" in msg


def test_storage_delta_replay_regenerates_system_columns(test_session, tmp_dir):
    data_dir = tmp_dir / f"regen_storage_{uuid.uuid4().hex[:8]}"
    data_dir.mkdir()
    storage_uri = data_dir.as_uri()
    result_name = f"regen_storage_result_{uuid.uuid4().hex[:8]}"

    def write_payload(index: int) -> None:
        (data_dir / f"item{index}.txt").write_text(f"payload-{index}")

    write_payload(1)
    write_payload(2)

    def build_chain(delta: bool):
        read_kwargs = {"session": test_session, "update": True}
        if delta:
            read_kwargs |= {
                "delta": True,
                "delta_on": ["file.path"],
                "delta_result_on": ["file.path"],
            }

        def get_measurement_id(file: File) -> int:
            match = re.search(r"item(\d+)\.txt$", file.path)
            assert match
            return int(match.group(1))

        def get_num(file: File) -> int:
            return get_measurement_id(file)

        chain = dc.read_storage(storage_uri, **read_kwargs)
        return (
            chain.mutate(num=1)
            .select_except("num")
            .map(measurement_id=get_measurement_id)
            .map(err=lambda file: "")
            .map(num=get_num)
            .filter(C.err == "")
            .select_except("err")
            .map(double=lambda num: num * 2, output=int)
            .select_except("num")
        )

    build_chain(delta=False).save(result_name)

    write_payload(3)

    build_chain(delta=True).save(result_name)

    assert set(
        dc.read_dataset(result_name, session=test_session).to_values("measurement_id")
    ) == {1, 2, 3}


def test_delta_update_from_storage(test_session, tmp_dir, tmp_path):
    ds_name = "delta_ds"
    path = tmp_dir.as_uri()
    tmp_dir = tmp_dir / "images"
    os.mkdir(tmp_dir)

    images = [
        {
            "name": f"img{i}.{'jpg' if i % 2 == 0 else 'png'}",
            "data": Image.new(mode="RGB", size=((i + 1) * 10, (i + 1) * 10)),
        }
        for i in range(20)
    ]

    # save only half of the images for now
    for img in images[:10]:
        img["data"].save(tmp_dir / img["name"])

    def create_delta_dataset():
        def my_embedding(file: File) -> list[float]:
            return [0.5, 0.5]

        def get_index(file: File) -> int:
            r = r".+\/img(\d+)\.jpg"
            return int(re.search(r, file.path).group(1))  # type: ignore[union-attr]

        (
            dc.read_storage(
                path,
                update=True,
                session=test_session,
                delta=True,
                delta_on=["file.source", "file.path"],
                delta_result_on=["file.source", "file.path"],
                delta_compare=["file.version", "file.etag"],
            )
            .filter(C("file.path").glob("*.jpg"))
            .map(emb=my_embedding)
            .mutate(dist=func.cosine_distance("emb", (0.1, 0.2)))
            .map(index=get_index)
            .filter(C("index") > 3)
            .save(ds_name)
        )

    # first version of delta dataset
    create_delta_dataset()

    # remember old etags for later comparison to prove modified images are also taken
    # into consideration on delta update
    etags = {
        r[0]: r[1].etag
        for r in dc.read_dataset(ds_name, version="1.0.0").to_iter("index", "file")
    }

    # remove last couple of images to simulate modification since we will re-create it
    for img in images[5:10]:
        os.remove(tmp_dir / img["name"])

    # save other half of images and the ones that are removed above
    for img in images[5:]:
        img["data"].save(tmp_dir / img["name"])

    # Remove files whose result rows must disappear on refresh.
    for img in images[0:5]:
        os.remove(tmp_dir / img["name"])

    # second version of delta dataset
    create_delta_dataset()

    assert (dc.read_dataset(ds_name, version="1.0.0").order_by("file.path")).to_values(
        "file.path"
    ) == [
        "images/img4.jpg",
        "images/img6.jpg",
        "images/img8.jpg",
    ]

    assert (dc.read_dataset(ds_name, version="1.0.1").order_by("file.path")).to_values(
        "file.path"
    ) == [
        "images/img10.jpg",
        "images/img12.jpg",
        "images/img14.jpg",
        "images/img16.jpg",
        "images/img18.jpg",
        "images/img6.jpg",
        "images/img8.jpg",
    ]

    # check that we have newest versions for modified rows since etags are mtime
    # and modified rows etags should be bigger than the old ones
    assert (
        dc.read_dataset(ds_name, version="1.0.1")
        .filter(C("index") == 6)
        .order_by("file.path", "file.etag")
    ).to_values("file.etag")[0] > etags[6]


@pytest.mark.parametrize("explicit_key", [False, True])
def test_delta_update_from_storage_after_overwrite_delete_and_add(
    test_session, tmp_path, explicit_key
):
    data_dir = tmp_path / "refresh"
    data_dir.mkdir()
    for index in range(1, 5):
        (data_dir / f"f{index}.txt").write_text(str(index))
    processed = []

    def read(file: File) -> int:
        processed.append(file.path)
        return int(file.read_text())

    def generate(file: File):
        value = read(file)
        for index in range(value):
            yield file, index

    def build(delta):
        kwargs = (
            {"delta_on": "file.path", "delta_compare": ["file.etag", "file.version"]}
            if explicit_key
            else {}
        )
        chain = dc.read_storage(
            data_dir.as_uri(),
            update=True,
            delta=delta,
            session=test_session,
            **kwargs,
        )
        return chain.gen(generate, output={"file": File, "value": int}).filter(
            C.value < 4
        )

    def rows(chain):
        return sorted(
            [
                (file.model_dump(), value)
                for file, value in chain.to_iter("file", "value")
            ],
            key=lambda row: (row[0]["path"], row[1]),
        )

    initial = build(True).save("refresh_result")
    initial_rows = rows(initial)
    old_mtime = (data_dir / "f2.txt").stat().st_mtime
    (data_dir / "f2.txt").write_text("9")
    os.utime(data_dir / "f2.txt", (old_mtime + 10, old_mtime + 10))
    (data_dir / "f3.txt").unlink()
    old_mtime = (data_dir / "f4.txt").stat().st_mtime
    (data_dir / "f4.txt").write_text("00")
    os.utime(data_dir / "f4.txt", (old_mtime + 10, old_mtime + 10))
    (data_dir / "f5.txt").write_text("5")
    processed.clear()

    refreshed = build(True).save("refresh_result")
    assert sorted(processed) == ["f2.txt", "f4.txt", "f5.txt"]
    assert rows(refreshed) == rows(build(False))
    assert rows(initial) == initial_rows
    processed.clear()
    assert build(True).save("refresh_result").version == refreshed.version
    assert processed == []


@pytest.mark.parametrize("cloud_type", ["s3", "gs"], indirect=True)
@pytest.mark.parametrize("version_aware", [True], indirect=True)
def test_delta_update_from_versioned_storage_matches_full_recompute(
    cloud_test_catalog_upload,
):
    ctc = cloud_test_catalog_upload
    prefix = "delta-" + uuid.uuid4().hex[:8]
    uri = f"{ctc.src_uri}/{prefix}/"
    File.upload(b"2", uri + "a.txt", ctc.catalog)
    File.upload(b"3", uri + "b.txt", ctc.catalog)
    processed = []

    def read_number(file: File) -> int:
        processed.append(file.path)
        return int(file.read_text())

    def generate(file: File):
        value = read_number(file)
        for index in range(value):
            yield file, value, index

    def build(delta, retry=False):
        return (
            dc.read_storage(
                uri,
                update=True,
                delta=delta,
                delta_retry=retry,
                delta_on="file.path",
                delta_compare=["file.etag", "file.version"],
                versions=True,
                session=ctc.session,
            )
            .gen(generate, output={"file": File, "value": int, "index": int})
            .filter(C.value < 5)
        )

    def rows(chain):
        return sorted(chain.to_iter("file.path", "value", "index"))

    first = build(True).save("versioned_result")
    initial = rows(first)
    File.upload(b"9", uri + "a.txt", ctc.catalog)
    (ctc.src / prefix / "b.txt").unlink()
    File.upload(b"4", uri + "c.txt", ctc.catalog)
    processed.clear()
    result = build(True).save("versioned_result")
    assert sorted(processed) == [prefix + "/a.txt", prefix + "/c.txt"]
    assert (
        rows(result)
        == rows(build(False))
        == [(prefix + "/c.txt", 4, index) for index in range(4)]
    )
    assert rows(first) == initial
    processed.clear()
    assert build(True).save("versioned_result").version == result.version
    assert processed == []
    # Only the current filtered-out a.txt is eligible for retry, never its old
    # version or deleted b.txt. Retrying it must not restore outdated results.
    assert build(True, retry=True).save("versioned_result").version == result.version
    assert processed == [prefix + "/a.txt"]
    for value in (3, 1):
        File.upload(str(value).encode(), uri + "c.txt", ctc.catalog)
        processed.clear()
        updated = build(True).save("versioned_result")
        assert processed == [prefix + "/c.txt"]
        assert (
            rows(updated)
            == rows(build(False))
            == [(prefix + "/c.txt", value, index) for index in range(value)]
        )
    processed.clear()
    (ctc.src / prefix / "c.txt").unlink()
    emptied = build(True).save("versioned_result")
    assert emptied.count() == 0
    assert processed == []
    assert build(True).save("versioned_result").version == emptied.version


def test_delta_update_check_num_calls(
    test_session, tmp_dir, tmp_path, capsys, monkeypatch
):
    ds_name = "delta_ds"
    path = tmp_dir.as_uri()
    tmp_dir = tmp_dir / "images"
    os.mkdir(tmp_dir)
    map_print = "In map"

    images = [
        {
            "name": f"img{i}.jpg",
            "data": Image.new(mode="RGB", size=((i + 1) * 10, (i + 1) * 10)),
        }
        for i in range(20)
    ]

    # save only half of the images for now
    for img in images[:10]:
        img["data"].save(tmp_dir / img["name"])

    def create_delta_dataset():
        def get_index(file: File) -> int:
            print(map_print)  # needed to count number of map calls
            r = r".+\/img(\d+)\.jpg"
            return int(re.search(r, file.path).group(1))  # type: ignore[union-attr]

        (
            dc.read_storage(
                path,
                update=True,
                session=test_session,
                delta=True,
                delta_on=["file.source", "file.path"],
                delta_result_on=["file.source", "file.path"],
                delta_compare=["file.version", "file.etag"],
            )
            .map(index=get_index)
            .save(ds_name)
        )

    # first version of delta dataset
    create_delta_dataset()
    # save other half of images
    for img in images[10:]:
        img["data"].save(tmp_dir / img["name"])
    # second version of delta dataset
    create_delta_dataset()

    captured = capsys.readouterr()
    assert captured.out == "\n".join([map_print] * 20) + "\n"


def test_delta_update_no_diff(test_session, tmp_dir, tmp_path, no_studio_dataset):
    ds_name = "delta_ds"
    path = tmp_dir.as_uri()
    tmp_dir = tmp_dir / "images"
    os.mkdir(tmp_dir)

    images = [
        {"name": f"img{i}.jpg", "data": Image.new(mode="RGB", size=(64, 128))}
        for i in range(10)
    ]

    for img in images:
        img["data"].save(tmp_dir / img["name"])

    def create_delta_dataset():
        def get_index(file: File) -> int:
            r = r".+\/img(\d+)\.jpg"
            return int(re.search(r, file.path).group(1))  # type: ignore[union-attr]

        (
            dc.read_storage(
                path,
                update=True,
                session=test_session,
                delta=True,
                delta_on=["file.source", "file.path"],
                delta_compare=["file.version", "file.etag"],
            )
            .filter(C("file.path").glob("*.jpg"))
            .map(index=get_index)
            .filter(C("index") > 5)
            .save(ds_name)
        )

    create_delta_dataset()
    create_delta_dataset()

    assert (dc.read_dataset(ds_name, version="1.0.0").order_by("file.path")).to_values(
        "file.path"
    ) == [
        "images/img6.jpg",
        "images/img7.jpg",
        "images/img8.jpg",
        "images/img9.jpg",
    ]

    with pytest.raises(DatasetNotFoundError):
        dc.read_dataset(ds_name, version="1.0.1")


def test_delta_update_after_file_added_outside_glob_reuses_version(
    test_session, tmp_path
):
    (tmp_path / "a.txt").write_text("1")
    processed = []

    def record(file: File) -> int:
        processed.append(file.path)
        return int(file.read_text())

    def build():
        return dc.read_storage(
            str(tmp_path / "*.txt"),
            update=True,
            delta=True,
            session=test_session,
        ).map(value=record)

    first = build().save("glob_result")
    processed.clear()
    (tmp_path / "ignored.log").write_text("2")
    result = build().save("glob_result")
    assert result.version == first.version
    assert result.to_values("value") == [1]
    assert processed == []


@pytest.fixture
def file_dataset(test_session):
    return dc.read_values(
        file=[
            File(path="a.jpg", source="s3://bucket"),
            File(path="b.jpg", source="s3://bucket"),
        ],
        session=test_session,
    ).save("file_ds")


def test_delta_update_union(test_session, file_dataset):
    dc.read_values(num=[10, 20], session=test_session).save("numbers")

    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                file_dataset.name,
                session=test_session,
                delta=True,
            ).union(dc.read_dataset("numbers"), session=test_session)
        )

    assert str(excinfo.value) == _delta_unsafe_required_message("union")


def test_delta_update_merge(test_session, file_dataset):
    dc.read_values(num=[10, 20], session=test_session).save("numbers")

    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                file_dataset.name,
                session=test_session,
                delta=True,
            ).merge(dc.read_dataset("numbers"), on="id", session=test_session)
        )

    assert str(excinfo.value) == _delta_unsafe_required_message("merge")


def test_delta_update_distinct(test_session, file_dataset):
    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                file_dataset.name,
                session=test_session,
                delta=True,
            ).distinct("file.path")
        )

    assert str(excinfo.value) == _delta_unsafe_required_message("distinct")


def test_delta_update_group_by(test_session, file_dataset):
    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                file_dataset.name,
                session=test_session,
                delta=True,
            ).group_by(cnt=func.count(), partition_by="file.path")
        )

    assert str(excinfo.value) == _delta_unsafe_required_message("group_by")


def test_delta_update_agg(test_session, file_dataset):
    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                file_dataset.name,
                session=test_session,
                delta=True,
            ).agg(cnt=func.count(), partition_by="file.path")
        )

    assert str(excinfo.value) == _delta_unsafe_required_message("agg")


@pytest.mark.parametrize("change", ["delete", "filter", "zero"])
def test_delta_update_after_record_loses_its_output_removes_old_rows(
    test_session, change
):
    processed = []

    def generate(id, value):
        processed.append(id)
        for index in range(value):
            yield id, index

    def build():
        return (
            dc.read_dataset(
                "empty_source", delta=True, delta_on="id", session=test_session
            )
            .filter(C.value >= 0)
            .gen(generate, output={"id": int, "index": int})
        )

    dc.read_values(id=[1, 2], value=[2, 2], session=test_session).save("empty_source")
    build().save("empty_result")
    dc.read_values(
        id=[2] if change == "delete" else [1, 2],
        value=[2] if change == "delete" else [-1 if change == "filter" else 0, 2],
        session=test_session,
    ).save("empty_source")
    processed.clear()
    result = build().save("empty_result")
    assert processed == ([1] if change == "zero" else [])
    assert sorted(result.to_iter("id", "index")) == [(2, 0), (2, 1)]
    assert build().save("empty_result").version == result.version


def test_train_test_split_after_delta_update_is_disjoint_and_complete(test_session):
    def split():
        chain = dc.read_dataset(
            "split_source", delta=True, delta_on="id", session=test_session
        ).map(value=lambda id: id * 2, output=int)
        train, test = train_test_split(chain, [0.7, 0.3], seed=0)
        return (
            set(train.save("train").to_values("id")),
            set(test.save("test").to_values("id")),
        )

    dc.read_values(id=list(range(10)), session=test_session).save("split_source")
    split()
    dc.read_values(id=list(range(200)), session=test_session).save("split_source")
    train, test = split()
    assert not train & test
    assert train | test == set(range(200))


@pytest.mark.parametrize(
    "source_key,result_key,rename",
    [
        (DeltaKey, DeltaKey, False),
        (DeltaKey | None, DeltaKey | None, False),
        (DeltaKey | None, ReorderedDeltaKey, True),
    ],
)
def test_delta_update_with_model_key_matches_fields_by_name(
    test_session, source_key, result_key, rename
):
    result_field = "result_key" if rename else "key"

    def source(keys, values):
        dc.read_values(
            key=keys,
            value=values,
            output={"key": source_key, "value": int},
            session=test_session,
        ).save("model_source")

    def generate(key, value):
        model = ReorderedDeltaKey if result_key is ReorderedDeltaKey else DeltaKey
        yield model(a=key.a, b=key.b), value

    def build(delta):
        return dc.read_dataset(
            "model_source",
            delta=delta,
            delta_on="key",
            delta_result_on=result_field,
            session=test_session,
        ).gen(generate, output={result_field: result_key, "value": int})

    keys = [DeltaKey(a=1, b="x"), DeltaKey(a=2, b="y"), DeltaKey(a=3, b="z")]
    source(keys, [10, 20, 30])
    build(True).save("model_result")
    source(keys[:2], [10, 50])
    result = build(True).save("model_result")

    def rows(chain):
        return sorted(
            (key.a, key.b, value) for key, value in chain.to_iter(result_field, "value")
        )

    assert rows(result) == rows(build(False)) == [(1, "x", 10), (2, "y", 50)]


def test_delta_update_with_model_key_of_different_fields_raises(test_session):
    def generate(key, value):
        yield IncompatibleDeltaKey(a=key.a, c=key.b), value

    def save():
        return (
            dc.read_dataset(
                "incompatible_source", delta=True, delta_on="key", session=test_session
            )
            .gen(generate, output={"key": IncompatibleDeltaKey, "value": int})
            .save("incompatible_result")
        )

    dc.read_values(key=[DeltaKey(a=1, b="x")], value=[10], session=test_session).save(
        "incompatible_source"
    )
    with pytest.raises(DataChainParamsError, match="same shape on both sides"):
        save()
    with pytest.raises(DatasetNotFoundError):
        dc.read_dataset("incompatible_result", session=test_session)


def test_delta_save_without_result_key_explains_required_fields(test_session):
    dc.read_values(id=[1], session=test_session).save("missing_key_source")
    chain = (
        dc.read_dataset(
            "missing_key_source", delta=True, delta_on="id", session=test_session
        )
        .map(score=lambda id: id * 2, output=int)
        .select("score")
    )
    with pytest.raises(DataChainParamsError, match=r"delta_on.*delta_result_on"):
        chain.save("missing_key_result")


@pytest.mark.parametrize("operation", ["mutate", "gen"])
def test_delta_rejects_chain_that_changes_the_key(test_session, operation):
    def generate(id, value):
        yield id * 10, value

    def build(delta_unsafe):
        chain = dc.read_dataset(
            "rewrite_source",
            delta=True,
            delta_on="id",
            delta_unsafe=delta_unsafe,
            session=test_session,
        )
        if operation == "mutate":
            return chain.mutate(id=C.id + 10)
        return chain.gen(generate, output={"id": int, "value": int})

    dc.read_values(id=[1, 2], value=[10, 20], session=test_session).save(
        "rewrite_source"
    )
    with pytest.raises(DataChainParamsError, match="change delta key 'id'"):
        build(delta_unsafe=False).save("rewrite_result")
    result = build(delta_unsafe=True).save("rewrite_result")
    assert sorted(result.to_values("id")) == (
        [11, 12] if operation == "mutate" else [10, 20]
    )
