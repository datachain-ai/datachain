import os
import uuid
from typing import Any

import pytest
import regex as re
from PIL import Image

import datachain as dc
from datachain import func
from datachain.error import DatasetNotFoundError, SchemaDriftError
from datachain.lib.dc import C
from datachain.lib.file import File, ImageFile


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

    starting_ds_name = "project.starting_ds"
    ds_name = "project.delta_ds"

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
        "img1.jpg",
        "img2.jpg",
        "img3.jpg",
        "img4.jpg",
    ]

    create_delta_dataset(ds_name)


def test_delta_falls_back_when_dependency_missing(test_session):
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

    deps_after_removal = catalog.get_dataset_dependencies(
        delta_ds,
        "1.0.0",
        namespace_name=catalog.metastore.default_project.namespace.name,
        project_name=catalog.metastore.default_project.name,
        indirect=False,
    )
    assert deps_after_removal == [None]

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


def test_delta_returns_correct_dataset_on_no_changes(test_session):
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


def test_delta_update_unsafe(test_session):
    catalog = test_session.catalog

    starting_ds_name = "starting_ds"
    merge_ds_name = "merge_ds"
    ds_name = "delta_ds"

    # create dataset which will be merged to delta one
    merge_ds = dc.read_values(
        id=[1, 2, 3, 4, 5, 6], value=[1, 2, 3, 4, 5, 6], session=test_session
    ).save(merge_ds_name)

    # first version of starting dataset
    dc.read_values(id=[1, 2, 3], session=test_session).save(starting_ds_name)
    # first version of delta dataset
    dc.read_dataset(
        starting_ds_name,
        session=test_session,
        delta_on="id",
        delta=True,
        delta_unsafe=True,
    ).merge(merge_ds, on="id", inner=True).save(ds_name)

    assert set(_get_dependencies(catalog, ds_name, "1.0.0")) == {
        (starting_ds_name, "1.0.0"),
        (merge_ds_name, "1.0.0"),
    }

    # second version of starting dataset
    dc.read_values(id=[1, 2, 3, 4, 5, 6], session=test_session).save(starting_ds_name)
    # second version of delta dataset
    dc.read_dataset(
        starting_ds_name,
        session=test_session,
        delta_on="id",
        delta=True,
        delta_unsafe=True,
    ).merge(merge_ds, on="id", inner=True).save(ds_name)

    assert set(_get_dependencies(catalog, ds_name, "1.0.1")) == {
        (starting_ds_name, "1.0.1"),
        (merge_ds_name, "1.0.0"),
    }

    assert set((dc.read_dataset(ds_name, version="1.0.0")).to_list("id", "value")) == {
        (1, 1),
        (2, 2),
        (3, 3),
    }

    assert set((dc.read_dataset(ds_name, version="1.0.1")).to_list("id", "value")) == {
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
    }


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


def test_delta_update_retry_union_detects_source_schema_change(test_session):
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
    assert "combining retry records with delta changes" in msg
    assert "columns missing in updated data: extra" in msg


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

    # remove first 5 images to check that deleted rows are not taken into consideration
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
        "images/img4.jpg",
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


def test_delta_update_check_num_calls(
    test_session, tmp_dir, tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", "True")
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
    # assert captured.out == "Garbage collecting 2 tables.\n"
    assert captured.out == "\n".join([map_print] * 20) + "\n"


def test_delta_update_no_diff(test_session, tmp_dir, tmp_path):
    catalog = test_session.catalog
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

    with pytest.raises(DatasetNotFoundError) as exc_info:
        dc.read_dataset(ds_name, version="1.0.1")

    assert str(exc_info.value) == (
        f"Dataset {ds_name} version 1.0.1 not found in namespace "
        f"{catalog.metastore.default_namespace_name}"
        f" and project {catalog.metastore.default_project_name}"
    )


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

    assert str(excinfo.value) == (
        "Cannot use union with delta datasets - may cause inconsistency."
        " Use delta_unsafe flag to allow this operation."
    )


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

    assert str(excinfo.value) == (
        "Cannot use merge with delta datasets - may cause inconsistency."
        " Use delta_unsafe flag to allow this operation."
    )


def test_delta_update_distinct(test_session, file_dataset):
    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                file_dataset.name,
                session=test_session,
                delta=True,
            ).distinct("file.path")
        )

    assert str(excinfo.value) == (
        "Cannot use distinct with delta datasets - may cause inconsistency."
        " Use delta_unsafe flag to allow this operation."
    )


def test_delta_update_group_by(test_session, file_dataset):
    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                file_dataset.name,
                session=test_session,
                delta=True,
            ).group_by(cnt=func.count(), partition_by="file.path")
        )

    assert str(excinfo.value) == (
        "Cannot use group_by with delta datasets - may cause inconsistency."
        " Use delta_unsafe flag to allow this operation."
    )


def test_delta_update_agg(test_session, file_dataset):
    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                file_dataset.name,
                session=test_session,
                delta=True,
            ).agg(cnt=func.count(), partition_by="file.path")
        )

    assert str(excinfo.value) == (
        "Cannot use agg with delta datasets - may cause inconsistency."
        " Use delta_unsafe flag to allow this operation."
    )
