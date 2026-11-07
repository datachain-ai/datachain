import uuid

import pytest
import regex as re

import datachain as dc
from datachain import C
from datachain.error import DatasetNotFoundError
from datachain.lib.file import File

from .test_delta import _delta_unsafe_required_message, _get_dependencies


@pytest.mark.parametrize("operation", ["merge", "subtract"])
def test_delta_update_unsafe_with_static_input_replays_changed_records(
    test_session, operation
):
    processed = []

    def score(id) -> int:
        processed.append(id)
        return id * 10

    def source(ids, groups):
        dc.read_values(id=ids, group=groups, session=test_session).save("enriched")

    def build(delta):
        chain = dc.read_dataset(
            "enriched",
            delta=delta,
            delta_on="id",
            delta_unsafe=True,
            session=test_session,
        ).map(score=score)
        labels = dc.read_dataset("labels", session=test_session)
        if operation == "subtract":
            return chain.subtract(labels, on="group")
        return chain.merge(labels, on="group")

    dc.read_values(group=[1, 2], label=["a", "b"], session=test_session).save("labels")
    source([1, 2, 3], [1, 2, 3])
    first = build(True).save("enriched_result")
    before = sorted(first.to_list(), key=repr)
    source([1, 3, 4], [1, 2, 3])
    processed.clear()
    result = build(True).save("enriched_result")
    assert sorted(processed) == [3, 4]
    assert sorted(result.to_list(), key=repr) == sorted(
        build(False).to_list(), key=repr
    )
    assert sorted(first.to_list(), key=repr) == before
    processed.clear()
    assert build(True).save("enriched_result").version == result.version
    assert processed == []


@pytest.mark.parametrize("reuse", [False, True])
def test_delta_update_unsafe_subtract_filtered_same_source_matches_full_recompute(
    test_session, reuse, mocker
):
    processed = []

    def record(id) -> int:
        processed.append(id)
        return id * 10

    def source(ids, groups, blocked):
        dc.read_values(
            id=ids, group=groups, blocked=blocked, session=test_session
        ).save("self_subtract_source")

    def build(delta):
        chain = dc.read_dataset(
            "self_subtract_source",
            delta=delta,
            delta_on="id",
            delta_unsafe=True,
            session=test_session,
        )
        chain = chain.subtract(chain.filter(C.blocked == 1), on="group")
        if reuse:
            chain = chain.filter(C.id < 3).union(chain.filter(C.id >= 3))
        return chain.map(value=record)

    source([1, 2, 3], [10, 10, 20], [0, 1, 0])
    initial = build(True).save("self_subtract_result")
    assert initial.to_values("id") == [3]
    source([1, 2, 3, 4], [10, 10, 20, 30], [0, 1, 0, 0])
    processed.clear()
    persist = mocker.spy(dc.DataChain, "_persist")
    result = build(True).save("self_subtract_result")
    assert (
        sum(call.kwargs.get("preserve_hash", False) for call in persist.call_args_list)
        == 1
    )
    assert processed == [4]
    assert (
        sorted(result.to_values("id")) == sorted(build(False).to_values("id")) == [3, 4]
    )
    assert initial.to_values("id") == [3]


@pytest.mark.parametrize("changed", ["left", "right"])
def test_delta_update_unsafe_union_after_append_to_one_source_processes_only_new_rows(
    test_session, changed
):
    processed = []

    def record(id) -> int:
        processed.append(id)
        return id * 10

    def source(name, ids):
        dc.read_values(**{f"{name[0]}_id": ids}, session=test_session).save(name)

    def build(delta):
        def read(name):
            key = f"{name[0]}_id"
            chain = dc.read_dataset(
                name,
                delta=delta,
                delta_on=key,
                delta_result_on="id",
                delta_unsafe=True,
                session=test_session,
            )
            return chain.select(id=C(key))

        return read("left").union(read("right")).map(value=record)

    source("left", [1, 2])
    source("right", [10, 20])
    first = build(True).save("independent_result")
    if changed == "left":
        source("left", [1, 2, 3])
    if changed == "right":
        source("right", [10, 20, 30])
    processed.clear()
    result = build(True).save("independent_result")
    assert sorted(processed) == ([3] if changed == "left" else [30])
    assert sorted(result.to_iter("id", "value")) == sorted(
        build(False).to_iter("id", "value")
    )
    assert sorted(first.to_values("id")) == [1, 2, 10, 20]
    assert build(True).save("independent_result").version == result.version


def test_delta_update_unsafe_same_source_with_different_compare_fields(test_session):
    processed = []

    def record(id) -> int:
        processed.append(id)
        return id

    def build():
        def read(compare):
            return dc.read_dataset(
                "compare_source",
                delta=True,
                delta_on="id",
                delta_compare=compare,
                delta_unsafe=True,
                session=test_session,
            )

        return read("value").union(read("blocked")).map(score=record)

    dc.read_values(id=[1], value=[10], blocked=[0], session=test_session).save(
        "compare_source"
    )
    build().save("compare_result")
    dc.read_values(
        id=[1, 2], value=[20, 30], blocked=[0, 0], session=test_session
    ).save("compare_source")
    processed.clear()
    result = build().save("compare_result")
    assert sorted(processed) == [1, 2, 2]
    assert sorted(result.to_list("id", "value")) == [(1, 20), (2, 30), (2, 30)]


def test_delta_update_unsafe_union_with_static_input_keeps_static_rows_once(
    test_session,
):
    processed = []

    def record(id) -> int:
        processed.append(id)
        return id * 10

    def build(delta):
        changing = dc.read_dataset(
            "changing",
            delta=delta,
            delta_on="id",
            delta_unsafe=True,
            session=test_session,
        ).map(value=record)
        static = dc.read_dataset("static", session=test_session)
        return changing.union(static)

    dc.read_values(id=[1, 2], session=test_session).save("changing")
    dc.read_values(id=[9], value=[90], session=test_session).save("static")
    first = build(True).save("static_result")
    dc.read_values(id=[1, 2, 3], session=test_session).save("changing")
    processed.clear()
    result = build(True).save("static_result")
    assert processed == [3]
    assert sorted(result.to_iter("id", "value")) == [(1, 10), (2, 20), (3, 30), (9, 90)]
    assert sorted(first.to_values("id")) == [1, 2, 9]


def test_delta_update_unsafe_after_only_lookup_changes_reuses_version(test_session):
    def build(delta):
        return dc.read_dataset(
            "changing",
            delta=delta,
            delta_on="id",
            delta_unsafe=True,
            session=test_session,
        ).merge(dc.read_dataset("labels", session=test_session), on="id")

    dc.read_values(id=[1, 2], session=test_session).save("changing")
    dc.read_values(id=[1, 2], label=["a", "b"], session=test_session).save("labels")
    first = build(True).save("lookup_result")
    dc.read_values(id=[1, 2], label=["a", "updated"], session=test_session).save(
        "labels"
    )
    unchanged = build(True).save("lookup_result")
    assert unchanged.version == first.version
    assert sorted(unchanged.to_iter("id", "label")) == [(1, "a"), (2, "b")]
    rebuilt = build(False).save("lookup_result")
    assert sorted(rebuilt.to_iter("id", "label")) == [(1, "a"), (2, "updated")]


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


def test_delta_update_unsafe_no_changes_reuses_previous_version(
    test_session, no_studio_dataset
):
    source_name = f"unsafe_same_source_{uuid.uuid4().hex[:8]}"
    merge_name = f"unsafe_same_merge_{uuid.uuid4().hex[:8]}"
    result_name = f"unsafe_same_result_{uuid.uuid4().hex[:8]}"

    dc.read_values(id=[1, 2, 3], value=[1, 2, 3], session=test_session).save(merge_name)
    dc.read_values(id=[1, 2, 3], session=test_session).save(source_name)

    dc.read_dataset(
        source_name,
        session=test_session,
        delta=True,
        delta_on="id",
        delta_unsafe=True,
    ).merge(
        dc.read_dataset(merge_name, session=test_session),
        on="id",
        inner=True,
    ).save(result_name)

    res = (
        dc.read_dataset(
            source_name,
            session=test_session,
            delta=True,
            delta_on="id",
            delta_unsafe=True,
        )
        .merge(
            dc.read_dataset(merge_name, session=test_session),
            on="id",
            inner=True,
        )
        .save(result_name)
    )

    assert res.version == "1.0.0"
    with pytest.raises(DatasetNotFoundError):
        dc.read_dataset(result_name, version="1.0.1", session=test_session)


def test_delta_update_unsafe_matches_same_named_aligned_sources_by_full_identity(
    test_session,
):
    catalog = test_session.catalog
    suffix = uuid.uuid4().hex[:8]
    source_short = f"shared_source_{suffix}"
    left_name = f"namespace_a_{suffix}.project_a_{suffix}.{source_short}"
    right_name = f"namespace_b_{suffix}.project_b_{suffix}.{source_short}"
    result_name = f"same_name_delta_result_{suffix}"
    processed: list[int] = []

    catalog.metastore.create_project(f"namespace_a_{suffix}", f"project_a_{suffix}")
    catalog.metastore.create_project(f"namespace_b_{suffix}", f"project_b_{suffix}")

    def record(id: int, left_value: int, right_value: int) -> int:
        processed.append(id)
        return left_value + right_value

    dc.read_values(
        id=[1],
        left_value=[10],
        session=test_session,
    ).save(left_name)
    dc.read_values(
        id=[1],
        right_value=[100],
        session=test_session,
    ).save(right_name)

    def build_chain():
        return (
            dc.read_dataset(
                left_name,
                session=test_session,
                delta=True,
                delta_on="id",
                delta_unsafe=True,
            )
            .merge(
                dc.read_dataset(
                    right_name,
                    session=test_session,
                    delta=True,
                    delta_on="id",
                    delta_unsafe=True,
                ),
                on="id",
                inner=True,
            )
            .map(
                joined_value=record,
                params=["id", "left_value", "right_value"],
                output=int,
            )
        )

    build_chain().save(result_name)

    assert processed == [1]
    assert _get_dependencies(catalog, result_name, "1.0.0") == [
        (left_name, "1.0.0"),
        (right_name, "1.0.0"),
    ]
    assert set(dc.read_dataset(result_name, session=test_session).to_values("id")) == {
        1
    }

    dc.read_values(
        id=[1, 2],
        left_value=[10, 20],
        session=test_session,
    ).save(left_name)
    dc.read_values(
        id=[1, 2],
        right_value=[100, 200],
        session=test_session,
    ).save(right_name)

    build_chain().save(result_name)

    assert processed[1:] == [2]
    assert _get_dependencies(catalog, result_name, "1.0.1") == [
        (left_name, "1.0.1"),
        (right_name, "1.0.1"),
    ]
    assert set(dc.read_dataset(result_name, session=test_session).to_values("id")) == {
        1,
        2,
    }


def test_delta_update_unsafe_after_self_join_on_non_key_processes_new_row_once(
    test_session,
):
    source_name = f"unsafe_same_occ_source_{uuid.uuid4().hex[:8]}"
    result_name = f"unsafe_same_occ_result_{uuid.uuid4().hex[:8]}"
    processed: list[int] = []

    def record(row_id: int) -> int:
        processed.append(row_id)
        return row_id

    dc.read_values(
        row_id=[1, 10],
        group=[1, 1],
        side=["L", "R"],
        session=test_session,
    ).save(source_name)

    def build_chain():
        left = dc.read_dataset(
            source_name,
            session=test_session,
            delta=True,
            delta_on="row_id",
            delta_unsafe=True,
        ).filter(C("side") == "L")
        right = dc.read_dataset(
            source_name,
            session=test_session,
            delta=True,
            delta_on="row_id",
            delta_unsafe=True,
        ).filter(C("side") == "R")
        return left.merge(right, on="group", inner=True).map(
            seen_id=record,
            params=["row_id"],
            output=int,
        )

    build_chain().save(result_name)

    assert processed == [1]
    assert dc.read_dataset(result_name, session=test_session).to_values("row_id") == [1]

    dc.read_values(
        row_id=[1, 2, 10, 20],
        group=[1, 1, 1, 1],
        side=["L", "L", "R", "R"],
        session=test_session,
    ).save(source_name)

    build_chain().save(result_name)

    assert processed[1:] == [2]
    assert sorted(
        dc.read_dataset(result_name, session=test_session).to_iter(
            "row_id", "right_row_id"
        )
    ) == [(1, 10), (2, 20)]


def test_delta_update_unsafe_merge_with_two_storage_sources(test_session, tmp_dir):
    left_dir = tmp_dir / f"unsafe_storage_left_{uuid.uuid4().hex[:8]}"
    right_dir = tmp_dir / f"unsafe_storage_right_{uuid.uuid4().hex[:8]}"
    left_dir.mkdir()
    right_dir.mkdir()

    result_name = f"unsafe_storage_merge_result_{uuid.uuid4().hex[:8]}"
    processed: list[int] = []

    def record(row_id: int) -> int:
        processed.append(row_id)
        return row_id

    def file_id(file: File) -> int:
        match = re.search(r"item(\d+)\.txt$", file.path)
        assert match
        return int(match.group(1))

    (left_dir / "item1.txt").write_text("left-1")
    (right_dir / "item1.txt").write_text("right-1")

    def build_chain():
        left = dc.read_storage(
            left_dir.as_uri(),
            session=test_session,
            update=True,
            delta=True,
            delta_on=["file.path"],
            delta_result_on=["id"],
            delta_unsafe=True,
        ).map(id=file_id, output=int)
        right = dc.read_storage(
            right_dir.as_uri(),
            session=test_session,
            update=True,
            delta=True,
            delta_on=["file.path"],
            delta_result_on=["id"],
            delta_unsafe=True,
        ).map(id=file_id, output=int)
        return left.merge(right, on="id", inner=True).map(
            seen_id=record,
            params=["id"],
            output=int,
        )

    build_chain().save(result_name)

    assert processed == [1]
    assert dc.read_dataset(result_name, session=test_session).to_values("id") == [1]

    (left_dir / "item2.txt").write_text("left-2")
    (right_dir / "item2.txt").write_text("right-2")

    build_chain().save(result_name)

    assert processed[1:] == [2]
    assert set(dc.read_dataset(result_name, session=test_session).to_values("id")) == {
        1,
        2,
    }


def test_delta_update_merge_requires_all_delta_inputs_unsafe(test_session):
    left_name = f"mixed_merge_left_{uuid.uuid4().hex[:8]}"
    right_name = f"mixed_merge_right_{uuid.uuid4().hex[:8]}"

    dc.read_values(id=[1, 2], left_value=[10, 20], session=test_session).save(left_name)
    dc.read_values(id=[1, 2], right_value=[100, 200], session=test_session).save(
        right_name
    )

    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                left_name,
                session=test_session,
                delta=True,
                delta_on="id",
                delta_unsafe=True,
            ).merge(
                dc.read_dataset(
                    right_name,
                    session=test_session,
                    delta=True,
                    delta_on="id",
                ),
                on="id",
                inner=True,
            )
        )

    assert str(excinfo.value) == _delta_unsafe_required_message("merge")


def test_delta_update_subtract_requires_all_delta_inputs_unsafe(test_session):
    left_name = f"mixed_subtract_left_{uuid.uuid4().hex[:8]}"
    right_name = f"mixed_subtract_right_{uuid.uuid4().hex[:8]}"

    dc.read_values(id=[1, 2], left_value=[10, 20], session=test_session).save(left_name)
    dc.read_values(id=[1, 2], right_value=[100, 200], session=test_session).save(
        right_name
    )

    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                left_name,
                session=test_session,
                delta=True,
                delta_on="id",
                delta_unsafe=True,
            ).subtract(
                dc.read_dataset(
                    right_name,
                    session=test_session,
                    delta=True,
                    delta_on="id",
                ),
                on="id",
            )
        )

    assert str(excinfo.value) == _delta_unsafe_required_message("subtract")

    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                left_name,
                session=test_session,
                delta=True,
                delta_on="id",
            ).subtract(
                dc.read_dataset(
                    right_name,
                    session=test_session,
                    delta=True,
                    delta_on="id",
                    delta_unsafe=True,
                ),
                on="id",
            )
        )

    assert str(excinfo.value) == _delta_unsafe_required_message("subtract")


def test_delta_update_diff_requires_all_delta_inputs_unsafe(test_session):
    left_name = f"mixed_diff_left_{uuid.uuid4().hex[:8]}"
    right_name = f"mixed_diff_right_{uuid.uuid4().hex[:8]}"

    dc.read_values(id=[1, 2], left_value=[10, 20], session=test_session).save(left_name)
    dc.read_values(id=[1, 2], right_value=[100, 200], session=test_session).save(
        right_name
    )

    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                left_name,
                session=test_session,
                delta=True,
                delta_on="id",
                delta_unsafe=True,
            ).diff(
                dc.read_dataset(
                    right_name,
                    session=test_session,
                    delta=True,
                    delta_on="id",
                ),
                on="id",
            )
        )

    assert str(excinfo.value) == _delta_unsafe_required_message("diff")


def test_delta_update_file_diff_requires_all_delta_inputs_unsafe(test_session):
    left_name = f"mixed_file_diff_left_{uuid.uuid4().hex[:8]}"
    right_name = f"mixed_file_diff_right_{uuid.uuid4().hex[:8]}"

    dc.read_values(
        file=[File(path="a.jpg", source="s3://left-bucket")],
        session=test_session,
    ).save(left_name)
    dc.read_values(
        file=[File(path="a.jpg", source="s3://right-bucket")],
        session=test_session,
    ).save(right_name)

    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                left_name,
                session=test_session,
                delta=True,
                delta_on=["file.source", "file.path"],
                delta_unsafe=True,
            ).file_diff(
                dc.read_dataset(
                    right_name,
                    session=test_session,
                    delta=True,
                    delta_on=["file.source", "file.path"],
                )
            )
        )

    assert str(excinfo.value) == _delta_unsafe_required_message("file_diff")

    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                left_name,
                session=test_session,
                delta=True,
                delta_on=["file.source", "file.path"],
            ).file_diff(
                dc.read_dataset(
                    right_name,
                    session=test_session,
                    delta=True,
                    delta_on=["file.source", "file.path"],
                    delta_unsafe=True,
                )
            )
        )

    assert str(excinfo.value) == _delta_unsafe_required_message("file_diff")

    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                left_name,
                session=test_session,
                delta=True,
                delta_on="id",
            ).diff(
                dc.read_dataset(
                    right_name,
                    session=test_session,
                    delta=True,
                    delta_on="id",
                    delta_unsafe=True,
                ),
                on="id",
            )
        )

    assert str(excinfo.value) == _delta_unsafe_required_message("diff")


def test_delta_update_union_requires_all_delta_inputs_unsafe(test_session):
    left_name = f"mixed_union_left_{uuid.uuid4().hex[:8]}"
    right_name = f"mixed_union_right_{uuid.uuid4().hex[:8]}"

    dc.read_values(id=[1, 2], session=test_session).save(left_name)
    dc.read_values(id=[3, 4], session=test_session).save(right_name)

    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                left_name,
                session=test_session,
                delta=True,
                delta_on="id",
                delta_unsafe=True,
            ).union(
                dc.read_dataset(
                    right_name,
                    session=test_session,
                    delta=True,
                    delta_on="id",
                )
            )
        )

    assert str(excinfo.value) == _delta_unsafe_required_message("union")


@pytest.mark.parametrize("operation", ["merge", "union"])
def test_delta_update_unsafe_after_same_changes_in_two_sources_matches_full_recompute(
    test_session, operation
):
    processed = []

    def record(id, value):
        processed.append(id)
        return value * 2

    def save_sources(ids, values):
        for name, scale in [("multi_left", 1), ("multi_right", 10)]:
            dc.read_values(
                id=ids, value=[value * scale for value in values], session=test_session
            ).save(name)

    def build(delta):
        kwargs = {"delta": delta, "delta_on": "id", "delta_unsafe": True}
        left = dc.read_dataset("multi_left", session=test_session, **kwargs)
        right = dc.read_dataset("multi_right", session=test_session, **kwargs)
        chain = (
            left.merge(right, on="id", inner=True)
            if operation == "merge"
            else left.union(right)
        )
        return chain.map(result=record, output=int)

    save_sources([1, 2, 3], [1, 2, 3])
    build(True).save("multi_result")
    save_sources([1, 3, 4], [9, 3, 4])
    processed.clear()
    result = build(True).save("multi_result")
    assert sorted(processed) == ([1, 4] if operation == "merge" else [1, 1, 4, 4])
    columns = ["id", "value", "result"]
    if operation == "merge":
        columns.append("right_value")
    assert sorted(result.to_iter(*columns)) == sorted(build(False).to_iter(*columns))
