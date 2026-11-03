import pytest

import datachain as dc
from datachain import C, File, func
from tests.utils import assert_delta_refresh


@pytest.mark.parametrize(
    "operation", ["merge", "inline_merge", "inner_merge", "grouped_merge", "subtract"]
)
def test_delta_with_unchanged_static_input_replays_changed_records(
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
        labels = (
            dc.read_values(group=[1, 2], label=["a", "b"], session=test_session)
            if operation == "inline_merge"
            else dc.read_dataset("labels", session=test_session)
        )
        if operation == "subtract":
            return chain.subtract(labels, on="group")
        if operation == "grouped_merge":
            labels = labels.group_by(count=func.count(), partition_by="group")
        return chain.merge(labels, on="group", inner=operation == "inner_merge")

    dc.read_values(group=[1, 2], label=["a", "b"], session=test_session).save("labels")
    source([1, 2, 3], [1, 2, 3])
    refreshed = assert_delta_refresh(
        build,
        "enriched_result",
        lambda: source([1, 3, 4], [1, 2, 3]),
        processed=processed,
    )
    assert refreshed == [3, 4]


def test_full_rebuild_cleans_legacy_delta_results(test_session, tmp_path):
    data = tmp_path / "files"
    data.mkdir()
    (data / "a.txt").write_text("2")
    (data / "b.txt").write_text("3")

    def read_number(file: File) -> int:
        return int(file.read_text())

    def build(delta):
        return (
            dc.read_storage(data, update=True, delta=delta, session=test_session)
            .map(value=read_number)
            .filter(C.value < 5)
        )

    first = build(True).save("numbers")
    (data / "a.txt").write_text("99")
    (data / "b.txt").unlink()
    (data / "c.txt").write_text("4")
    legacy = (
        build(False)
        .union(dc.read_dataset("numbers", version=first.version, session=test_session))
        .save("numbers")
    )
    assert sorted(legacy.to_values("value")) == [2, 3, 4]
    assert sorted(build(True).save("numbers").to_values("value")) == [2, 3, 4]
    rebuilt = build(False).save("numbers")
    assert rebuilt.to_values("value") == [4]
    (data / "d.txt").write_text("1")
    assert sorted(build(True).save("numbers").to_values("value")) == [1, 4]
    assert sorted(legacy.to_values("value")) == [2, 3, 4]


@pytest.mark.parametrize("reuse", [False, True])
def test_delta_subtracts_filtered_occurrence_of_same_source(test_session, reuse):
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
    result = build(True).save("self_subtract_result")
    assert processed == [4]
    assert (
        sorted(result.to_values("id")) == sorted(build(False).to_values("id")) == [3, 4]
    )
    assert initial.to_values("id") == [3]


@pytest.mark.parametrize("changed", ["left", "right", "both"])
@pytest.mark.parametrize("rename", [False, True])
def test_delta_unsafe_union_replays_independent_append_only_sources(
    test_session, changed, rename
):
    processed = []

    def record(id) -> int:
        processed.append(id)
        return id * 10

    def source(name, ids):
        dc.read_values(
            **{f"{name[0]}_id" if rename else "id": ids}, session=test_session
        ).save(name)

    def build(delta):
        def read(name):
            key = f"{name[0]}_id" if rename else "id"
            chain = dc.read_dataset(
                name,
                delta=delta,
                delta_on=key,
                delta_result_on="id",
                delta_unsafe=True,
                session=test_session,
            )
            return chain.select(id=C(key)) if rename else chain

        return read("left").union(read("right")).map(value=record)

    source("left", [1, 2])
    source("right", [10, 20])
    first = build(True).save("independent_result")
    if changed in ("left", "both"):
        source("left", [1, 2, 3])
    if changed in ("right", "both"):
        source("right", [10, 20, 30])
    processed.clear()
    result = build(True).save("independent_result")
    assert sorted(processed) == (
        [3, 30] if changed == "both" else [3] if changed == "left" else [30]
    )
    assert sorted(result.to_iter("id", "value")) == sorted(
        build(False).to_iter("id", "value")
    )
    assert sorted(first.to_values("id")) == [1, 2, 10, 20]
    assert build(True).save("independent_result").version == result.version


@pytest.mark.parametrize("static_first", [False, True])
def test_delta_unsafe_union_with_static_input_does_not_recompute_source(
    test_session, static_first
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
        left, right = (static, changing) if static_first else (changing, static)
        return left.union(right)

    dc.read_values(id=[1, 2], session=test_session).save("changing")
    dc.read_values(id=[9], value=[90], session=test_session).save("static")
    first = build(True).save("static_result")
    dc.read_values(id=[1, 2, 3], session=test_session).save("changing")
    processed.clear()
    result = build(True).save("static_result")
    assert processed == [3]
    assert sorted(result.to_iter("id", "value")) == [(1, 10), (2, 20), (3, 30), (9, 90)]
    assert sorted(first.to_values("id")) == [1, 2, 9]


@pytest.mark.parametrize("operation", ["distinct", "group_by"])
def test_delta_unsafe_row_local_grouping_does_not_recompute(test_session, operation):
    processed = []

    def record(id) -> int:
        processed.append(id)
        return id * 10

    def build(delta):
        chain = dc.read_dataset(
            "group_source",
            delta=delta,
            delta_on="id",
            delta_unsafe=True,
            session=test_session,
        ).map(value=record)
        if operation == "distinct":
            return chain.distinct("id")
        return chain.group_by(partition_by="id", value=func.sum("value"))

    dc.read_values(id=[1, 2], session=test_session).save("group_source")
    build(True).save("group_result")
    dc.read_values(id=[1, 2, 3], session=test_session).save("group_source")
    processed.clear()
    result = build(True).save("group_result")
    assert processed == [3]
    assert sorted(result.to_iter("id", "value")) == [(1, 10), (2, 20), (3, 30)]


def test_delta_changes_to_static_lookup_do_not_trigger_full_recompute(test_session):
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


@pytest.mark.parametrize("retry", [False, True])
def test_storage_delta_changes_outside_glob_reuse_result_version(
    test_session, tmp_path, retry
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
            delta_retry=retry,
            session=test_session,
        ).map(value=record)

    first = build().save("glob_result")
    processed.clear()
    for index in range(2):
        (tmp_path / f"ignored{index}.log").write_text("2")
        result = build().save("glob_result")
        assert result.version == first.version
        assert result.to_values("value") == [1]
        assert processed == []
