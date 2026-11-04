import pytest

import datachain as dc
from datachain import C, func


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
