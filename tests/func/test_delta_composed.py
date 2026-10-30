import pytest

import datachain as dc
from datachain import C, File


@pytest.mark.parametrize("changed", ["left", "right", "both"])
@pytest.mark.parametrize("how", ["inner", "left", "full"])
def test_delta_join_with_independent_changes(test_session, changed, how):
    def source(name, ids, values, column):
        dc.read_values(id=ids, **{column: values}, session=test_session).save(name)

    source("join_left", [1, 2], [10, 20], "left_value")
    source("join_right", [1, 2, 3], [100, 200, 300], "right_value")

    def build(delta):
        kwargs = {
            "delta": delta,
            "delta_on": "id",
            "delta_unsafe": True,
            "session": test_session,
        }
        return dc.read_dataset("join_left", **kwargs).merge(
            dc.read_dataset("join_right", **kwargs),
            on="id",
            inner=how == "inner",
            full=how == "full",
        )

    def rows(chain):
        return sorted(
            chain.to_iter("id", "left_value", "right_id", "right_value"), key=repr
        )

    first = build(True).save("join_result")
    initial = rows(first)
    if changed in ("left", "both"):
        source("join_left", [1, 3], [11, 30], "left_value")
    if changed in ("right", "both"):
        source("join_right", [1, 2, 4], [100, 201, 400], "right_value")
    result = build(True).save("join_result")
    assert rows(result) == rows(build(False))
    assert rows(first) == initial
    assert build(True).save("join_result").version == result.version


@pytest.mark.parametrize("changed", ["left", "right"])
@pytest.mark.parametrize("delete", [False, True])
def test_delta_union_with_overlapping_source_keys(test_session, changed, delete):
    def source(name, ids, values):
        dc.read_values(id=ids, value=values, session=test_session).save(name)

    source("left", [1, 2], [10, 20])
    source("right", [1, 3], [100, 300])

    def build(delta):
        kwargs = {
            "delta": delta,
            "delta_on": "id",
            "delta_unsafe": True,
            "session": test_session,
        }
        return dc.read_dataset("left", **kwargs).union(
            dc.read_dataset("right", **kwargs)
        )

    build(True).save("overlap_result")
    source(changed, [4] if delete else [1, 4], [40] if delete else [11, 40])
    result = build(True).save("overlap_result")
    assert sorted(result.to_iter("id", "value")) == sorted(
        build(False).to_iter("id", "value")
    )
    assert build(True).save("overlap_result").version == result.version


@pytest.mark.parametrize("operation", ["union", "merge"])
@pytest.mark.parametrize("static_first", [False, True])
def test_delta_composed_query_refreshes_changed_static_input(
    test_session, operation, static_first
):
    dc.read_values(id=[1, 2], value=[10, 20], session=test_session).save("changing")
    dc.read_values(id=[1, 2], value=[100, 200], session=test_session).save("static")

    def build(delta):
        changing = dc.read_dataset(
            "changing",
            delta=delta,
            delta_on="id",
            delta_unsafe=True,
            session=test_session,
        )
        static = dc.read_dataset("static", session=test_session)
        left, right = (static, changing) if static_first else (changing, static)
        return left.union(right) if operation == "union" else left.merge(right, on="id")

    build(True).save("static_result")
    dc.read_values(id=[2, 3], value=[201, 300], session=test_session).save("static")
    result = build(True).save("static_result")
    columns = ["id", "value"]
    if operation == "merge":
        columns.extend(["right_id", "right_value"])
    assert sorted(result.to_iter(*columns), key=repr) == sorted(
        build(False).to_iter(*columns), key=repr
    )


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


@pytest.mark.parametrize(
    "operation", ["limit", "offset", "distinct", "group_by", "subtract", "window"]
)
def test_delta_cross_record_operations_match_full_recompute(test_session, operation):
    from datachain import func

    def source(ids, values):
        dc.read_values(id=ids, value=values, session=test_session).save("global_source")

    dc.read_values(value=[10], session=test_session).save("excluded")

    def build(delta):
        chain = dc.read_dataset(
            "global_source",
            delta=delta,
            delta_on="id",
            delta_unsafe=True,
            session=test_session,
        )
        if operation == "limit":
            return chain.order_by("id").limit(2)
        if operation == "offset":
            return chain.order_by("id").offset(1)
        if operation == "distinct":
            return chain.distinct("id")
        if operation == "group_by":
            return chain.group_by(partition_by="id", value=func.sum("value"))
        if operation == "subtract":
            return chain.subtract(
                dc.read_dataset("excluded", session=test_session), on="value"
            )
        return chain.mutate(value=func.row_number().over(func.window(order_by="id")))

    source([1, 2, 3], [10, 20, 30])
    initial = build(True).save("global_result")
    before = sorted(initial.to_iter("id", "value"))
    source([2, 3, 4], [21, 30, 40])
    result = build(True).save("global_result")
    assert sorted(result.to_iter("id", "value")) == sorted(
        build(False).to_iter("id", "value")
    )
    assert sorted(initial.to_iter("id", "value")) == before
    assert build(True).save("global_result").version == result.version
