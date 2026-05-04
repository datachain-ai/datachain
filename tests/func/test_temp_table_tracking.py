import pytest

import datachain as dc
from datachain import func
from datachain.query.dataset import DatasetQuery


def _capture_temp_tables(mocker):
    captured: list[list[str]] = []
    original_cleanup = DatasetQuery.cleanup

    def capture(self):
        captured.append(list(self.temp_table_names))
        return original_cleanup(self)

    mocker.patch("datachain.query.dataset.DatasetQuery.cleanup", capture)
    return captured


def _assert_no_duplicate_temp_tables(captured: list[list[str]]):
    for tables in captured:
        assert len(tables) == len(set(tables))


def _query_plan(chain) -> str:
    query = chain._query.apply_steps().select()
    try:
        args = chain.session.catalog.warehouse.db.compile_to_args(query)
        rows = chain.session.catalog.warehouse.db.execute_str(
            f"EXPLAIN QUERY PLAN {args[0]}",
            args[1] if len(args) > 1 else None,
        ).fetchall()
        return "\n".join(str(row) for row in rows)
    finally:
        chain._query.cleanup()


def test_nested_merge_has_no_duplicate_temp_tables(test_session, mocker):
    captured = _capture_temp_tables(mocker)

    base = dc.read_values(num=[1, 2], session=test_session)
    generated = base.map(num_plus=lambda num: str(num + 10))
    inner = generated.merge(base, on="num", inner=True)
    chain = base.merge(inner, on="num", inner=True)

    values = chain.select("num").to_pandas()["num"].tolist()
    assert set(values) == {1, 2}
    assert len(values) == 2

    rerun = chain.select("num").to_pandas()["num"].tolist()
    assert set(rerun) == {1, 2}
    assert len(rerun) == len(values)

    _assert_no_duplicate_temp_tables(captured)


@pytest.mark.parametrize("right_mode", ["mutated", "inline"])
def test_sqlite_merge_dynamic_right_key_uses_automatic_index(test_session, right_mode):
    if test_session.catalog.warehouse.db.dialect.name != "sqlite":
        pytest.skip("SQLite-specific query plan")

    left = dc.read_values(
        path=["/l/a.txt", "/l/b.txt", "/l/c.txt"],
        session=test_session,
    )
    right = dc.read_values(
        path=["/r/a.csv", "/r/c.csv"],
        value=[10, 30],
        session=test_session,
    )
    if right_mode == "mutated":
        right = right.mutate(stem=func.file_stem("path")).select("stem", "value")
        right_on = "stem"
    else:
        right_on = func.file_stem("path")

    chain = left.merge(right, on=func.file_stem("path"), right_on=right_on)
    plan = _query_plan(chain)
    assert "AUTOMATIC" in plan
    assert "tmp_" in plan


def test_sqlite_merge_physical_key_does_not_materialize(test_session, mocker):
    if test_session.catalog.warehouse.db.dialect.name != "sqlite":
        pytest.skip("SQLite-specific query plan")
    captured = _capture_temp_tables(mocker)

    left = dc.read_values(key=["a", "b", "c"], session=test_session)
    right = dc.read_values(key=["a", "c"], value=[10, 30], session=test_session)

    rows = left.merge(right, on="key").select("key", "value").results()
    assert sorted(rows) == [("a", 10), ("b", None), ("c", 30)]
    assert all(not tables for tables in captured)


def test_union_has_no_duplicate_temp_tables(test_session, mocker):
    captured = _capture_temp_tables(mocker)

    left = dc.read_values(num=[1, 2], session=test_session)
    right = dc.read_values(num=[3], session=test_session)
    union_chain = left.union(right)

    values = union_chain.select("num").to_pandas()["num"].tolist()
    assert set(values) == {1, 2, 3}
    assert len(values) == 3

    rerun = union_chain.select("num").to_pandas()["num"].tolist()
    assert set(rerun) == {1, 2, 3}
    assert len(rerun) == len(values)

    _assert_no_duplicate_temp_tables(captured)


def test_subtract_has_no_duplicate_temp_tables(test_session, mocker):
    captured = _capture_temp_tables(mocker)

    source = dc.read_values(num=[1, 2], session=test_session)
    target = dc.read_values(num=[2], session=test_session)
    subtract_chain = source.subtract(target, on="num")

    values = subtract_chain.select("num").to_pandas()["num"].tolist()
    assert values == [1]

    rerun = subtract_chain.select("num").to_pandas()["num"].tolist()
    assert rerun == values

    _assert_no_duplicate_temp_tables(captured)
