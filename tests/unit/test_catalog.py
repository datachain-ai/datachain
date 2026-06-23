import pytest

from datachain.catalog.catalog import DatasetRowsFetcher, _round_robin_batch


@pytest.mark.parametrize(
    "urls, num_workers, expected",
    [
        ([], 3, [[], [], []]),
        (["a"], 3, [["a"], [], []]),
        (["a", "b", "c"], 3, [["a"], ["b"], ["c"]]),
        (
            ["a", "b", "c", "d", "e"],
            3,
            [["a", "d"], ["b", "e"], ["c"]],
        ),
        (
            ["u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7"],
            5,
            [["u0", "u5"], ["u1", "u6"], ["u2", "u7"], ["u3"], ["u4"]],
        ),
        (["a", "b"], 1, [["a", "b"]]),
    ],
)
def test_round_robin_batch(urls, num_workers, expected):
    assert _round_robin_batch(urls, num_workers) == expected


def test_fix_columns_converts_optional_datetime():
    """pull_dataset must convert epoch timestamps back to datetimes for both plain
    and Optional[datetime] columns. A nullable type deserializes to an instance, a
    plain one to the class, so the column-selection must accept both."""
    pd = pytest.importorskip("pandas")
    from datachain.dataset import parse_schema
    from datachain.sql.types import DateTime, Int

    opt_dt = DateTime()
    opt_dt.dc_nullable = True
    # serialize + parse to mirror what pull_dataset receives from the remote schema
    schema = parse_schema({"ts": DateTime().to_dict(), "ts_opt": opt_dt.to_dict()})
    fetcher = object.__new__(DatasetRowsFetcher)
    fetcher.schema = {"sys__id": Int, **schema}

    df = pd.DataFrame(
        {
            "sys__id": [1, 2],
            "ts": [1700000000, 1700000001],
            "ts_opt": [1700000000, None],
        }
    )
    out = fetcher.fix_columns(df)
    assert pd.api.types.is_datetime64_any_dtype(out["ts"])
    assert pd.api.types.is_datetime64_any_dtype(out["ts_opt"])
