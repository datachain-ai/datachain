import datachain as dc


def test_union_persist_no_duplication_large_session(test_session, monkeypatch):
    # See https://github.com/datachain-ai/datachain/issues/1356
    # Lower insert batch size to keep the test fast and still cross the boundary.
    monkeypatch.setattr(
        "datachain.data_storage.warehouse.INSERT_BATCH_SIZE", 20, raising=False
    )
    monkeypatch.setattr("datachain.query.dataset.INSERT_BATCH_SIZE", 20, raising=False)
    n = 20 + 7

    x_ids = list(range(n))
    y_ids = list(range(n, 2 * n))

    x = dc.read_values(idx=x_ids, session=test_session)
    y = dc.read_values(idx=y_ids, session=test_session)

    xy = x.union(y)
    assert xy.count() == 2 * n

    xy_p = xy.persist()
    assert xy_p.count() == 2 * n

    distinct_idx = {v for (v,) in xy_p.select("idx").results()}
    assert len(distinct_idx) == 2 * n

    j = xy_p.merge(x, on="idx", inner=True)
    assert j.count() == n


def test_union_parallel_udf_ids_only_no_dup(test_session_tmpfile, monkeypatch):
    # Validate that after union, running a parallel UDF that uses the ids-only
    # path does not duplicate rows due to sys__id collisions across branches.
    # This specifically exercises the parallel dispatch path where input rows
    # are split by sys__id and fetched per worker using IN (...) filters.
    # See https://github.com/datachain-ai/datachain/issues/1356

    # Make worker/ids fetch batches small to exercise splitting on tiny inputs.
    monkeypatch.setattr("datachain.query.dispatch.DEFAULT_BATCH_SIZE", 5, raising=False)
    n = 30

    x_ids = list(range(n))
    y_ids = list(range(n, 2 * n))

    x = dc.read_values(idx=x_ids, session=test_session_tmpfile)
    y = dc.read_values(idx=y_ids, session=test_session_tmpfile)

    xy = x.union(y)
    mapped = xy.settings(parallel=2).map(
        out=lambda idx: idx, output=int, params=("idx",)
    )

    total = mapped.count()
    distinct_idx = {v for (v,) in mapped.select("idx").results()}

    assert total == 2 * n
    assert len(distinct_idx) == 2 * n
    assert total == len(distinct_idx)
