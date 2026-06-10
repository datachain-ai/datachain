import uuid
from datetime import datetime, timezone

import pytest

import datachain as dc


def _save(test_session, **columns):
    name = uuid.uuid4().hex
    return dc.read_values(session=test_session, **columns).save(name)


def _dataset_version(catalog, chain):
    dataset = catalog.get_dataset(chain.name, versions=[chain.version])
    return dataset, dataset.get_version(chain.version)


def test_compute_dataset_stats_numeric(test_session):
    chain = _save(test_session, num=[1, 2, 3, 4, 100, 7])
    catalog = test_session.catalog
    dataset, _ = _dataset_version(catalog, chain)

    stats = catalog.warehouse.compute_dataset_stats(dataset, chain.version)

    assert stats["stats_version"] == 1
    assert stats["row_count"] == 6
    assert stats["sampled"] is False
    col = stats["columns"]["num"]
    assert col["kind"] == "numeric"
    assert col["type"] == "int"
    assert col["min"] == 1
    assert col["max"] == 100
    assert col["non_null_count"] == 6
    assert col["null_count"] == 0
    assert col["avg"] == pytest.approx(19.5)
    assert col["stddev"] > 0
    # histogram buckets must cover exactly the non-null rows
    assert sum(col["histogram"]["counts"]) == 6
    assert len(col["histogram"]["edges"]) == len(col["histogram"]["counts"]) + 1


def test_compute_dataset_stats_categorical(test_session):
    chain = _save(test_session, label=["a", "a", "b", "c", "a", "a"])
    catalog = test_session.catalog
    dataset, _ = _dataset_version(catalog, chain)

    stats = catalog.warehouse.compute_dataset_stats(dataset, chain.version)
    col = stats["columns"]["label"]

    assert col["kind"] == "categorical"
    assert col["distinct_count"] == 3
    assert col["distinct_approx"] is False
    # top_k is ordered by descending frequency
    assert col["top_k"][0] == {"value": "a", "count": 4}
    assert {t["value"] for t in col["top_k"]} == {"a", "b", "c"}


def test_compute_dataset_stats_boolean_and_nulls(test_session):
    chain = _save(test_session, flag=[True, False, True, None, True])
    catalog = test_session.catalog
    dataset, _ = _dataset_version(catalog, chain)

    stats = catalog.warehouse.compute_dataset_stats(dataset, chain.version)
    col = stats["columns"]["flag"]

    assert col["kind"] == "boolean"
    assert col["true_count"] == 3
    assert col["false_count"] == 1
    assert col["null_count"] == 1
    assert col["non_null_count"] == 4


def test_compute_dataset_stats_temporal(test_session):
    a = datetime(2020, 1, 1, tzinfo=timezone.utc)
    b = datetime(2024, 6, 1, tzinfo=timezone.utc)
    chain = _save(test_session, ts=[a, b, a])
    catalog = test_session.catalog
    dataset, _ = _dataset_version(catalog, chain)

    stats = catalog.warehouse.compute_dataset_stats(dataset, chain.version)
    col = stats["columns"]["ts"]

    assert col["kind"] == "temporal"
    assert str(a.year) in str(col["min"])
    assert str(b.year) in str(col["max"])


def test_compute_dataset_stats_skips_complex_columns(test_session):
    chain = _save(
        test_session,
        num=[1, 2, 3],
        emb=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    )
    catalog = test_session.catalog
    dataset, _ = _dataset_version(catalog, chain)

    stats = catalog.warehouse.compute_dataset_stats(dataset, chain.version)

    assert "num" in stats["columns"]
    assert "emb" not in stats["columns"]
    assert "emb" in stats["skipped_columns"]


def test_stats_not_computed_eagerly_on_sqlite(test_session):
    """SQLite is a 'slow' backend: stats stay None after save (lazy)."""
    chain = _save(test_session, num=[1, 2, 3])
    catalog = test_session.catalog
    assert catalog.warehouse.stats_compute_is_cheap() is False
    _, version = _dataset_version(catalog, chain)
    assert version.stats is None


def test_stats_computed_eagerly_when_backend_is_cheap(test_session, monkeypatch):
    monkeypatch.setattr(
        type(test_session.catalog.warehouse),
        "stats_compute_is_cheap",
        lambda self: True,
    )
    chain = _save(test_session, num=[1, 2, 3])
    catalog = test_session.catalog
    _, version = _dataset_version(catalog, chain)
    assert version.stats is not None
    assert version.stats["columns"]["num"]["max"] == 3


def test_python_accessor_computes_and_caches(test_session):
    chain = _save(test_session, num=[1, 2, 3], label=["x", "y", "x"])
    catalog = test_session.catalog

    stats = chain.stats()
    assert stats["row_count"] == 3
    assert stats["columns"]["label"]["distinct_count"] == 2

    # The result was persisted on the version and is loaded on a fresh read.
    _, version = _dataset_version(catalog, chain)
    assert version.stats is not None
    assert version.stats["columns"]["num"]["max"] == 3


def test_get_dataset_stats_uses_cache(test_session):
    chain = _save(test_session, num=[1, 2, 3])
    catalog = test_session.catalog

    first = catalog.get_dataset_stats(chain.name, chain.version)

    calls = {"n": 0}
    original = catalog.warehouse.compute_dataset_stats

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    catalog.warehouse.compute_dataset_stats = counting  # type: ignore[method-assign]

    cached = catalog.get_dataset_stats(chain.name, chain.version)
    assert calls["n"] == 0  # served from cache, not recomputed
    assert cached["computed_at"] == first["computed_at"]

    forced = catalog.get_dataset_stats(chain.name, chain.version, force=True)
    assert calls["n"] == 1
    assert forced["row_count"] == 3


def test_cli_dataset_stats_table_and_json(test_session, capsys):
    from datachain.cli.commands.datasets import dataset_stats

    chain = _save(test_session, num=[1, 2, 3], label=["a", "a", "b"])
    catalog = test_session.catalog

    dataset_stats(catalog, chain.name, chain.version)
    out = capsys.readouterr().out
    assert "rows: 3" in out
    assert "num" in out
    assert "label" in out

    dataset_stats(catalog, chain.name, chain.version, as_json=True)
    out = capsys.readouterr().out
    assert '"stats_version": 1' in out
    assert '"row_count": 3' in out


def test_stats_empty_dataset(test_session):
    chain = (
        dc.read_values(x=[1, 2, 3], y=[1.0, 2.0, 3.0], session=test_session)
        .filter(dc.C("x") > 100)
        .save(uuid.uuid4().hex)
    )
    catalog = test_session.catalog
    dataset, _ = _dataset_version(catalog, chain)

    stats = catalog.warehouse.compute_dataset_stats(dataset, chain.version)
    assert stats["row_count"] == 0
    assert stats["columns"]["x"]["non_null_count"] == 0
    assert stats["columns"]["x"]["null_count"] == 0
    assert stats["columns"]["x"]["min"] is None
    assert stats["columns"]["y"]["histogram"] is None


def test_stats_numeric_with_nulls(test_session):
    chain = _save(test_session, num=[10, None, 30, None])
    catalog = test_session.catalog
    dataset, _ = _dataset_version(catalog, chain)

    col = catalog.warehouse.compute_dataset_stats(dataset, chain.version)["columns"][
        "num"
    ]
    assert col["null_count"] == 2
    assert col["non_null_count"] == 2
    assert col["min"] == 10
    assert col["max"] == 30


def test_stats_single_value_histogram(test_session):
    chain = _save(test_session, num=[5, 5, 5])
    catalog = test_session.catalog
    dataset, _ = _dataset_version(catalog, chain)

    hist = catalog.warehouse.compute_dataset_stats(dataset, chain.version)["columns"][
        "num"
    ]["histogram"]
    assert sum(hist["counts"]) == 3
    assert len(hist["edges"]) == len(hist["counts"]) + 1


def test_stats_bins_zero_is_clamped(test_session):
    chain = _save(test_session, num=[1, 2, 3, 4])
    catalog = test_session.catalog
    dataset, _ = _dataset_version(catalog, chain)

    # bins=0 must be clamped, not raise ZeroDivisionError
    stats = catalog.warehouse.compute_dataset_stats(dataset, chain.version, bins=0)
    assert stats["columns"]["num"]["histogram"] is not None


def test_stats_max_columns_truncation(test_session):
    chain = _save(test_session, a=[1], b=[2], c=[3])
    catalog = test_session.catalog
    dataset, _ = _dataset_version(catalog, chain)

    stats = catalog.warehouse.compute_dataset_stats(
        dataset, chain.version, max_columns=1
    )
    assert len(stats["columns"]) == 1
    assert len(stats["skipped_columns"]) == 2
    assert "max_columns_exceeded" in stats["skipped_columns"].values()


def test_stats_columns_filter(test_session):
    chain = _save(test_session, a=[1, 2], b=[3, 4])
    catalog = test_session.catalog
    dataset, _ = _dataset_version(catalog, chain)

    stats = catalog.warehouse.compute_dataset_stats(
        dataset, chain.version, columns=["a"]
    )
    assert set(stats["columns"]) == {"a"}


def test_stats_temporal_json_serializable(test_session):
    import json as stdlib_json

    a = datetime(2020, 1, 1, tzinfo=timezone.utc)
    b = datetime(2021, 1, 1, tzinfo=timezone.utc)
    chain = _save(test_session, ts=[a, b])
    catalog = test_session.catalog
    dataset, _ = _dataset_version(catalog, chain)

    stats = catalog.warehouse.compute_dataset_stats(dataset, chain.version)
    col = stats["columns"]["ts"]
    assert isinstance(col["min"], str)
    assert isinstance(col["max"], str)
    # Regression: temporal values must be coerced so the result is stdlib-JSON
    # serializable and consistent with the value reloaded from the DB.
    stdlib_json.dumps(stats)
