import pytest

import datachain as dc


def _items_chain(test_session):
    return dc.read_values(
        name=["a", "b", "c", "d", "e"],
        emb=[
            [1.0, 0.0, 0.0],  # closest to query
            [0.9, 0.1, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],  # farthest
        ],
        session=test_session,
    )


@pytest.mark.parametrize("metric", ["cosine", "euclidean", "l2"])
def test_returns_top_k_in_order(test_session, metric):
    """For all metrics, the closest match comes first and we get exactly k rows."""
    chain = _items_chain(test_session)
    result = chain.similarity_search(
        "emb", [1.0, 0.0, 0.0], k=3, metric=metric
    ).to_values("name")
    assert result[0] == "a"
    assert len(result) == 3


@pytest.mark.parametrize(
    "k,expected_count",
    [
        (1, 1),
        (3, 3),
        (10, 5),  # k > N → all rows
        (None, 5),  # no limit → all rows
    ],
)
def test_k_controls_result_size(test_session, k, expected_count):
    chain = _items_chain(test_session)
    result = chain.similarity_search("emb", [1.0, 0.0, 0.0], k=k).to_values("name")
    assert len(result) == expected_count
    assert result[0] == "a"  # closest first regardless of k


def test_score_column_hidden_by_default(test_session):
    chain = _items_chain(test_session)
    result = chain.similarity_search("emb", [1.0, 0.0, 0.0], k=2)
    assert set(result.signals_schema.values) - {"sys"} == {"name", "emb"}


def test_score_column_visible_when_named(test_session):
    chain = _items_chain(test_session)
    rows = chain.similarity_search(
        "emb", [1.0, 0.0, 0.0], k=2, score_column="dist"
    ).to_list("name", "dist")
    assert [r[0] for r in rows] == ["a", "b"]
    assert rows[0][1] == pytest.approx(0.0, abs=1e-6)


def test_l2_matches_euclidean(test_session):
    chain = _items_chain(test_session)
    a = chain.similarity_search(
        "emb", [1.0, 0.0, 0.0], k=5, metric="euclidean", score_column="d"
    ).to_list("name", "d")
    b = chain.similarity_search(
        "emb", [1.0, 0.0, 0.0], k=5, metric="l2", score_column="d"
    ).to_list("name", "d")
    assert a == b


def test_invalid_metric_raises(test_session):
    chain = _items_chain(test_session)
    with pytest.raises(ValueError, match="Unsupported metric"):
        chain.similarity_search("emb", [1.0, 0.0, 0.0], metric="manhattan")
