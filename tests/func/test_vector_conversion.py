import math

import pytest

import datachain as dc


@pytest.mark.parametrize(
    "annotation,nested",
    [(list[float], False), (list[float], True), (tuple[float, ...], True)],
)
def test_vector_values_and_shape_survive_saved_dataset(
    test_session, annotation, nested
):
    values = [[0.5, 1.5], [], [math.nan, 2.0]]
    if annotation == tuple[float, ...]:
        values = [tuple(value) for value in values]

    class Vector(dc.DataModel):
        values: annotation

    dc.read_values(
        index=list(range(len(values))),
        vector=[Vector(values=value) for value in values] if nested else values,
        output={"index": int, "vector": Vector if nested else annotation},
        session=test_session,
    ).save("vectors")
    chain = dc.read_dataset("vectors", session=test_session).order_by("index")
    expected_type = tuple if annotation == tuple[float, ...] else list

    for actual, expected in zip(chain.to_values("vector"), values, strict=True):
        actual = actual.values if nested else actual
        assert type(actual) is expected_type
        assert actual == pytest.approx(expected, nan_ok=True)

    def size(vector) -> int:
        assert type(vector) is expected_type
        return len(vector)

    assert (
        chain.map(size=size, params=["vector.values" if nested else "vector"])
        .order_by("index")
        .to_values("size")
    ) == [2, 0, 2]
