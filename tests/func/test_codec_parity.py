"""Round trips adopted from #1979's parity tests (the typed-codec draft). They test
the stored values only: #1979's per-signal codec metadata is not ours."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

import datachain as dc

DT_0 = datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=UTC)
DT_1 = datetime(2024, 2, 3, 4, 5, 6, 654321, tzinfo=UTC)
DT_2 = datetime(2024, 3, 4, 5, 6, 7, tzinfo=UTC)

WRITERS = ["read_records", "read_values", "map"]


def _collection_chain(test_session, writer: str, annotation: Any, values: list[Any]):
    row_ids = list(range(len(values)))
    if writer == "read_records":
        return dc.read_records(
            [
                {"row_id": row_id, "value": value}
                for row_id, value in zip(row_ids, values, strict=True)
            ],
            schema={"row_id": int, "value": annotation},
            session=test_session,
        )
    if writer == "read_values":
        return dc.read_values(
            row_id=row_ids,
            value=values,
            output={"row_id": int, "value": annotation},
            session=test_session,
        )
    by_id = dict(zip(row_ids, values, strict=True))

    def emit(row_id: int):
        return by_id[row_id]

    return dc.read_values(row_id=row_ids, session=test_session).map(
        value=emit, output=annotation
    )


def _save_and_reload(chain, test_session, prefix: str):
    name = f"{prefix}_{uuid4().hex}"
    chain.save(name)
    return dc.read_dataset(name, session=test_session)


@pytest.mark.parametrize(
    ("annotation", "values"),
    [
        pytest.param(
            list[int | None],
            [[], [None], [None, 1], [1, None], [None, None], [1, 2]],
            id="int",
        ),
        pytest.param(
            list[float | None],
            [[], [None], [None, 1.25], [1.25, None], [None, None], [1.25, 2.5]],
            id="float",
        ),
    ],
)
@pytest.mark.parametrize("writer", WRITERS)
def test_nullable_numeric_array_write_path_parity(
    test_session, writer, annotation, values
):
    """Every writer keeps a None item a None, wherever it sits."""
    chain = _collection_chain(test_session, writer, annotation, values)
    assert chain.order_by("row_id").to_values("value") == values
    reloaded = _save_and_reload(chain, test_session, f"codec_nullable_{writer}")
    assert reloaded.order_by("row_id").to_values("value") == values


@pytest.mark.parametrize("writer", WRITERS)
def test_nested_nullable_datetime_round_trip(writer, test_session):
    """A nullable nested annotation must not turn datetimes into strings; the
    first row has no None, so the data cannot be what decides."""
    values = [
        [[DT_0, DT_1], [DT_2]],
        [None, [DT_2], None],
        [None, None],
        [],
    ]
    chain = _collection_chain(test_session, writer, list[list[datetime] | None], values)
    reloaded = _save_and_reload(chain, test_session, f"codec_datetime_{writer}")

    actual = reloaded.order_by("row_id").to_values("value")
    assert actual == values
    assert isinstance(actual[0][0][0], datetime)
    assert isinstance(actual[1][1][0], datetime)


@pytest.mark.parametrize("writer", WRITERS)
def test_optional_collection_distinguishes_none_from_empty(writer, test_session):
    values = [None, [], [1, 2]]
    chain = _collection_chain(test_session, writer, list[int] | None, values)
    reloaded = _save_and_reload(chain, test_session, f"codec_optional_{writer}")

    assert reloaded.order_by("row_id").to_values("value") == values


@pytest.mark.parametrize(
    ("annotation", "value"),
    [
        pytest.param(list[int | None], [None, 1, None], id="nullable-list"),
        pytest.param(
            list[list[datetime] | None],
            [None, [DT_0, DT_1], None],
            id="nested-datetime",
        ),
    ],
)
def test_read_values_single_signal_preserves_collection_as_one_value(
    annotation, value, test_session
):
    """A list value is one output, not a multi-output row wrapper."""
    chain = dc.read_values(
        value=[value], output={"value": annotation}, session=test_session
    )
    assert chain.to_values("value") == [value]

    reloaded = _save_and_reload(chain, test_session, "codec_single_signal")
    assert reloaded.to_values("value") == [value]
