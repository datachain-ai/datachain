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


class _DatedModel(dc.DataModel):
    when: datetime


class _DictModel(dc.DataModel):
    d: dict


class _EmptyMarker(dc.DataModel):
    pass


def test_mixed_numeric_array_and_dict_keys_in_array(test_session):
    chain = _save_and_reload(
        dc.read_records(
            [{"v": [1.5, 2], "d": [{b"a": "bytes", (1, "x"): "tuple"}]}],
            schema={"v": list[float], "d": list[dict]},
            session=test_session,
        ),
        test_session,
        "codec_mixed_array",
    )
    assert chain.filter(dc.C("v") == [1.5, 2]).count() == 1
    assert chain.to_values("d") == [[{"a": "bytes", "(1, 'x')": "tuple"}]]


def test_leading_none_model_list_round_trip_and_key_collision(test_session):
    from datachain.lib.udf import UdfError

    values = [None, _DatedModel(when=DT_0)]
    chain = _save_and_reload(
        dc.read_values(
            v=[values], output={"v": list[_DatedModel]}, session=test_session
        ),
        test_session,
        "codec_leading_none_model",
    )
    assert chain.to_values("v") == [values]

    with pytest.raises(UdfError, match="would be lost"):
        dc.read_records(
            [{"v": [None, _DictModel(d={"1": "first", 1: "second"})]}],
            schema={"v": list[_DictModel]},
            session=test_session,
        ).save("codec_leading_none_collision")


def test_mutable_and_nan_defaults_do_not_alias_across_rows():
    import math

    import numpy as np
    import sqlalchemy as sa

    from datachain.data_storage.codec import RowCodec, TypeCodec, decode_list
    from datachain.sql.types import JSON, Array, Float

    rc = RowCodec(
        [
            TypeCodec(frozenset(), lambda x: x, []),
            TypeCodec(frozenset(), lambda x: x, {}),
            TypeCodec(frozenset(), lambda x: x, float("nan")),
        ]
    )
    r1, r2 = rc.decode_rows([(None, None, None), (None, None, None)])
    r1[0].append(1)
    r1[1]["k"] = 2
    assert r2[0] == []
    assert r2[1] == {}
    assert math.isnan(r1[2]) and math.isnan(r2[2])

    cols = rc.decode_columns([[None, None], [None, None], [None, None]])
    cols[0][0].append(1)
    cols[1][0]["k"] = 2
    assert cols[0][1] == []
    assert cols[1][1] == {}
    assert math.isnan(cols[2][0]) and math.isnan(cols[2][1])

    assert decode_list(float, np.array([1, 2])) == [1.0, 2.0]
    for t in (JSON(), Float(), Array(Float())):
        assert isinstance(t.impl, sa.types.TypeEngine)


def test_empty_marker_columnar_chunks_keep_shape(test_session):
    chain = dc.read_records(
        [{"v": _EmptyMarker()}],
        schema={"v": _EmptyMarker | None},
        session=test_session,
    )
    names, chunks = chain.to_columnar_data_with_names(1, include_sentinels=False)
    assert names == []
    assert list(chunks) == [[]]


def test_model_dict_key_check_edge_cases():
    import numpy as np
    from pydantic import BaseModel, Field, RootModel, field_serializer

    from datachain.data_storage.codec import decode_json, encode_json

    assert decode_json(encode_json(RootModel[str]("hello"))) == "hello"
    assert decode_json(encode_json(RootModel[str]("123"))) == "123"

    class WithExcluded(BaseModel):
        visible: int = 1
        hidden: dict = Field(default_factory=lambda: {1: "a", "1": "b"}, exclude=True)

    assert decode_json(encode_json(WithExcluded())) == {"visible": 1}

    class WithUnlessNone(BaseModel):
        value: dict

        @field_serializer("value", when_used="unless-none")
        def serialize_value(self, v):
            return list(v.values())

    assert decode_json(encode_json(WithUnlessNone(value={1: "a", "1": "b"}))) == {
        "value": ["a", "b"]
    }

    class WithNumpyKey(BaseModel):
        value: dict

    assert decode_json(encode_json(WithNumpyKey(value={np.int64(2): "a"}))) == {
        "value": {"2": "a"}
    }
