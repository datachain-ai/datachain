import uuid
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
import pytest
import ujson as json
from pydantic import BaseModel, ConfigDict

from datachain import json as dcjson
from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.sql.types import (
    JSON,
    Array,
    Boolean,
    DateTime,
    Float,
    Float32,
    Float64,
    Int,
    Int64,
    SQLType,
    String,
)
from tests.utils import (
    DEFAULT_TREE,
    TARRED_TREE,
    create_tar_dataset,
)

COMPLEX_TREE: dict[str, Any] = {
    **TARRED_TREE,
    **DEFAULT_TREE,
    "nested": {"dir": {"path": {"abc.txt": "abc"}}},
}


@pytest.mark.parametrize("tree", [COMPLEX_TREE], indirect=True)
def test_dir_expansion(cloud_test_catalog, version_aware, cloud_type):
    has_version = version_aware or cloud_type == "gs"

    ctc = cloud_test_catalog
    session = ctc.session
    catalog = ctc.catalog
    src_uri = ctc.src_uri
    if cloud_type == "file":
        # we don't want to index things in parent directory
        src_uri += "/"

    chain = create_tar_dataset(session, ctc.src_uri, "dc")
    dataset = catalog.get_dataset(chain.name, versions=["1.0.0"])
    with catalog.warehouse.clone() as warehouse:
        dr = warehouse.dataset_rows(dataset, column="file")
        de = dr.dir_expansion()
        q = de.query(dr.get_table())

        columns = (
            "id",
            "is_dir",
            "source",
            "path",
            "version",
            "location",
        )

        result = [dict(zip(columns, r, strict=False)) for r in warehouse.db.execute(q)]
        to_compare = [(r["path"], r["is_dir"], r["version"] != "") for r in result]

    assert all(r["source"] == ctc.src_uri for r in result)

    # Note, we have both a file and a directory entry for expanded tar files
    expected = [
        ("animals.tar", 0, has_version),
        ("animals.tar", 1, False),
        ("animals.tar/cats", 1, False),
        ("animals.tar/cats/cat1", 0, has_version),
        ("animals.tar/cats/cat2", 0, has_version),
        ("animals.tar/description", 0, has_version),
        ("animals.tar/dogs", 1, False),
        ("animals.tar/dogs/dog1", 0, has_version),
        ("animals.tar/dogs/dog2", 0, has_version),
        ("animals.tar/dogs/dog3", 0, has_version),
        ("animals.tar/dogs/others", 1, False),
        ("animals.tar/dogs/others/dog4", 0, has_version),
        ("cats", 1, False),
        ("cats/cat1", 0, has_version),
        ("cats/cat2", 0, has_version),
        ("description", 0, has_version),
        ("dogs", 1, False),
        ("dogs/dog1", 0, has_version),
        ("dogs/dog2", 0, has_version),
        ("dogs/dog3", 0, has_version),
        ("dogs/others", 1, False),
        ("dogs/others/dog4", 0, has_version),
        ("nested", 1, False),
        ("nested/dir", 1, False),
        ("nested/dir/path", 1, False),
        ("nested/dir/path/abc.txt", 0, has_version),
    ]

    assert to_compare == expected


def test_convert_type(test_session):
    catalog = test_session.catalog
    warehouse = catalog.warehouse
    now = datetime.now()

    def run_convert_type(value, sql_type):
        return warehouse.convert_type(
            value,
            sql_type,
            warehouse.python_type(sql_type),
            type(sql_type).__name__,
            "test_column",
        )

    # convert int to float
    for f in [Float, Float32, Float64]:
        converted = run_convert_type(1, f())
        assert converted == 1.0
        assert isinstance(converted, float)

    # types match, nothing to convert
    assert run_convert_type(1, Int()) == 1
    assert run_convert_type(1.5, Float()) == 1.5
    assert run_convert_type(True, Boolean()) is True
    assert run_convert_type("s", String()) == "s"
    assert run_convert_type(now, DateTime()) == now
    assert run_convert_type([1, 2], Array(Int)) == [1, 2]
    assert run_convert_type((1, 2), Array(Int)) == [1, 2]
    assert run_convert_type([1.5, 2.5], Array(Float)) == [1.5, 2.5]
    assert run_convert_type((1.5, 2.5), Array(Float)) == [1.5, 2.5]
    assert run_convert_type(["a", "b"], Array(String)) == ["a", "b"]
    assert run_convert_type([[1, 2], [3, 4]], Array(Array(Int))) == [
        [1, 2],
        [3, 4],
    ]
    assert run_convert_type(((1, 2), (3, 4)), Array(Array(Int))) == [
        [1, 2],
        [3, 4],
    ]

    # JSON Tests
    assert run_convert_type('{"a": 1}', JSON()) == '{"a": 1}'
    assert run_convert_type({"a": 1}, JSON()) == '{"a":1}'
    assert run_convert_type([{"a": 1}], JSON()) == '[{"a":1}]'
    assert run_convert_type([[1, 2], [3, 4]], JSON()) == "[[1,2],[3,4]]"
    assert run_convert_type(None, JSON()) == "null"
    assert run_convert_type({"a": None}, JSON()) == '{"a":null}'
    dt_value = datetime(2024, 1, 2, 3, 4, 5)
    assert run_convert_type({"ts": dt_value}, JSON()) == '{"ts":"2024-01-02T03:04:05"}'
    # primitives should serialize to valid JSON
    assert run_convert_type(0.5, JSON()) == "0.5"

    out = run_convert_type(
        {
            "a": np.array([1, 2], dtype=np.int64),
            "b": {"score": np.float32(0.5)},
        },
        JSON(),
    )
    assert json.loads(out) == {"a": [1, 2], "b": {"score": 0.5}}

    out = run_convert_type({np.int64(7): "v"}, JSON())
    assert json.loads(out) == {"7": "v"}

    # JSON with Pydantic models (values and nested)
    class MyFr(BaseModel):
        model_config = ConfigDict(frozen=True)
        nnn: str
        count: int

    fr1 = MyFr(nnn="x", count=1)
    fr2 = MyFr(nnn="y", count=2)

    # Pydantic as dict value
    out = run_convert_type({"a": fr1}, JSON())
    assert out == '{"a":{"nnn":"x","count":1}}'

    # Pydantic in list
    out = run_convert_type([fr1, fr2], JSON())
    assert out == '[{"nnn":"x","count":1},{"nnn":"y","count":2}]'

    # Nested structures with Pydantic
    out = run_convert_type({"k": [{"inner": fr1}]}, JSON())
    assert out == '{"k":[{"inner":{"nnn":"x","count":1}}]}'

    # Complex dict key (tuple) becomes a JSON-encoded string key
    out = run_convert_type({(1, "a"): 3}, JSON())
    # Decode and compare to expected mapping using encoded key
    loaded = json.loads(out)
    assert loaded == {json.dumps([1, "a"]): 3}

    # Pydantic model as dict key
    key_model = MyFr(nnn="k", count=7)
    d: dict[Any, Any] = {}
    d[key_model] = "v"
    out = run_convert_type(d, JSON())
    loaded = json.loads(out)
    expected_key = json.dumps({"nnn": "k", "count": 7})
    assert loaded == {expected_key: "v"}

    # convert array to compatible type
    converted = run_convert_type([1, 2], Array(Float))
    assert converted == [1.0, 2.0]
    assert all(isinstance(c, float) for c in converted)

    # convert nested array to compatible type
    converted = run_convert_type([[1, 2], [3, 4]], Array(Array(Float)))
    assert converted == [[1.0, 2.0], [3.0, 4.0]]
    assert all(isinstance(c, float) for c in converted[0])
    assert all(isinstance(c, float) for c in converted[1])

    # error, float to int
    with pytest.raises(ValueError):
        run_convert_type(1.5, Int())

    # error, float to int in list
    with pytest.raises(ValueError):
        run_convert_type([1.5, 1], Array(Int))


class NumpyHolder(BaseModel):
    name: str
    payload: dict


def test_convert_type_writes_numpy_held_inside_a_model(test_session):
    warehouse = test_session.catalog.warehouse

    def to_json(value):
        return json.loads(
            warehouse.convert_type(
                value, JSON(), warehouse.python_type(JSON()), "JSON", "test_column"
            )
        )

    assert to_json(
        NumpyHolder(name="a", payload={"v": np.array([1, 2], dtype=np.int64)})
    ) == {
        "name": "a",
        "payload": {"v": [1, 2]},
    }
    assert to_json(NumpyHolder(name="b", payload={"v": np.float32(0.5)})) == {
        "name": "b",
        "payload": {"v": 0.5},
    }
    assert to_json(NumpyHolder(name="c", payload={"k": [np.int64(7)]})) == {
        "name": "c",
        "payload": {"k": [7]},
    }
    assert to_json([NumpyHolder(name="d", payload={"v": np.array([1.5])})]) == [
        {"name": "d", "payload": {"v": [1.5]}},
    ]


def test_convert_type_refuses_a_type_no_encoder_can_write(test_session):
    warehouse = test_session.catalog.warehouse

    class Opaque:
        pass

    with pytest.raises(TypeError, match="not JSON serializable"):
        warehouse.convert_type(
            NumpyHolder(name="a", payload={"v": Opaque()}),
            JSON(),
            warehouse.python_type(JSON()),
            "JSON",
            "test_column",
        )


REFUSED_BY_THE_ENCODER = [
    pytest.param(lambda: np.longdouble(3), id="longdouble-scalar"),
    pytest.param(lambda: np.array([1, 2], dtype=np.longdouble), id="longdouble-array"),
    pytest.param(lambda: np.clongdouble(1 + 2j), id="clongdouble-scalar"),
    pytest.param(
        lambda: np.array([1, 2], dtype=np.clongdouble), id="clongdouble-array"
    ),
    pytest.param(lambda: np.complex64(1 + 2j), id="complex"),
    pytest.param(lambda: np.timedelta64(3, "D"), id="timedelta"),
    pytest.param(lambda: np.array([b"ab"], dtype=object), id="bytes-in-object-array"),
    pytest.param(lambda: pd.Series([1, 2]), id="not-numpy-at-all"),
]

STORED_BY_THE_ENCODER = [
    pytest.param(lambda: np.datetime64("2024-01-02"), id="datetime"),
    pytest.param(lambda: np.float16(1.5), id="float16"),
    pytest.param(lambda: np.array([[1, 2], [3, 4]]), id="int-matrix"),
    pytest.param(lambda: np.array([1.5, 2.5], dtype=np.float32), id="float32-array"),
    pytest.param(lambda: np.array(["a", "b"]), id="str-array"),
    pytest.param(lambda: np.array([1, None], dtype=object), id="object-array"),
    pytest.param(
        lambda: np.array([Decimal("1.25")], dtype=object), id="decimal-in-object-array"
    ),
    pytest.param(lambda: np.array([{None: 1}], dtype=object), id="none-key"),
    pytest.param(lambda: np.array([{(1, "a"): 3}], dtype=object), id="tuple-key"),
    pytest.param(
        lambda: np.array([{datetime(2024, 1, 2): 1}], dtype=object), id="datetime-key"
    ),
    pytest.param(
        lambda: np.array([{uuid.UUID(int=7): 1}], dtype=object), id="uuid-key"
    ),
]


def _model_payload(warehouse, value):
    return json.loads(
        warehouse.convert_type(
            NumpyHolder(name="a", payload={"v": value}),
            JSON(),
            warehouse.python_type(JSON()),
            "JSON",
            "test_column",
        )
    )["payload"]


@pytest.mark.parametrize("make", REFUSED_BY_THE_ENCODER)
def test_a_model_refuses_the_numpy_a_plain_value_refuses(test_session, make):
    warehouse = test_session.catalog.warehouse

    with pytest.raises(TypeError) as plain:
        dcjson.dumps({"v": make()}, serialize_numpy=True)

    with pytest.raises(TypeError) as wrapped:
        _model_payload(warehouse, make())

    assert str(wrapped.value) == str(plain.value)


@pytest.mark.parametrize("make", STORED_BY_THE_ENCODER)
def test_a_model_stores_the_numpy_a_plain_value_stores(test_session, make):
    warehouse = test_session.catalog.warehouse

    plain = json.loads(dcjson.dumps({"v": make()}, serialize_numpy=True))

    assert _model_payload(warehouse, make()) == plain


NON_FINITE_NUMPY = [
    pytest.param(lambda: np.array([1.0, np.nan], dtype=np.float32), id="nan-in-array"),
    pytest.param(lambda: np.array([np.inf]), id="inf-in-array"),
    pytest.param(lambda: np.array([-np.inf]), id="negative-inf-in-array"),
    pytest.param(lambda: np.float32("nan"), id="nan-scalar"),
    pytest.param(
        lambda: np.array([{"k": float("inf")}], dtype=object), id="inf-in-object-array"
    ),
]


@pytest.mark.parametrize("make", NON_FINITE_NUMPY)
def test_a_model_refuses_numpy_that_would_be_stored_as_null(test_session, make):
    warehouse = test_session.catalog.warehouse

    with pytest.raises(ValueError, match="writes NaN and infinities as null"):
        _model_payload(warehouse, make())


@pytest.mark.parametrize(
    "value,item_type,expected",
    [
        pytest.param((1, None), Int64, [1, None], id="int-none-last"),
        pytest.param((None, 2), Int64, [None, 2], id="int-none-first"),
        pytest.param((None, None), Int64, [None, None], id="int-all-none"),
        pytest.param([1, None], Float, [1.0, None], id="float-none-last"),
        pytest.param([None, 2], Float, [None, 2.0], id="float-none-first"),
        pytest.param(["a", None], String, ["a", None], id="str-none-last"),
        pytest.param([None, "b"], String, [None, "b"], id="str-none-first"),
    ],
)
def test_convert_type_keeps_none_wherever_it_sits_in_an_array(
    test_session, value, item_type, expected
):
    warehouse = test_session.catalog.warehouse
    col_type = Array(SQLType.as_nullable(item_type))

    converted = warehouse.convert_type(
        value,
        col_type,
        warehouse.python_type(col_type),
        "Array",
        "test_column",
    )

    assert converted == expected


def test_convert_type_stores_a_json_array_the_same_wherever_none_sits(test_session):
    warehouse = test_session.catalog.warehouse
    col_type = Array(JSON())

    def to_db(value):
        return warehouse.convert_type(
            value, col_type, warehouse.python_type(col_type), "Array", "test_column"
        )

    # What an item becomes is the backend's business; that its neighbours do not
    # change the answer is not.
    assert to_db([{"k": 1}, None]) == list(reversed(to_db([None, {"k": 1}])))


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([None, 2], id="none-first"),
        pytest.param([1, None], id="none-last"),
    ],
)
def test_convert_type_refuses_none_in_a_non_nullable_array(test_session, value):
    warehouse = test_session.catalog.warehouse
    col_type = Array(Int64)

    with pytest.raises(ValueError, match="incompatible"):
        warehouse.convert_type(
            value, col_type, warehouse.python_type(col_type), "Array", "test_column"
        )


class NestedItem(BaseModel):
    n: int


@pytest.mark.parametrize(
    "value",
    [
        pytest.param((None, {"a": NestedItem(n=2)}), id="none-first"),
        pytest.param(({"a": NestedItem(n=2)}, None), id="none-last"),
    ],
)
def test_convert_type_leaves_no_model_in_a_json_array_holding_none(test_session, value):
    warehouse = test_session.catalog.warehouse
    col_type = Array(JSON())

    converted = warehouse.convert_type(
        value, col_type, warehouse.python_type(col_type), "Array", "test_column"
    )

    # Whatever shape the backend asks for, nothing unserializable may survive.
    json.dumps(converted)


class SubclassedInt(Int64):
    pass


@pytest.mark.parametrize(
    "annotation,value",
    [
        pytest.param(list[int | None], [1, None], id="plain-int"),
        pytest.param(list[int | None], [None, 2], id="plain-int-none-first"),
        pytest.param(
            list[Annotated[int, "meta"] | None], [1, None], id="annotated-int"
        ),
        pytest.param(list[Literal["a", "b"] | None], ["a", None], id="literal-str"),
        pytest.param(
            list[Literal["a", "b"] | None], [None, "b"], id="literal-str-none-first"
        ),
        pytest.param(
            list[Annotated[int | None, "meta"]], [1, None], id="optional-in-annotated"
        ),
        pytest.param(
            list[Annotated[int | None, "meta"]],
            [None, 2],
            id="optional-in-annotated-none-first",
        ),
        pytest.param(
            list[Annotated[Literal["a", None], "meta"]],  # noqa: PYI061
            ["a", None],
            id="none-among-literal-values",
        ),
        pytest.param(list[SubclassedInt | None], [1, None], id="subclassed-sql-type"),
        pytest.param(
            list[str | Literal[None]],  # noqa: PYI061
            ["a", None],
            id="literal-none-in-a-union",
        ),
        pytest.param(
            list[Annotated[str, "meta"] | Literal[None]],  # noqa: PYI061
            ["x", None],
            id="annotated-beside-literal-none",
        ),
        pytest.param(
            tuple[str, Literal[None]],  # noqa: PYI061
            ("a", None),
            id="null-only-literal-slot",
        ),
        pytest.param(
            list[Literal[None]],  # noqa: PYI061
            [None],
            id="null-only-literal-item",
        ),
        pytest.param(tuple[int, int | None], (1, None), id="second-tuple-slot"),
        pytest.param(tuple[int | None, int], (None, 1), id="first-tuple-slot"),
    ],
)
def test_convert_type_keeps_none_for_a_wrapped_nullable_scalar(
    test_session, annotation, value
):
    warehouse = test_session.catalog.warehouse
    col_type = python_to_sql(annotation)

    converted = warehouse.convert_type(
        value, col_type, warehouse.python_type(col_type), "Array", "test_column"
    )

    assert converted == list(value)


def test_convert_type_json_encodes_an_all_none_array(test_session):
    warehouse = test_session.catalog.warehouse
    col_type = Array(SQLType.as_nullable(JSON()))

    converted = warehouse.convert_type(
        (None, None), col_type, warehouse.python_type(col_type), "Array", "test_column"
    )

    # An array with no object in it is not an array of objects; each None stays
    # whatever JSON writes for one, as it did before.
    assert converted == [json.dumps(None)] * 2


class SlotA(BaseModel):
    x: int


class SlotB(BaseModel):
    y: int


def test_python_to_sql_refuses_a_tuple_slot_that_reads_back_wrong():
    # Every slot of a fixed tuple shares the column, so one that cannot be read
    # back as a model has to refuse the whole annotation rather than be stored
    # and handed back as a plain dict.
    with pytest.raises(TypeError):
        python_to_sql(tuple[SlotA | None, Annotated[SlotB, "meta"] | None])

    assert python_to_sql(tuple[SlotA | None, SlotB | None]).to_dict() == {
        "type": "Array",
        "item_type": {"type": "JSON"},
    }
