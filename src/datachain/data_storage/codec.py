"""The one conversion between Python values and a warehouse's stored values: a
`TypeCodec` per SQL type (`AbstractWarehouse.type_codec`), compiled per row of
columns into a `RowCodec`. Codecs hold functions and partials only, so they pickle."""

import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import datetime
from functools import partial
from itertools import islice
from typing import TYPE_CHECKING, Any, NamedTuple

from datachain import json

if TYPE_CHECKING:
    import pyarrow as pa

DECODE_BATCH_SIZE = 1_000


class TypeCodec(NamedTuple):
    """`kept` value types are stored as they are, `encode` converts (or rejects)
    any other non-None value, None is stored and NULL read as `default`, `decode`
    reads a stored value (None: as it is); `encode_column` encodes an Arrow column."""

    kept: frozenset[type]
    encode: Callable[[Any], Any]
    default: Any = None
    decode: Callable[[Any], Any] | None = None
    encode_column: "Callable[[pa.Array], list[Any] | None] | None" = None


class RowEncodeError(ValueError):
    """A value its column cannot take; `column` is the column's index."""

    def __init__(self, column: int, value: Any) -> None:
        self.column = column
        self.value = value
        super().__init__(f"invalid value {value!r} for column {column}")

    def __reduce__(self):
        return RowEncodeError, (self.column, self.value)


def _identity(value: Any) -> Any:
    return value


# A column whose values are stored and read as they are (sys and non-DataChain types).
PASSTHROUGH = TypeCodec(kept=frozenset(), encode=_identity)


class RowCodec:
    """Encodes and decodes rows of a fixed list of columns, one `TypeCodec` each."""

    def __init__(self, codecs: Sequence[TypeCodec]) -> None:
        self.codecs = tuple(codecs)
        # values of these types are stored as they are, by column (None: all)
        self._kept = [
            None
            if c is PASSTHROUGH
            else c.kept | ({type(None)} if c.default is None else set())
            for c in self.codecs
        ]
        self._encoded = [
            (i, c, kept)
            for i, (c, kept) in enumerate(zip(self.codecs, self._kept, strict=True))
            if kept is not None
        ]
        self._decoded = [
            (i, c.decode or _identity, c.default)
            for i, c in enumerate(self.codecs)
            if c.decode is not None or c.default is not None
        ]

    @staticmethod
    def _encode_one(i: int, codec: TypeCodec, value: Any) -> Any:
        if type(value) in codec.kept:
            return value
        if value is None:
            return codec.default
        try:
            return codec.encode(value)
        except Exception as e:
            raise RowEncodeError(i, value) from e

    def encode(self, row: Sequence[Any]) -> tuple:
        """One row of leaf values, one per column, as stored values."""
        values = None
        for i, codec, kept in self._encoded:
            value = row[i]
            if type(value) in kept:
                continue
            if values is None:
                values = list(row)
            values[i] = self._encode_one(i, codec, value)
        if values is not None:
            return tuple(values)
        return row if type(row) is tuple else tuple(row)

    def encode_arrow(self, batch: "pa.RecordBatch") -> list[tuple]:
        """An Arrow batch of leaf columns as stored tuples; missing trailing
        columns are None."""
        n, columns = batch.num_rows, []
        for i, codec in enumerate(self.codecs):
            if i >= batch.num_columns:
                columns.append([codec.default] * n)
                continue
            column = batch.column(i)
            if hasattr(column, "dictionary_decode"):
                column = column.dictionary_decode()
            if codec.encode_column and (values := codec.encode_column(column)):
                if codec.default is not None:  # stored already, but for None
                    values = [codec.default if v is None else v for v in values]
            else:
                values = column.to_pylist()
                kept = self._kept[i]
                if kept is not None and not set(map(type, values)) <= kept:
                    values = [self._encode_one(i, codec, v) for v in values]
            columns.append(values)
        return list(zip(*columns, strict=True))

    def decode_columns(self, columns: list[Sequence[Any]]) -> list[Sequence[Any]]:
        """Stored columns, each a sequence of values, as Python values. A column
        that reads back as it is stays the same object."""
        for i, decode, default in self._decoded:
            columns[i] = [default if v is None else decode(v) for v in columns[i]]
        return columns

    def decode_rows(
        self, rows: Iterable[Sequence[Any]], batch_size: int = DECODE_BATCH_SIZE
    ) -> Iterator[Sequence[Any]]:
        """Stored rows as Python values, decoded column by column a batch at a time.
        Rows come back as they are when no column needs decoding."""
        if not self._decoded:
            yield from rows
            return
        rows = iter(rows)
        while batch := list(islice(rows, batch_size)):
            columns: list[Sequence[Any]] = list(zip(*batch, strict=True))
            yield from zip(*self.decode_columns(columns), strict=True)


def _numpy_to_python(value: Any) -> Any:
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value


def check_scalar(python_type: type, type_name: str, value: Any) -> Any:
    """A value of `python_type`, or an int for a float; raises on anything else."""
    value = _numpy_to_python(value)
    if isinstance(value, python_type):
        if isinstance(value, datetime) and type(value) is not datetime:
            # sqlite3 can't bind subclasses such as pandas.Timestamp.
            # This drops nanoseconds, as the ClickHouse driver does.
            return datetime.combine(value.date(), value.timetz())
        return value
    if python_type is float and isinstance(value, int):
        return float(value)
    raise ValueError(
        f"Value {value!r} with type {type(value)} incompatible for "
        f"column type {type_name}"
    )


def check_list(item_type: type, item: Callable[[Any], Any], value: Any) -> list:
    """A list, tuple, set or array as a list. Items are checked by `item`, unless
    the first one already is an `item_type`."""
    value = _numpy_to_python(value)
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(  # noqa: TRY004
            f"Value {value!r} with type {type(value)} incompatible for "
            "column type Array"
        )
    value = list(value)
    if value and item_type is not list:
        if isinstance(value[0], item_type):
            return value
        if item_type is float and isinstance(value[0], int):
            return [float(i) for i in value]
    return [item(v) for v in value]


def to_jsonable(obj: Any) -> Any:
    """Python and pydantic structures as JSON-serializable objects."""
    from datachain.lib.model_store import ModelStore

    if ModelStore.is_pydantic(type(obj)):
        # Use Pydantic's JSON mode to ensure datetime and other non-JSON
        # native types are serialized in a compatible way. Numpy is the one
        # thing it refuses and this project's encoder handles.
        return obj.model_dump(mode="json", fallback=json.numpy_to_python)

    if isinstance(obj, dict):
        return {
            k if isinstance(k, str) else to_json_text(k): to_jsonable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(i) for i in obj]

    return obj


def to_json_text(value: Any) -> str:
    """A value as the JSON text a JSON column stores."""
    return json.dumps(to_jsonable(value), ensure_ascii=False, serialize_numpy=True)


def encode_json(value: Any) -> str:
    from datachain.lib.udf import JsonSerializationError

    if type(value) is str:
        return value
    json_ready = to_jsonable(value)
    try:
        return json.dumps(json_ready, ensure_ascii=False, serialize_numpy=True)
    except TypeError as e:
        raise JsonSerializationError(
            f"JSON serialization error: {e}", column_name="", value_repr=repr(value)
        ) from e


def decode_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value) if value else {}
    return value


def decode_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0"}:
            return False
    return value


def decode_float(value: Any) -> Any:
    if isinstance(value, str) and value.lower() == "nan":
        return float("nan")
    return value


def decode_datetime(value: Any) -> Any:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(
        f"datetime decoder expected str, datetime, or None; got {type(value).__name__}"
    )


def decode_binary(value: Any) -> Any:
    return value.encode() if isinstance(value, str) else value


def decode_list(item: Callable[[Any], Any] | None, null: Any, value: Any) -> Any:
    """A list with each item decoded by `item`; a None item reads as `null`."""
    if not isinstance(value, (list, tuple)):
        return value
    if item is None or (item is decode_float and set(map(type, value)) <= {float, int}):
        return list(value)  # items read back as they are
    return [null if v is None else item(v) for v in value]


def compose(outer: Callable, inner: Callable, value: Any) -> Any:
    return outer(inner(value))


def base_decode(
    sql_type: Any, decode_of: Callable[[Any], Callable[[Any], Any] | None]
) -> Callable[[Any], Any] | None:
    """How `sql_type` reads back; `decode_of` decodes an array's items."""
    from datachain.sql.types import JSON, Array, Binary, Boolean, DateTime, Float

    if isinstance(sql_type, Array):
        item_type = sql_type.item_type
        null = float("nan") if isinstance(item_type, Float) else None
        return partial(decode_list, decode_of(item_type), null)
    for type_, decode in (
        (JSON, decode_json),
        (Boolean, decode_bool),
        (Float, decode_float),
        (DateTime, decode_datetime),
        (Binary, decode_binary),
    ):
        if isinstance(sql_type, type_):
            return decode
    return None


def default_decode(sql_type: Any) -> Callable[[Any], Any] | None:
    return base_decode(sql_type, default_decode)


def check_value(sql_type: Any) -> Callable:
    """The check and normalization of a Python value for `sql_type`."""
    py_type = sql_type.python_type
    if py_type is list:
        item_type = sql_type.item_type
        return partial(check_list, item_type.python_type, check_value(item_type))
    if py_type is dict or type(sql_type).__name__ == "JSON":
        return encode_json
    return partial(check_scalar, py_type, type(sql_type).__name__)


def value_codec(
    sql_type: Any,
    default: Any,
    decode_of: Callable[[Any], Callable[[Any], Any] | None] = default_decode,
) -> TypeCodec:
    """`sql_type`'s codec for a warehouse that stores Python values as they are."""
    from datachain.sql.types import SQLType

    if not isinstance(sql_type, SQLType):
        return PASSTHROUGH
    py_type = sql_type.python_type
    encode = check_value(sql_type)
    if py_type is list:
        kept: frozenset[type] = frozenset()
    elif encode is encode_json:
        kept = frozenset({str})
    else:
        kept = frozenset({py_type})
    return TypeCodec(kept, encode, default, decode_of(sql_type))
