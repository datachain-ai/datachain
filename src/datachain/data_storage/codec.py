"""The one conversion between Python values and a warehouse's stored values: a
`TypeCodec` per SQL type (`AbstractWarehouse.type_codec`), compiled per row of
columns into a `RowCodec`. Codecs hold functions and partials only, so they pickle."""

import math
import numbers
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from copy import deepcopy
from datetime import datetime
from functools import partial
from itertools import islice
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import BaseModel

from datachain import json
from datachain.lib.model_store import ModelStore

if TYPE_CHECKING:
    import pyarrow as pa

DECODE_BATCH_SIZE = 16
_IMMUTABLE_DEFAULT_TYPES = frozenset(
    {type(None), str, bytes, bool, int, float, datetime}
)


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


def _fresh_default(default: Any) -> Any:
    # deepcopy treats float as atomic; NaN identity affects container equality.
    if type(default) is float and math.isnan(default):
        return float("nan")
    return deepcopy(default)


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
            (
                i,
                c.decode or _identity,
                c.default,
                type(c.default) not in _IMMUTABLE_DEFAULT_TYPES
                or (type(c.default) is float and math.isnan(c.default)),
            )
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
        if not self.codecs:
            return [()] * n
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
        for i, decode, default, copy_default in self._decoded:
            if copy_default:
                columns[i] = [
                    _fresh_default(default) if v is None else decode(v)
                    for v in columns[i]
                ]
            else:
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


def check_list(
    item_type: type,
    item: Callable[[Any], Any],
    value: Any,
    *,
    json_array_text: bool = False,
) -> list:
    """A list, tuple, set or array as a list. Every item is checked by `item`,
    unless all already are `item_type`; a None item stays None. JSON items are
    made JSON-ready one by one."""
    value = _numpy_to_python(value)
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(  # noqa: TRY004
            f"Value {value!r} with type {type(value)} incompatible for "
            "column type Array"
        )
    if item_type is dict:
        first = next((v for v in value if v is not None), None)
        if json_array_text or (first is not None and not isinstance(first, dict)):
            # Array(JSON) stores each item as JSON text when json_array_text=True
            # or in the legacy non-dict shape; "null" decodes to None.
            return [encode_json(v, array_json_key) for v in value]
        return [None if v is None else to_jsonable(v, array_json_key) for v in value]
    value = list(value)
    first = next((v for v in value if v is not None), None)
    keep_compatible = item_type is not list and isinstance(first, item_type)
    if item_type is not list:
        # the types of every item, not of the first: one item cannot answer
        # for the rest
        kinds = set(map(type, value)) - {type(None)}
        if kinds <= {item_type}:
            return value
        if item_type is float and kinds <= {int, float}:
            return (
                value
                if keep_compatible
                else [v if v is None else float(v) for v in value]
            )
    checked = [None if v is None else item(v) for v in value]
    return value if keep_compatible else checked


def array_json_key(key: Any) -> str:
    if isinstance(key, str):
        return str.__str__(key)
    if key is None:
        return "null"
    if isinstance(key, bool):
        return "true" if key else "false"
    if isinstance(key, bytes):
        return bytes.decode(key)
    return str(key)


def _python_dump(obj: BaseModel) -> Any:
    try:
        return obj.model_dump()
    except TypeError:
        return {k: getattr(obj, k) for k in type(obj).model_fields}


def _check_merged_keys(obj: Any, dumped: Any) -> None:
    """Raises when the JSON dump of a model merged two keys of one of its dicts."""
    obj = _numpy_to_python(obj)
    if isinstance(obj, BaseModel):
        obj = _python_dump(obj)
    if isinstance(obj, dict) and isinstance(dumped, dict):
        if len(obj) != len(dumped):
            from datachain.lib.udf import JsonSerializationError

            raise JsonSerializationError(
                "Two keys serialize to the same JSON key, so one of their values "
                "would be lost.",
                column_name="",
                value_repr=repr(obj),
            )
        for value, dumped_value in zip(obj.values(), dumped.values(), strict=True):
            _check_merged_keys(value, dumped_value)
    elif (
        isinstance(obj, (list, tuple, set))
        and isinstance(dumped, list)
        and len(obj) == len(dumped)
    ):
        for value, dumped_value in zip(obj, dumped, strict=True):
            _check_merged_keys(value, dumped_value)


def to_jsonable(obj: Any, key_converter: Callable[[Any], str] | None = None) -> Any:
    """Python and pydantic structures as JSON-serializable objects. Raises
    JsonSerializationError when two keys of a dict become one JSON key."""
    obj = _numpy_to_python(obj)  # an object array can hold dicts
    if ModelStore.is_pydantic(type(obj)):
        # Use Pydantic's JSON mode to ensure datetime and other non-JSON
        # native types are serialized in a compatible way. Numpy is the one
        # thing it refuses and this project's encoder handles.
        dumped = obj.model_dump(mode="json", fallback=json.numpy_to_python)
        _check_merged_keys(_python_dump(obj), dumped)
        return dumped

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        items = dict.items(obj) if key_converter is array_json_key else obj.items()
        for k, v in items:
            # str.__str__: the key json writes for a str subclass (an enum)
            key = (
                key_converter(k)
                if key_converter is not None
                else str.__str__(k)
                if isinstance(k, str)
                else to_json_text(k)
            )
            if key in out:
                from datachain.lib.udf import JsonSerializationError

                raise JsonSerializationError(
                    f"Two keys serialize to the JSON key {key!r}, so one of "
                    "their values would be lost.",
                    column_name="",
                    value_repr=repr(obj),
                )
            out[key] = to_jsonable(v, key_converter)
        return out

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(i, key_converter) for i in obj]

    return obj


def to_json_text(value: Any) -> str:
    """A value as the JSON text a JSON column stores."""
    return json.dumps(to_jsonable(value), ensure_ascii=False, serialize_numpy=True)


def encode_json(value: Any, key_converter: Callable[[Any], str] | None = None) -> str:
    from datachain.lib.udf import JsonSerializationError

    value = _numpy_to_python(value)
    if type(value) is str:
        return value
    json_ready = to_jsonable(value, key_converter)
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
    if isinstance(value, (int, numbers.Integral)):
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


def decode_list(item: Callable[[Any], Any] | None, value: Any) -> Any:
    """A list with each item decoded by `item`; a None item stays None."""
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, np.ndarray):
        value = list(value)
    if not isinstance(value, (list, tuple)):
        return value
    if item is None or (item is decode_float and set(map(type, value)) <= {float, int}):
        return list(value)  # items read back as they are
    return [None if v is None else item(v) for v in value]


def compose(outer: Callable, inner: Callable, value: Any) -> Any:
    return outer(inner(value))


def base_decode(
    sql_type: Any, decode_of: Callable[[Any], Callable[[Any], Any] | None]
) -> Callable[[Any], Any] | None:
    """How `sql_type` reads back; `decode_of` decodes an array's items."""
    from datachain.sql.types import JSON, Array, Binary, Boolean, DateTime, Float

    if isinstance(sql_type, Array):
        return partial(decode_list, decode_of(sql_type.item_type))
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


def check_value(sql_type: Any, *, json_array_text: bool = False) -> Callable:
    """The check and normalization of a Python value for `sql_type`."""
    py_type = sql_type.python_type
    if py_type is list:
        item_type = sql_type.item_type
        return partial(
            check_list,
            item_type.python_type,
            check_value(item_type, json_array_text=json_array_text),
            json_array_text=json_array_text,
        )
    if py_type is dict:
        return encode_json
    return partial(check_scalar, py_type, type(sql_type).__name__)


def value_codec(
    sql_type: Any,
    default: Any,
    decode_of: Callable[[Any], Callable[[Any], Any] | None] = default_decode,
    *,
    json_array_text: bool = False,
) -> TypeCodec:
    """`sql_type`'s codec for a warehouse that stores Python values as they are."""
    from datachain.sql.types import SQLType

    if not isinstance(sql_type, SQLType):
        return PASSTHROUGH
    py_type = sql_type.python_type
    encode = check_value(sql_type, json_array_text=json_array_text)
    if py_type is list:
        kept: frozenset[type] = frozenset()
    elif encode is encode_json:
        kept = frozenset({str})
    else:
        kept = frozenset({py_type})
    return TypeCodec(kept, encode, default, decode_of(sql_type))
