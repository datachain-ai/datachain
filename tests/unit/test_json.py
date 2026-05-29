import datetime as dt
import io
import sys

import numpy as np
import pytest
from pydantic import BaseModel

from datachain import json


class DatePayload(BaseModel):
    timestamp: dt.datetime
    clock: dt.time
    date: dt.date


class Unexpected:
    pass


@pytest.mark.parametrize("serialize_bytes", [False, True])
def test_unhandled_type_raises_typeerror(serialize_bytes: bool) -> None:
    with pytest.raises(TypeError, match="Unexpected"):
        json.dumps({"value": Unexpected()}, serialize_bytes=serialize_bytes)


@pytest.mark.parametrize("payload", [b"\x00\x01", bytearray(b"\xff\x7f")])
def test_bytes_serialization_enabled(payload: bytes | bytearray) -> None:
    encoded = json.loads(json.dumps({"payload": payload}, serialize_bytes=True))
    assert encoded["payload"] == list(payload)


@pytest.mark.parametrize("payload", [b"\x00\x01", bytearray(b"\xff\x7f")])
def test_bytes_serialization_disabled(payload: bytes | bytearray) -> None:
    with pytest.raises(TypeError):
        json.dumps({"payload": payload})


def test_bytes_serialization_truncates_preview() -> None:
    payload = bytes(range(256)) * 5  # 1280 bytes
    encoded = json.loads(json.dumps({"payload": payload}, serialize_bytes=True))
    assert encoded["payload"] == list(payload)[: json.DEFAULT_PREVIEW_BYTES]


def test_dump_serialize_bytes_writes_expected_stream() -> None:
    buffer = io.StringIO()
    json.dump({"payload": b"abc"}, buffer, serialize_bytes=True)
    buffer.seek(0)
    assert json.loads(buffer.read()) == {"payload": [97, 98, 99]}


def test_datetime_serialization_matches_pydantic_json_mode() -> None:
    payload = DatePayload(
        timestamp=dt.datetime(2024, 1, 1, 12, 30, tzinfo=dt.timezone.utc),
        clock=dt.time(6, 45, tzinfo=dt.timezone.utc),
        date=dt.date(2024, 1, 2),
    )
    expected = payload.model_dump(mode="json")
    assert (
        json.loads(
            json.dumps(
                {
                    "timestamp": payload.timestamp,
                    "clock": payload.clock,
                    "date": payload.date,
                }
            )
        )
        == expected
    )


def test_numpy_serialization_is_opt_in() -> None:
    with pytest.raises(TypeError, match="ndarray"):
        json.dumps({"payload": np.array([1, 2])})


def test_numpy_serialization_handles_nested_values() -> None:
    payload = np.empty(2, dtype=object)
    payload[0] = np.array([1, 2], dtype=np.int64)
    payload[1] = {np.int64(7): np.float32(0.5)}

    assert json.loads(
        json.dumps({"payload": payload}, serialize_bytes=True, serialize_numpy=True)
    ) == {"payload": [[1, 2], {"7": 0.5}]}


def test_numpy_serialization_does_not_import_numpy_for_other_values(
    monkeypatch,
) -> None:
    monkeypatch.delitem(sys.modules, "numpy", raising=False)

    with pytest.raises(TypeError, match="Unexpected"):
        json.dumps({"payload": Unexpected()}, serialize_numpy=True)

    assert "numpy" not in sys.modules


def test_numpy_serialization_preserves_tuple_dict_keys_for_encoder() -> None:
    payload = np.empty(1, dtype=object)
    payload[0] = {(np.int64(1), np.int64(2)): "value"}

    with pytest.raises(TypeError, match="keys must be"):
        json.dumps({"payload": payload}, serialize_bytes=True, serialize_numpy=True)


def test_numpy_serialization_handles_scalars_and_fast_array_path() -> None:
    encoded = json.loads(
        json.dumps(
            {
                "score": np.float32(0.5),
                "values": np.array([1, 2], dtype=np.int64),
            },
            serialize_numpy=True,
        )
    )

    assert encoded == {"score": 0.5, "values": [1, 2]}


def test_numpy_serialization_preserves_user_default() -> None:
    def default(value):
        if isinstance(value, Unexpected):
            return "handled"
        raise TypeError

    encoded = json.loads(
        json.dumps(
            {"payload": [np.array([1, 2], dtype=np.int64), Unexpected()]},
            default=default,
            serialize_numpy=True,
        )
    )

    assert encoded == {"payload": [[1, 2], "handled"]}


def test_dump_serialize_numpy_writes_expected_stream() -> None:
    buffer = io.StringIO()
    json.dump(
        {"payload": np.array([1, 2], dtype=np.int64)}, buffer, serialize_numpy=True
    )
    buffer.seek(0)
    assert json.loads(buffer.read()) == {"payload": [1, 2]}
