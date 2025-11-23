import datetime as dt
import io

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
