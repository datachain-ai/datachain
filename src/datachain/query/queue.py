import datetime
from collections.abc import Iterable, Iterator
from queue import Empty, Full
from struct import pack, unpack
from time import sleep
from typing import Any

import msgpack
from multiprocess.queues import Queue

from datachain.query.batch import RowsOutput

DEFAULT_BATCH_SIZE = 10000
STOP_SIGNAL = "STOP"
OK_STATUS = "OK"
FINISHED_STATUS = "FINISHED"
FAILED_STATUS = "FAILED"
NOTIFY_STATUS = "NOTIFY"


# For more context on the get_from_queue and put_into_queue functions, see the
# discussion here:
# https://github.com/iterative/dvcx/pull/1297#issuecomment-2026308773
# This problem is not exactly described by, but is also related to these Python issues:
# https://github.com/python/cpython/issues/66587
# https://github.com/python/cpython/issues/88628
# https://github.com/python/cpython/issues/108645


def get_from_queue(queue: Queue) -> Any:
    """
    Gets an item from a queue.
    This is required to handle signals, such as KeyboardInterrupt exceptions
    while waiting for items to be available, although only on certain installations.
    (See the above comment for more context.)
    """
    while True:
        try:
            return queue.get_nowait()
        except Empty:
            sleep(0.01)


def put_into_queue(queue: Queue, item: Any) -> None:
    """
    Puts an item into a queue.
    This is required to handle signals, such as KeyboardInterrupt exceptions
    while waiting for items to be queued, although only on certain installations.
    (See the above comment for more context.)
    """
    while True:
        try:
            queue.put_nowait(item)
            return
        except Full:
            sleep(0.01)


MSGPACK_EXT_TYPE_DATETIME = 42


def _msgpack_pack_extended_types(obj: Any) -> msgpack.ExtType:
    if isinstance(obj, datetime.datetime):
        # packing date object as 1 or 2 variables, depending if timezone info is present
        #   - timestamp
        #   - [OPTIONAL] timezone offset from utc in seconds if timezone info exists
        if obj.tzinfo:
            data = (obj.timestamp(), int(obj.utcoffset().total_seconds()))  # type: ignore   # noqa: PGH003
            return msgpack.ExtType(MSGPACK_EXT_TYPE_DATETIME, pack("!dl", *data))
        data = (obj.timestamp(),)  # type: ignore   # noqa: PGH003
        return msgpack.ExtType(MSGPACK_EXT_TYPE_DATETIME, pack("!d", *data))

    raise TypeError(f"Unknown type: {obj}")


def msgpack_pack(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_msgpack_pack_extended_types)


def _msgpack_unpack_extended_types(code: int, data: bytes) -> Any:
    if code == MSGPACK_EXT_TYPE_DATETIME:
        has_timezone = False
        if len(data) == 8:
            # we send only timestamp without timezone if data is 8 bytes
            values = unpack("!d", data)
        else:
            has_timezone = True
            values = unpack("!dl", data)

        timestamp = values[0]
        tz_info = None
        if has_timezone:
            timezone_offset = values[1]
            tz_info = datetime.timezone(datetime.timedelta(seconds=timezone_offset))
        return datetime.datetime.fromtimestamp(timestamp, tz=tz_info)

    return msgpack.ExtType(code, data)


def msgpack_unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, ext_hook=_msgpack_unpack_extended_types)


def marshal(obj: Iterator[RowsOutput]) -> Iterable[bytes]:
    for row in obj:
        yield msgpack_pack(row)


def unmarshal(obj: Iterator[bytes]) -> Iterable[RowsOutput]:
    for row in obj:
        yield msgpack_unpack(row)
