from queue import Empty, Full
from time import sleep
from typing import Any

from multiprocess.queues import Queue

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
