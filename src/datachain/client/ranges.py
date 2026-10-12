import asyncio
import operator
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, NamedTuple

from datachain.progress import tqdm

if TYPE_CHECKING:
    from .fsspec import RangeListedClient, ResultQueue

LIST_WORKERS = 16
SPLIT_PAGES = 4  # a range splits at most this many pages ahead


class Page(NamedTuple):
    """One listing request's reply."""

    rows: list[Any]  # File objects of the keys kept
    first: str | None  # the first object's key
    last: str | None  # the last key of any kind, delete markers included
    token: Any  # resumes after this page; None on the last page


def code_points(key: str) -> str:
    return key


def utf16_units(key: str) -> bytes:
    return key.encode("utf-16-be", "surrogatepass")


class KeyRange:
    """The read state of a range ``(lo, hi]``. Keys compare in ``order``, the
    server's order."""

    def __init__(self, hi: str | None, order: Callable[[str], Any]) -> None:
        self.order = order
        self.hi = hi

    @property
    def hi(self) -> str | None:
        return self._hi

    @hi.setter
    def hi(self, hi: str | None) -> None:
        self._hi = hi
        self.hi_key = None if hi is None else self.order(hi)

    def covers(self, key: str) -> bool:
        return self.hi_key is None or self.order(key) <= self.hi_key


def key_midpoint(
    lo: str,
    hi: str | None,
    alphabet: Iterable[str],
    first: str | None = None,
    prefix: str = "",
    order: Callable[[str], Any] = code_points,
) -> str | None:
    """Shortest key strictly between ``lo`` and ``hi`` in ``order`` (``None`` is
    the end of ``prefix``); with ``first``, at most ``SPLIT_PAGES`` pages of
    ``first..lo`` past ``lo``."""
    chars = sorted(
        set(alphabet) | set(lo) | set(hi or "") | set(first or "") | set(prefix),
        key=order,
    )
    digit = {c: i + 1 for i, c in enumerate(chars)}
    base = len(chars) + 1
    width = max(len(lo), len(hi or ""), len(first or ""), len(prefix)) + 1

    def value(key: str) -> int:
        n = 0
        for i in range(width):
            n = n * base + (digit[key[i]] if i < len(key) else 0)
        return n

    end = value(prefix) + base ** (width - len(prefix)) if hi is None else value(hi)
    n = (value(lo) + end) // 2
    if first is not None and order(first) < order(lo):
        n = min(n, value(lo) + (value(lo) - value(first)) * SPLIT_PAGES)
    digits = []
    for _ in range(width):
        n, d = divmod(n, base)
        digits.append(chars[max(d - 1, 0)])
    mid = "".join(reversed(digits))
    mid = next(
        (mid[:k] for k in range(1, len(mid) + 1) if order(mid[:k]) > order(lo)), mid
    )
    inside = order(lo) < order(mid) and (hi is None or order(mid) < order(hi))
    return mid if inside and mid.startswith(prefix) else None


class RangeListing:
    """One listing of a prefix as key ranges ``(lo, hi]`` read in parallel.

    While a worker is idle, a busy range hands over its upper part, split at a key
    midpoint (``key_midpoint``). It keeps these rules:

    - the ranges cover the listing without overlap;
    - ``hi`` only shrinks;
    - a token advances only after its page is accepted.
    """

    def __init__(
        self,
        client: "RangeListedClient",
        prefix: str,
        result_queue: "ResultQueue",
    ) -> None:
        self.client = client
        self.prefix = prefix
        self.result_queue = result_queue
        self.order = client.key_order
        self.queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()  # lo, hi
        self.queue.put_nowait(("", None))
        self.alphabet: set[str] = set()
        self.open_ranges = 1
        self.listed_any = False
        self.server_seeks = True
        self.server_sorts = True
        self.has_split = False
        self.pbar = tqdm(desc=f"Listing {client.uri}", unit=" objects", leave=False)

    async def run(self) -> None:
        workers = [asyncio.create_task(self._worker()) for _ in range(LIST_WORKERS)]
        all_read = asyncio.create_task(self.queue.join())
        try:
            await asyncio.wait(
                [all_read, *workers], return_when=asyncio.FIRST_COMPLETED
            )
            if failed := next((w for w in workers if w.done()), None):
                await failed
            if not self.listed_any and self.prefix:
                raise FileNotFoundError(f"Unable to resolve remote path: {self.prefix}")
        finally:
            tasks = [all_read, *workers]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.pbar.close()

    async def _worker(self) -> None:
        while True:
            lo, hi = await self.queue.get()
            try:
                await self._read(lo, KeyRange(hi, self.order))
            finally:
                self.open_ranges -= 1
                self.queue.task_done()

    async def _fetch(self, lo: str, token: Any, key_range: KeyRange) -> Page:
        return await self.client._list_page(
            self.prefix, lo, token, self.client._keeps(key_range.hi), hi=key_range.hi
        )

    async def _read(self, lo: str, key_range: KeyRange) -> None:
        order = self.order
        may_split = bool(lo)  # a one-page listing stays one request
        lo_key = order(lo)
        previous_key = None
        page = await self._fetch(lo, None, key_range)
        next_page: asyncio.Task | None = None
        try:
            while True:
                read_to, token = page.last, page.token
                # splits follow the keys kept; any object makes the prefix found
                page_start = page.rows[0].path if page.rows else None
                keys = [order(r.path) for r in page.rows]
                if keys and not (
                    (previous_key is None or previous_key <= keys[0])
                    and all(map(operator.le, keys, keys[1:]))
                ):
                    self._out_of_order()
                previous_key = keys[-1] if keys else previous_key
                if page_start is not None and order(page_start) < lo_key:
                    self.server_seeks = False  # the server ignores the start
                self.listed_any = self.listed_any or page.first is not None
                # GCS startOffset and Azure start_from include the start key
                hi_key = key_range.hi_key
                rows = [
                    r
                    for r, key in zip(page.rows, keys, strict=True)
                    if lo_key < key and (hi_key is None or key <= hi_key)
                ]
                if read_to is not None and not key_range.covers(read_to):
                    token = None
                elif read_to is not None and order(read_to) > lo_key:
                    self.alphabet.update(read_to)
                    in_key_order = (
                        self.server_seeks
                        and self.server_sorts
                        and self.client._lists_in_key_order()
                    )
                    if token is not None and in_key_order and may_split:
                        self._maybe_split(key_range, read_to, page_start)
                    may_split = True
                if token is not None:
                    next_page = asyncio.create_task(self._fetch(lo, token, key_range))
                if rows:
                    await self.result_queue.put(rows)
                    self.pbar.update(len(rows))
                if next_page is None:
                    return
                page, next_page = await next_page, None
        finally:
            if next_page is not None:
                next_page.cancel()
                await asyncio.gather(next_page, return_exceptions=True)

    def _out_of_order(self) -> None:
        """Keys came back out of the server's order, so ranges no longer split. A
        range split earlier may have ended before keys it holds."""
        if self.has_split:
            raise RuntimeError(
                f"Listing {self.client.uri}: keys came back out of order after the"
                " listing split, so it could be incomplete"
            )
        self.server_sorts = False

    def _hand_over(self, key_range: KeyRange, lo: str, hi: str | None) -> None:
        """``(lo, hi]`` becomes a range of its own; ``key_range`` ends at ``lo``."""
        self.queue.put_nowait((lo, hi))
        key_range.hi = lo

    def _maybe_split(
        self, key_range: KeyRange, read_to: str, page_start: str | None
    ) -> None:
        if self.open_ranges >= LIST_WORKERS:
            return
        mid = key_midpoint(
            read_to, key_range.hi, self.alphabet, page_start, self.prefix, self.order
        )
        if mid is not None:
            self.open_ranges += 1
            self.has_split = True
            self._hand_over(key_range, mid, key_range.hi)
