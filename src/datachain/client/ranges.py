import asyncio
import logging
import operator
from collections.abc import Callable, Iterable
from contextlib import suppress
from typing import TYPE_CHECKING, Any, NamedTuple

from datachain.progress import tqdm

if TYPE_CHECKING:
    from .fsspec import RangeListedClient, ResultQueue
    from .listing_worker import Pool

logger = logging.getLogger("datachain")

LIST_WORKERS = 16
PROCESS_LIST_WORKERS = 64  # in flight once child processes parse the pages
SPLIT_PAGES = 4  # a young range splits at most this many pages ahead
YOUNG_PAGES = 4  # a range is young until it and its parents read this many pages


class Page(NamedTuple):
    """One listing request's reply."""

    rows: list[Any]  # FileRow rows of the keys kept
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

    def __init__(self, hi: str | None, order: Callable[[str], Any], age: int) -> None:
        self.order = order
        self.hi = hi
        self.read_to: str | None = None
        self.page_start: str | None = None  # the first key kept from the last page
        self.age = age  # pages read by this range and the ranges it split off from
        self.probing = False  # a split waits for its probe
        self.done = False

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
    pages: int = SPLIT_PAGES,
    prefix: str = "",
    order: Callable[[str], Any] = code_points,
) -> str | None:
    """Shortest key strictly between ``lo`` and ``hi`` in ``order`` (``None`` is
    the end of ``prefix``); with ``first``, at most ``pages`` pages of
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
        n = min(n, value(lo) + (value(lo) - value(first)) * pages)
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


def split_point(
    key_range: KeyRange,
    read_to: str,
    page_start: str | None,
    alphabet: Iterable[str],
    prefix: str = "",
) -> str | None:
    """Where a range that read to ``read_to`` splits. A young range splits close
    ahead, so a small listing stays a few requests. An old range with an open end
    splits at most age squared pages ahead, so a large listing soon reaches its
    end; an old bounded range splits at its midpoint."""
    order = key_range.order
    if key_range.age <= YOUNG_PAGES:
        return key_midpoint(
            read_to, key_range.hi, alphabet, page_start, SPLIT_PAGES, prefix, order
        )
    if key_range.hi is None:
        return key_midpoint(
            read_to, None, alphabet, page_start, key_range.age**2, prefix, order
        )
    return key_midpoint(
        read_to, key_range.hi, alphabet, None, SPLIT_PAGES, prefix, order
    )


class RangeListing:
    """One listing of a prefix as key ranges ``(lo, hi]`` read in parallel.

    While a worker is idle, a busy range hands over its upper part, split at a key
    midpoint (``split_point``). With a pool, child processes fetch pages; the
    parent owns every range. The fuzz tests check these rules:

    - the ranges cover the listing without overlap;
    - ``hi`` only shrinks;
    - a finished range never splits;
    - a range records how far it read before it waits on the result queue;
    - a token advances only after its page is accepted.
    """

    def __init__(
        self,
        client: "RangeListedClient",
        prefix: str,
        result_queue: "ResultQueue",
        pool: "Pool | None",
    ) -> None:
        self.client = client
        self.prefix = prefix
        self.result_queue = result_queue
        self.pool = pool
        self.order = client.key_order
        # (lo, hi, age of the range it split off from)
        self.queue: asyncio.Queue[tuple[str, str | None, int]] = asyncio.Queue()
        self.queue.put_nowait(("", None, 0))
        self.alphabet: set[str] = set()
        self.open_ranges = 1
        self.max_ranges = PROCESS_LIST_WORKERS if pool and pool.size else LIST_WORKERS
        self.listed_any = False
        self.server_seeks = True
        self.server_sorts = True
        self.has_split = False
        self.local = asyncio.Semaphore(LIST_WORKERS)  # requests from this process
        self.probes: set[asyncio.Task] = set()
        self.pbar = tqdm(desc=f"Listing {client.uri}", unit=" objects", leave=False)

    async def run(self) -> None:
        workers = [asyncio.create_task(self._worker()) for _ in range(self.max_ranges)]
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
            tasks = [all_read, *workers, *self.probes]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.pool:
                self.pool.close()
            self.pbar.close()

    async def _worker(self) -> None:
        while True:
            lo, hi, age = await self.queue.get()
            key_range = KeyRange(hi, self.order, age)
            try:
                await self._read(lo, key_range)
            finally:
                key_range.done = True
                self.open_ranges -= 1
                self.queue.task_done()

    async def _fetch(self, lo: str, token: Any, key_range: KeyRange) -> Page:
        from .listing_worker import ChildDied

        while self.pool and self.pool.ready():
            with suppress(ChildDied):
                return await self.pool.list_page(self.prefix, lo, token, key_range.hi)
        async with self.local:
            return await self.client._list_page(
                self.prefix,
                lo,
                token,
                self.client._keeps(key_range.hi),
                hi=key_range.hi,
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
                key_range.age += 1
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
                if read_to is not None:  # before the put: a probe may return meanwhile
                    key_range.read_to = max(
                        key_range.read_to or read_to, read_to, key=order
                    )
                    key_range.page_start = page_start
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
                    if token is not None and in_key_order:
                        if self.pool and (lo or key_range.age > 1):
                            self.pool.start()  # more than a page: worth a process
                        if may_split and not key_range.probing:
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
        self.queue.put_nowait((lo, hi, key_range.age))
        key_range.hi = lo

    def _maybe_split(
        self, key_range: KeyRange, read_to: str, page_start: str | None
    ) -> None:
        if self.open_ranges >= self.max_ranges:
            return
        mid = split_point(key_range, read_to, page_start, self.alphabet, self.prefix)
        if mid is None:
            return
        self.open_ranges += 1
        self.has_split = True
        young = key_range.age <= YOUNG_PAGES
        bounded = key_range.hi is not None
        if young or (bounded and not self.client.PROBE_BOUNDED_SPLITS):
            self._hand_over(key_range, mid, key_range.hi)
        else:
            key_range.probing = True
            self.probes.add(task := asyncio.create_task(self._split(mid, key_range)))
            task.add_done_callback(self.probes.discard)

    async def _split(self, mid: str, key_range: KeyRange) -> None:
        """Open ``(mid, hi]`` if a probe finds a key there. If not, the range ends
        at ``mid`` and splits where its keys are dense instead: keys that cluster
        leave most midpoints empty."""
        order = self.order
        try:
            async with self.local:
                key: str | None = await self.client._probe(self.prefix, mid)
        except Exception:  # errors come from many libraries
            logger.debug("Split probe failed, splitting unprobed", exc_info=True)
            key = mid
        finally:
            key_range.probing = False
        if (
            key_range.done
            or order(key_range.read_to or "") >= order(mid)
            or not key_range.covers(mid)
        ):
            self.open_ranges -= 1  # the range got there first
        elif key is not None and key_range.covers(key):
            self._hand_over(key_range, mid, key_range.hi)
        else:  # nothing past mid: no empty range to read
            key_range.hi = mid
            dense = key_midpoint(
                key_range.read_to or "",
                mid,
                self.alphabet,
                key_range.page_start,
                order=order,
            )
            if key_range.page_start is not None and dense is not None:
                self._hand_over(key_range, dense, mid)
            else:
                self.open_ranges -= 1
