import asyncio
import unicodedata
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, NamedTuple

from datachain.progress import tqdm
from datachain.utils import getenv_bool

if TYPE_CHECKING:
    from .fsspec import RangeListedClient, ResultQueue
    from .listing_worker import Pool

# A range has one request in flight, so these cap both open ranges and requests.
LIST_WORKERS = 16  # while this process fetches every page
PROCESS_LIST_WORKERS = 64  # once child processes fetch pages: they parse in parallel
# Child processes start once a listing has this many rows; below it one process
# keeps up with the network, and children would only cost CPU and memory.
POOL_ROWS = 100_000
SPLIT_PAGES = 4  # a young range splits at most this many pages ahead
YOUNG_PAGES = 4  # a range is young until it and its parents read this many pages


class Page(NamedTuple):
    """One listing request's reply."""

    rows: list[Any]  # FileRow rows of the keys kept
    first: str | None  # the first object's key; None when the page has none
    last: str | None  # the last key of any kind, delete markers included
    token: Any  # resumes after this page; None on the last page
    past_lo: str | None = None  # the first key listed past the start key
    past_hi: str | None = None  # the first key listed past ``hi``
    in_order: bool = True  # the keys came in the server's order


class Keep:
    """Which listed keys become rows: valid keys up to ``hi``. Notes the first key
    past ``lo``, the first past ``hi``, and whether the keys came in order."""

    def __init__(self, client: "RangeListedClient", lo: str, hi: str | None) -> None:
        self.order, self.valid = client.key_order, client._is_valid_key
        self.lo_key = self.order(lo)
        self.hi_key = None if hi is None else self.order(hi)
        self.past_lo: str | None = None
        self.past_hi: str | None = None
        self.in_order = True
        self.previous: Any = None

    def __call__(self, key: str) -> bool:
        order_key = self.order(key)
        if self.previous is not None and order_key < self.previous:
            self.in_order = False
        self.previous = order_key
        if self.past_lo is None and order_key > self.lo_key:
            self.past_lo = key
        if self.hi_key is None or order_key <= self.hi_key:
            return self.valid(key)
        self.past_hi = self.past_hi or key
        return False


def code_points(key: str) -> str:
    return key


def utf16_units(key: str) -> bytes:
    return key.encode("utf-16-be", "surrogatepass")


class KeyRange:
    """A range ``(lo, hi]`` of keys in ``order``, the server's order."""

    def __init__(
        self, lo: str, hi: str | None, order: Callable[[str], Any], age: int
    ) -> None:
        self.lo = lo
        self.hi = hi
        self.order = order
        self.age = age  # pages read by this range and the ranges it split off from
        self.lo_key = self.read_to = order(lo)  # read_to: the last key listed

    def covers(self, key: str) -> bool:
        return self.hi is None or self.order(key) <= self.order(self.hi)


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
    the end of ``prefix``). With ``first``, at most ``pages`` steps of
    ``first..lo`` past ``lo``, and none when under two steps are left."""
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
        step = value(lo) - value(first)
        if end - value(lo) < 2 * step:
            return None  # a split this close adds a request, not speed
        n = min(n, value(lo) + step * pages)
    digits = []
    for _ in range(width):
        n, d = divmod(n, base)
        digits.append(chars[max(d - 1, 0)])
    mid = "".join(reversed(digits))
    mid = next(
        (mid[:k] for k in range(1, len(mid) + 1) if order(mid[:k]) > order(lo)), mid
    )
    inside = order(lo) < order(mid) and (hi is None or order(mid) < order(hi))
    inside = inside and mid.startswith(prefix)
    # Azure starts wrongly at a mark or modifier right after the prefix
    after = unicodedata.category(mid[len(prefix) :][:1] or "a")
    return mid if inside and after not in ("Mn", "Me", "Cf", "Lm", "Sk") else None


def split_points(
    key_range: KeyRange,
    read_to: str,
    page_start: str | None,
    alphabet: Iterable[str],
    prefix: str = "",
) -> list[str]:
    """Where a range that read to ``read_to`` splits, farthest first. A young
    range splits close ahead, so a small listing stays a few requests. An old
    range splits at its age times as far as it read from ``lo``, which jumps over
    folders of a page or so, and then at its age squared pages ahead, which stays
    in data that ends or clusters."""
    if key_range.age <= YOUNG_PAGES:
        steps = [(page_start, SPLIT_PAGES)]
    else:  # the first range has no lo to measure from: its midpoint
        steps = [(key_range.lo or None, key_range.age), (page_start, key_range.age**2)]
    points: list[str] = []
    hi = key_range.hi
    for first, pages in steps:
        mid = key_midpoint(read_to, hi, alphabet, first, pages, prefix, key_range.order)
        if mid is not None:
            points.append(hi := mid)
    return points


class RangeListing:
    """One listing of a prefix as key ranges ``(lo, hi]`` read in parallel.

    When a page arrives and workers are idle, its range hands upper parts over
    (``split_points``). With a pool, child processes fetch pages; the parent owns
    every range. The fuzz tests check these rules:

    - the ranges cover the listing without overlap;
    - ``hi`` only shrinks, and only before the range's next page is asked for;
    - a token advances only after its page is accepted.
    """

    def __init__(
        self,
        client: "RangeListedClient",
        prefix: str,
        result_queue: "ResultQueue",
        pool: "Pool | None",
        delimiter: str = "",
    ) -> None:
        self.client = client
        self.prefix = prefix
        self.delimiter = delimiter  # with it, keys past a delimiter are skipped
        self.result_queue = result_queue
        self.pool = pool
        self.order = client.key_order
        self.ranges: asyncio.Queue[KeyRange] = asyncio.Queue()
        self.ranges.put_nowait(KeyRange("", None, self.order, 0))
        self.open_ranges = 1  # queued or being read
        self.max_ranges = PROCESS_LIST_WORKERS if pool and pool.size else LIST_WORKERS
        self.alphabet: set[str] = set()
        self.listed_any = False
        self.rows = 0
        # off when the server ignores start keys or lists keys out of order
        self.splits = getenv_bool("DATACHAIN_RANGE_LISTING", default=True)
        self.has_split = False
        # the first key past a range start: as read from below it, and from above
        self.seams: dict[str, list[str | None]] = {}
        self.own_requests = asyncio.Semaphore(LIST_WORKERS)  # this process's
        self.pbar = tqdm(desc=f"Listing {client.uri}", unit=" objects", leave=False)

    async def run(self) -> None:
        workers = [asyncio.create_task(self._worker()) for _ in range(self.max_ranges)]
        all_read = asyncio.create_task(self.ranges.join())
        try:
            await asyncio.wait(
                [all_read, *workers], return_when=asyncio.FIRST_COMPLETED
            )
            if failed := next((w for w in workers if w.done()), None):
                await failed
            if not self.listed_any and self.prefix:
                raise FileNotFoundError(f"Unable to resolve remote path: {self.prefix}")
        finally:
            for task in (all_read, *workers):
                task.cancel()
            if self.pool:  # before the await: gc may close this coroutine
                self.pool.close()
            # again until they end: a library that loses a cancel must not hang this
            pending = {all_read, *workers}
            while pending := (await asyncio.wait(pending, timeout=1))[1]:
                for task in pending:
                    task.cancel()
            for task in (all_read, *workers):
                if not task.cancelled():
                    task.exception()  # retrieved: the first failure is raised above
            if self.pool:
                await self.pool.wait_closed()
            self.pbar.close()

    async def _worker(self) -> None:
        while True:
            key_range = await self.ranges.get()
            try:
                await self._read(key_range)
            finally:
                self.open_ranges -= 1
                self.ranges.task_done()

    async def _fetch(self, key_range: KeyRange, token: Any) -> Page:
        lo, hi = key_range.lo, key_range.hi
        page = await self.pool.list_page(lo, token, hi) if self.pool else None
        if page is None:
            async with self.own_requests:
                # only a level listing passes it: a client that lists no levels
                # needn't take it
                kwargs: dict[str, Any] = (
                    {"delimiter": self.delimiter} if self.delimiter else {}
                )
                page = await self.client._list_page(
                    self.prefix, lo, token, hi, **kwargs
                )
        if token is not None and page.token == token:
            raise RuntimeError(
                f"Listing {self.client.uri}: the server returned the page token it"
                " was given, so the listing would never end."
            )
        if hi is not None and page.past_hi is not None:
            await self._seam(hi, 0, page.past_hi)
        if lo and (page.past_lo or page.token is None):
            await self._seam(lo, 1, page.past_lo or "")
        return page

    async def _read(self, key_range: KeyRange) -> None:
        """Reads the range a page at a time; the next page is on its way while
        this one waits for room in the result queue."""
        page = await self._fetch(key_range, None)
        next_page: asyncio.Task | None = None
        try:
            while True:
                rows, token = self._advance(key_range, page)
                if token is not None:
                    next_page = asyncio.create_task(self._fetch(key_range, token))
                if rows:
                    await self.result_queue.put(rows)
                    self.pbar.update(len(rows))
                    self.rows += len(rows)
                if next_page is None:
                    return
                page, next_page = await next_page, None
        finally:
            if next_page is not None:
                next_page.cancel()
                await asyncio.gather(next_page, return_exceptions=True)

    def _advance(self, key_range: KeyRange, page: Page) -> tuple[list[Any], Any]:
        """Checks and records what ``page`` read, and maybe splits the range.
        Returns the page's rows in the range and the token of the range's next
        page, or None at its end."""
        key_range.age += 1
        self.listed_any = self.listed_any or page.first is not None
        first = None if page.first is None else self.order(page.first)
        if first is not None and first < key_range.lo_key:
            self.splits = False  # the server ignores the start
        elif not page.in_order or (first is not None and first < key_range.read_to):
            self._out_of_order()
        rows = page.rows
        keys = [self.order(r.path) for r in rows]
        if keys and keys[0] <= key_range.lo_key:  # GCS and Azure list lo itself
            rows = [r for r, k in zip(rows, keys, strict=True) if k > key_range.lo_key]
        if (last := page.last) is None:
            return rows, page.token
        key_range.read_to = self.order(last)
        if not key_range.covers(last):
            return rows, None  # read past hi
        if key_range.read_to > key_range.lo_key:
            self.alphabet.update(last)
            past_first_page = bool(key_range.lo) or key_range.age > 1
            if page.token is not None and past_first_page and self._may_split():
                if self.pool and self.rows >= POOL_ROWS:
                    self.pool.start()
                page_start = page.rows[0].path if page.rows else None
                self._split(key_range, last, page_start)
        return rows, page.token

    def _may_split(self) -> bool:
        return self.splits and self.client._lists_in_key_order()

    def _split(self, key_range: KeyRange, read_to: str, page_start: str | None) -> None:
        """Hands the range's upper parts over to idle workers."""
        for mid in split_points(
            key_range, read_to, page_start, self.alphabet, self.prefix
        ):
            if self.open_ranges >= self.max_ranges:
                return
            self.open_ranges += 1
            self.has_split = True
            self.ranges.put_nowait(
                KeyRange(mid, key_range.hi, self.order, key_range.age)
            )
            key_range.hi = mid

    async def _seam(self, start: str, side: int, key: str) -> None:
        """The range above ``start`` (side 1) must see the first key past it that
        the range below (side 0) sees. A key written meanwhile can hide it, so the
        server is asked again before the listing fails."""
        seen = self.seams.setdefault(start, [None, None])
        if seen[side] is not None:
            return  # an earlier page saw it
        seen[side] = key
        below, above = seen
        if below is None or above is None:
            return
        if above and self.order(above) <= self.order(below):
            return  # nothing the range below saw is missing above
        async with self.own_requests:
            above = await self.client._probe(self.prefix, start) or ""
            skipped = not above or self.order(above) > self.order(below)
            # a key deleted meanwhile was not skipped
            skipped = skipped and await self.client._probe(below, "") == below
        seen[1] = above
        if skipped:
            raise RuntimeError(
                f"Listing {self.client.uri}: the server skipped keys after a start"
                " key, so the listing would be incomplete. DATACHAIN_RANGE_LISTING=0"
                " lists in one range."
            )

    def _out_of_order(self) -> None:
        """Keys came back out of the server's order, so ranges no longer split. A
        range split earlier may have ended before keys it holds."""
        if self.has_split:
            raise RuntimeError(
                f"Listing {self.client.uri}: keys came back out of order after the"
                " listing split, so it could be incomplete. DATACHAIN_RANGE_LISTING=0"
                " lists in one range."
            )
        self.splits = False
