import asyncio
import bisect
import os
import unicodedata
from collections.abc import Callable, Coroutine, Iterable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

# light imports only: a listing child process loads this module
if TYPE_CHECKING:
    from .fsspec import RangeListedClient, ResultQueue
    from .listing_worker import Child, Pool

# A range has one request in flight, so these cap both open ranges and requests.
LIST_WORKERS = 16  # while this process reads the ranges
CHILD_WORKERS = 16  # in each child process, once children read them
PROCESS_LIST_WORKERS = 64  # a child's requests: two per range at most
# Child processes start once a listing has this many rows; below it one process
# keeps up with the network, and children would only cost CPU and memory.
POOL_ROWS = 10_000
CUT_RANGES = 64  # the most ranges a listing starts at folders
SPLIT_PAGES = 4  # a young range splits at most this many pages ahead
YOUNG_PAGES = 4  # a range is young until it and its parents read this many pages


class Lister(Protocol):
    """What reading key ranges takes of a client: ``RangeListedClient``, or a
    lighter one in a listing child process."""

    @property
    def uri(self) -> str: ...

    def key_order(self, key: str) -> Any: ...

    def _is_valid_key(self, key: str) -> bool: ...

    def _lists_in_key_order(self) -> bool: ...

    async def _list_page(
        self, prefix: str, start_after: str, token: Any, hi: str | None = None, /
    ) -> "Page": ...


class Page(NamedTuple):
    """One listing request's reply."""

    rows: list[Any]  # FileRow rows of the keys kept
    first: str | None  # the first object's key; None when the page has none
    last: str | None  # the last key of any kind, delete markers included
    token: Any  # resumes after this page; None on the last page
    past_lo: str | None = None  # the first key listed past the start key
    past_hi: str | None = None  # the first key listed past ``hi``
    in_order: bool = True  # the keys came in the server's order
    # with a delimiter: the folders past it; None if the client doesn't say
    folders: tuple[str, ...] | None = None


class Keep:
    """Which listed keys become rows: valid keys up to ``hi``. Notes the first key
    past ``lo``, the first past ``hi``, and whether the keys came in order."""

    def __init__(self, client: Lister, lo: str, hi: str | None) -> None:
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

    def take(self, rows: list[Any], keys: list[str]) -> list[Any]:
        """The ``rows`` to keep, of a page that listed ``keys`` (a row's key
        each). A page in code point order is checked as a whole: a page of
        S3 keys takes no Python call per key."""
        if self.order not in (str, code_points) or self.previous is not None:
            return [row for row, key in zip(rows, keys, strict=True) if self(key)]
        if keys != sorted(keys):
            return [row for row, key in zip(rows, keys, strict=True) if self(key)]
        if keys:
            self.previous = keys[-1]
        first = bisect.bisect_right(keys, self.lo_key)
        self.past_lo = keys[first] if first < len(keys) else None
        end = len(keys)
        if self.hi_key is not None:
            end = bisect.bisect_right(keys, self.hi_key)
            self.past_hi = keys[end] if end < len(keys) else None
        rows, keys = rows[:end], keys[:end]
        if self.valid is not valid_key:
            return [row for row, key in zip(rows, keys, strict=True) if self.valid(key)]
        return [
            row
            for row, key in zip(rows, keys, strict=True)
            if key[:1] != "/" and key[-1:] != "/"
        ]


def valid_key(key: str) -> bool:
    """Whether a listed key becomes a row. Folder markers (``a/``) and keys with
    a leading ``/`` are skipped (see #2040); keys with ``//`` inside (``a//b``)
    are kept."""
    return not (key.startswith("/") or key.endswith("/"))


def code_points(key: str) -> str:
    return key


def utf16_units(key: str) -> bytes:
    return key.encode("utf-16-be", "surrogatepass")


class KeyRange:
    """A range ``(lo, hi]`` of keys in ``order``, the server's order, and how far
    it has been read: the last key listed and the token of the next page."""

    def __init__(
        self,
        lo: str,
        hi: str | None,
        order: Callable[[str], Any],
        age: int,
        last: str | None = None,
        token: Any = None,
    ) -> None:
        self.lo = lo
        self.hi = hi
        self.order = order
        self.age = age  # pages read by this range and the ranges it split off from
        self.lo_key = order(lo)
        self.last = last  # the last key listed, of any kind
        self.read_to = self.lo_key if last is None else order(last)
        self.token = token  # resumes after ``last``; None: start after ``lo``
        self.ended = False  # its last page has been read

    def covers(self, key: str) -> bool:
        return self.hi is None or self.order(key) <= self.order(self.hi)

    def rest(self) -> "KeyRange | None":
        """What is left to read, or None when the range has been read."""
        if self.ended:
            return None
        return KeyRange(self.lo, self.hi, self.order, self.age, self.last, self.token)


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


class RangeReader:
    """Reads key ranges a page at a time, and splits a range when there is room
    for more. ``RangeListing`` reads ranges in this process; a listing child
    process (``listing_worker``) reads the ranges the listing hands it. Each says
    where pages, splits and checks go."""

    def __init__(self, client: Lister, prefix: str, delimiter: str = "") -> None:
        self.client = client
        self.prefix = prefix
        self.delimiter = delimiter  # with it, keys past a delimiter are skipped
        self.order = client.key_order
        self.alphabet: set[str] = set()
        self.listed_any = False
        # off when the server ignores start keys or lists keys out of order
        self.splits = os.getenv("DATACHAIN_RANGE_LISTING", "1").lower() in (
            "1", "true", "yes", "on",
        )  # fmt: skip

    async def _list_page(self, lo: str, token: Any, hi: str | None) -> Page:
        # only a level listing passes it: a client that lists no levels needn't
        # take it
        kwargs: dict[str, Any] = {"delimiter": self.delimiter} if self.delimiter else {}
        return await self.client._list_page(self.prefix, lo, token, hi, **kwargs)

    async def _fetch(self, key_range: KeyRange, token: Any) -> Page:
        lo, hi = key_range.lo, key_range.hi
        page = await self._list_page(lo, token, hi)
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

    async def _read(self, key_range: KeyRange, hand_over: bool = True) -> None:
        """Reads the range a page at a time; the next page is on its way while
        this one is taken. With ``hand_over``, the rest may go elsewhere."""
        page = await self._fetch(key_range, key_range.token)
        next_page: asyncio.Task | None = None
        try:
            while True:
                rows = self._advance(key_range, page)
                token = key_range.token
                if token is not None and not (hand_over and self._hand_over(key_range)):
                    next_page = asyncio.create_task(self._fetch(key_range, token))
                    # its request goes out before the work on this page
                    await asyncio.sleep(0)
                await self._accept(key_range, rows, page)
                if next_page is None:
                    return
                page, next_page = await next_page, None
        finally:
            if next_page is not None:
                next_page.cancel()
                await asyncio.gather(next_page, return_exceptions=True)

    def _advance(self, key_range: KeyRange, page: Page) -> list[Any]:
        """Checks and records what ``page`` read, and maybe splits the range.
        Returns the page's rows in the range; ``key_range.token`` is then the
        token of its next page, or None at its end."""
        key_range.age += 1
        self.listed_any = self.listed_any or page.first is not None
        first = None if page.first is None else self.order(page.first)
        if first is not None and first < key_range.lo_key:
            self._start_ignored()
        elif not page.in_order or (first is not None and first < key_range.read_to):
            self._out_of_order()
        rows = page.rows
        keys = [self.order(r.path) for r in rows]
        if keys and keys[0] <= key_range.lo_key:  # GCS and Azure list lo itself
            rows = [r for r, k in zip(rows, keys, strict=True) if k > key_range.lo_key]
        key_range.token = page.token
        key_range.ended = page.token is None
        if (last := page.last) is None:
            return rows
        key_range.last, key_range.read_to = last, self.order(last)
        if not key_range.covers(last):
            key_range.token, key_range.ended = None, True  # read past hi
            return rows
        if key_range.read_to > key_range.lo_key:
            self.alphabet.update(last)
            past_first_page = bool(key_range.lo) or key_range.age > 1
            if page.token is not None and past_first_page and self._may_split():
                page_start = page.rows[0].path if page.rows else None
                self._split(key_range, last, page_start)
        return rows

    def _may_split(self) -> bool:
        return self.splits and self.client._lists_in_key_order()

    def _split(self, key_range: KeyRange, read_to: str, page_start: str | None) -> None:
        """Hands the range's upper parts over while there is room for them."""
        for mid in split_points(
            key_range, read_to, page_start, self.alphabet, self.prefix
        ):
            if not self._room():
                return
            self._add(key_range, mid)

    def _start_ignored(self) -> None:
        self.splits = False

    def _out_of_order(self) -> None:
        self.splits = False

    def _hand_over(self, key_range: KeyRange) -> bool:
        """Whether the rest of ``key_range`` went to be read elsewhere."""
        return False

    def _room(self) -> bool:
        raise NotImplementedError

    def _add(self, key_range: KeyRange, mid: str) -> None:
        """Ends ``key_range`` at ``mid``; the part above is a range of its own."""
        raise NotImplementedError

    async def _accept(self, key_range: KeyRange, rows: list[Any], page: Page) -> None:
        """Takes the rows of a page of ``key_range``, read to ``key_range.last``."""
        raise NotImplementedError

    async def _seam(self, start: str, side: int, key: str) -> None:
        """The first key past ``start``, as the range below it sees it (side 0)
        and as the range above it does (side 1)."""
        raise NotImplementedError


class RangeListing(RangeReader):
    """One listing of a prefix as key ranges ``(lo, hi]`` read in parallel.

    When a page arrives and there is room for more ranges, its range hands upper
    parts over (``split_points``). With a pool, child processes read ranges once
    they run: this process hands them ranges, passes their rows on, and checks
    the seams between them. The fuzz tests check these rules:

    - the ranges cover the listing without overlap;
    - ``hi`` only shrinks, and only before the range's next page is asked for;
    - a token advances only after its page is accepted.
    """

    client: "RangeListedClient"

    def __init__(
        self,
        client: "RangeListedClient",
        prefix: str,
        result_queue: "ResultQueue",
        pool: "Pool | None",
        delimiter: str = "",
    ) -> None:
        super().__init__(client, prefix, delimiter)
        self.result_queue = result_queue
        self.pool = pool
        if pool:
            pool.listing = self
        self.ranges: asyncio.Queue[KeyRange] = asyncio.Queue()
        self.ranges.put_nowait(KeyRange("", None, self.order, 0))
        self.open_ranges = 1  # queued or being read
        self.rows = 0
        self.has_split = False
        # the first key past a range start: as read from below it, and from above
        self.seams: dict[str, list[str | None]] = {}
        self.own_requests = asyncio.Semaphore(LIST_WORKERS)  # this process's
        self.local = asyncio.Event()  # set while this process reads ranges
        self.local.set()
        self.tasks: set[asyncio.Task] = set()
        self.failed: asyncio.Future | None = None
        from datachain.progress import tqdm

        self.pbar = tqdm(desc=f"Listing {client.uri}", unit=" objects", leave=False)

    async def run(self) -> None:
        self.failed = asyncio.get_running_loop().create_future()
        try:
            await self._cut()
        except BaseException:
            self.pbar.close()
            raise
        for _ in range(LIST_WORKERS):
            self.start_task(self._worker())
        all_read = asyncio.create_task(self.ranges.join())
        try:
            await asyncio.wait(
                [all_read, self.failed], return_when=asyncio.FIRST_COMPLETED
            )
            if self.failed.done():
                self.failed.result()
            if not self.listed_any and self.prefix:
                raise FileNotFoundError(f"Unable to resolve remote path: {self.prefix}")
        finally:
            self.failed.cancel()
            tasks = {all_read, *self.tasks}
            for task in tasks:
                task.cancel()
            if self.pool:  # before the await: gc may close this coroutine
                self.pool.close()
            # again until they end: a library that loses a cancel must not hang this
            pending = tasks
            while pending := (await asyncio.wait(pending, timeout=1))[1]:
                for task in pending:
                    task.cancel()
            for task in tasks:
                if not task.cancelled():
                    task.exception()  # retrieved: the first failure is raised above
            if self.pool:
                await self.pool.wait_closed()
            self.pbar.close()

    async def _cut(self) -> None:
        """Reads the first page; when more follow, starts ranges at folders
        right under the prefix past it, which one request with a delimiter
        names. A large listing then starts spread over its keys instead of
        splitting its way out of one range. A one-page listing takes one
        request; a listing of many folders starts at most ``CUT_RANGES``
        ranges, as each costs a request that may read keys past its end."""
        if self.delimiter or not self.client.NAMES_FOLDERS or not self._may_split():
            return
        key_range = self.ranges.get_nowait()  # the one range of the whole prefix
        page = await self._fetch(key_range, None)
        await self._accept(key_range, self._advance(key_range, page), page)
        if key_range.ended:
            self._close_range()
            return
        folders: list[str] = []
        if self._may_split():  # the first page may show keys out of order
            # "/": the delimiter of the folders the listing's paths name
            level = await self._list_page_level(key_range.last or "", "/")
            past = (f for f in level.folders or () if self.order(f) > key_range.read_to)
            folders = sorted(past, key=self.order)
            folders = folders[:: -(-len(folders) // CUT_RANGES) or 1]
        if folders:
            self.has_split = True
            for lo, hi in zip(folders, [*folders[1:], key_range.hi], strict=True):
                self._queue(KeyRange(lo, hi, self.order, 0))
            key_range.hi = folders[0]  # its next page is not asked for yet
        self.ranges.put_nowait(key_range)  # read on from its token
        self.ranges.task_done()

    async def _list_page_level(self, start_after: str, delimiter: str) -> Page:
        async with self.own_requests:
            return await self.client._list_page(
                self.prefix, start_after, None, delimiter=delimiter
            )

    def start_task(self, coro: Coroutine) -> asyncio.Task:
        """A task whose failure fails the listing."""
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self._ended)
        return task

    def _ended(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        if (error := task.exception()) and self.failed and not self.failed.done():
            self.failed.set_exception(error)

    def has_failed(self) -> bool:
        return self.failed is not None and self.failed.done()

    def fail(self, error: BaseException) -> None:
        if self.failed and not self.failed.done():
            self.failed.set_exception(error)

    async def _worker(self) -> None:
        """Reads ranges in this process while no child process runs."""
        while True:
            await self.local.wait()
            key_range = await self.ranges.get()
            if not self.local.is_set():  # a child came up meanwhile: it reads this
                self.ranges.put_nowait(key_range)
                self.ranges.task_done()
                continue
            try:
                await self._read(key_range)
            finally:
                self._close_range()

    async def read_in(self, child: "Child") -> None:
        """Hands ranges to ``child`` one at a time, until it is gone. The rest of
        a range it failed is read here from the page it failed, so that a lasting
        failure raises as it would, and goes back to the children after a page."""
        while True:
            key_range = await self.ranges.get()
            outcome = "gone"
            try:
                outcome, rest = await child.read(key_range)
                if rest is not None and outcome == "failed":
                    await self._read(rest)
                elif rest is not None:  # the same range, read on by another
                    self.ranges.put_nowait(rest)
                    self.open_ranges += 1
            finally:
                self._close_range()
            if outcome != "done":
                return

    def _queue(self, key_range: KeyRange) -> None:
        self.open_ranges += 1
        self.ranges.put_nowait(key_range)
        self.changed()

    def _close_range(self) -> None:
        self.open_ranges -= 1
        self.ranges.task_done()
        self.changed()

    def changed(self) -> None:
        """The room for ranges changed: children learn how much they may split,
        which is none once splitting is off."""
        if self.pool:
            room = self._capacity() - self.open_ranges if self._may_split() else 0
            self.pool.share_room(room)

    def _capacity(self) -> int:
        """Ranges read at once: by this process, or by the children once any runs."""
        if self.pool and self.pool.children:
            return CHILD_WORKERS * len(self.pool.children)
        return LIST_WORKERS

    def _room(self) -> bool:
        return self.open_ranges < self._capacity()

    def _add(self, key_range: KeyRange, mid: str) -> None:
        self.has_split = True
        self._queue(KeyRange(mid, key_range.hi, self.order, key_range.age))
        key_range.hi = mid

    def _hand_over(self, key_range: KeyRange) -> bool:
        """Once a child runs, this process stops reading: the rest goes to them."""
        if self.local.is_set() or (rest := key_range.rest()) is None:
            return False
        self._queue(rest)
        key_range.token, key_range.ended = None, True
        return True

    async def _list_page(self, lo: str, token: Any, hi: str | None) -> Page:
        async with self.own_requests:
            return await super()._list_page(lo, token, hi)

    async def _accept(self, key_range: KeyRange, rows: list[Any], page: Page) -> None:
        await self.emit(rows)

    async def emit(self, rows: Sequence[Any]) -> None:
        if rows:
            await self.result_queue.put(rows)
            self.pbar.update(len(rows))
            self.rows += len(rows)
            if self.pool and self.rows >= POOL_ROWS and self._may_split():
                self.pool.start()

    def progress(
        self, key_range: KeyRange, last: str | None, token: Any, age: int
    ) -> None:
        """A child read ``key_range`` to ``last``; ``token`` resumes it."""
        key_range.age, key_range.token, key_range.ended = age, token, token is None
        if last is not None:
            key_range.last, key_range.read_to = last, self.order(last)

    def _start_ignored(self) -> None:
        self.splits = False
        self.changed()  # children split no more

    def _out_of_order(self) -> None:
        """Keys came back out of the server's order, so ranges no longer split. A
        range split earlier may have ended before keys it holds."""
        if self.has_split:
            raise RuntimeError(
                f"Listing {self.client.uri}: keys came back out of order after the"
                " listing split, so it could be incomplete. DATACHAIN_RANGE_LISTING=0"
                " lists in one range."
            )
        self._start_ignored()

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
