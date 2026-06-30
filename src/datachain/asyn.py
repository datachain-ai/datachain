import asyncio
import threading
from collections.abc import (
    AsyncIterable,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Iterable,
    Iterator,
)
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Any, Generic, TypeVar

from fsspec.asyn import get_loop

from datachain.utils import safe_closing

ASYNC_WORKERS = 20

InputT = TypeVar("InputT", contravariant=True)  # noqa: PLC0105
ResultT = TypeVar("ResultT", covariant=True)  # noqa: PLC0105
T = TypeVar("T")


@dataclass
class _ThreadState:
    pool: ThreadPoolExecutor
    shutdown: threading.Event = field(default_factory=threading.Event)
    is_shutdown: threading.Event = field(default_factory=threading.Event)


@dataclass
class _QueueState:
    work: Any = None
    result: Any = None


class AsyncMapper(Generic[InputT, ResultT]):
    """
    Asynchronous unordered mapping iterable compatible with fsspec.

    `AsyncMapper(func, it)` is roughly equivalent to `map(func, it)`, except
    that `func` is an async function, which is executed concurrently by up to
    `workers` asyncio coroutines, and the results are yielded in arbitrary
    order.

    If `func` needs to call synchronous functions that may themselves run coroutines
    on the same loop, it must use `mapper.to_thread()`.

    Note that `loop`, which defaults to the fsspec loop, must be running on a different
    thread than the one calling `mapper.iterate()`.
    """

    order_preserving = False
    loop: asyncio.AbstractEventLoop

    def __init__(
        self,
        func: Callable[[InputT], Awaitable[ResultT]],
        iterable: Iterable[InputT],
        *,
        workers: int = ASYNC_WORKERS,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.func = func
        self.iterable = iterable
        self.workers = workers
        self.loop = get_loop() if loop is None else loop
        self._thread = _ThreadState(pool=ThreadPoolExecutor(workers))
        self._tasks: set[asyncio.Task] = set()

    def start_task(self, coro: Coroutine) -> asyncio.Task:
        task = self.loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _produce(self) -> None:
        try:
            with safe_closing(self.iterable):
                for item in self.iterable:
                    if self._thread.shutdown.is_set():
                        return
                    coro = self._queues.work.put(item)
                    fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
                    fut.result()  # wait until the item is in the queue
        finally:
            self._thread.is_shutdown.set()

    async def produce(self) -> None:
        await self.to_thread(self._produce)

    def shutdown_producer(self) -> None:
        """
        Signal the producer to stop and drain any remaining items from the work_queue.

        This method sets an internal event, `_shutdown_producer`, which tells the
        producer that it should stop adding items to the queue. To ensure that the
        producer notices this signal promptly, we also attempt to drain any items
        currently in the queue, clearing it so that the event can be checked without
        delay.
        """
        self._thread.shutdown.set()
        q = self._queues.work
        while not q.empty():
            q.get_nowait()
            q.task_done()

    async def worker(self) -> None:
        while (item := await self._queues.work.get()) is not None:
            try:
                result = await self.func(item)
                await self._queues.result.put(result)
            finally:
                self._queues.work.task_done()

    async def init(self) -> None:
        self._queues = _QueueState(
            work=asyncio.Queue(2 * self.workers),
            result=asyncio.Queue(self.workers),
        )

    async def run(self) -> None:
        producer = self.start_task(self.produce())
        for _i in range(self.workers):
            self.start_task(self.worker())
        try:
            done, _pending = await asyncio.wait(
                self._tasks, return_when=asyncio.FIRST_COMPLETED
            )
            self.gather_exceptions(done)
            assert producer.done()
            join = self.start_task(self._queues.work.join())
            done, _pending = await asyncio.wait(
                self._tasks, return_when=asyncio.FIRST_COMPLETED
            )
            self.gather_exceptions(done)
            assert join.done()
        except:
            await self.cancel_all()
            await self._break_iteration()
            raise
        else:
            await self.cancel_all()
            await self._end_iteration()

    async def cancel_all(self) -> None:
        if self._tasks:
            for task in self._tasks:
                task.cancel()
            await asyncio.wait(self._tasks)

    def gather_exceptions(self, done_tasks):
        # Check all exceptions to avoid "Task exception was never retrieved" warning
        exceptions = [task.exception() for task in done_tasks]
        # Raise the first exception found, if any. Additional ones are ignored.
        for exc in exceptions:
            if exc:
                raise exc

    async def _pop_result(self) -> ResultT | None:
        return await self._queues.result.get()

    def next_result(self, timeout=None) -> ResultT | None:
        """
        Return the next available result.

        Blocks as long as the result queue is empty.
        """
        future = asyncio.run_coroutine_threadsafe(self._pop_result(), self.loop)
        return future.result(timeout=timeout)

    async def _end_iteration(self) -> None:
        """Signal successful end of iteration."""
        await self._queues.result.put(None)

    async def _break_iteration(self) -> None:
        """Signal that iteration must stop ASAP."""
        while not self._queues.result.empty():
            self._queues.result.get_nowait()
        await self._queues.result.put(None)

    def iterate(self, timeout=None) -> Generator[ResultT, None, None]:
        init = asyncio.run_coroutine_threadsafe(self.init(), self.loop)
        init.result(timeout=1)
        async_run = asyncio.run_coroutine_threadsafe(self.run(), self.loop)
        try:
            while True:
                if (result := self.next_result(timeout)) is not None:
                    yield result
                else:
                    break
            if exc := async_run.exception():
                raise exc
        finally:
            self.shutdown_producer()
            if not async_run.done():
                async_run.cancel()
                wait([async_run])
            self._thread.is_shutdown.wait()

    def __iter__(self):
        return self.iterate()

    async def to_thread(self, func, *args):
        return await self.loop.run_in_executor(self._thread.pool, func, *args)


@dataclass
class _OrderState:
    waiters: dict[int, Any] = field(default_factory=dict)
    getters: dict[int, Any] = field(default_factory=dict)
    heap: list = field(default_factory=list)
    next_yield: int = 0
    items_seen: int = 0
    window: int = 0


class OrderedMapper(AsyncMapper[InputT, ResultT]):
    """
    Asynchronous ordered mapping iterable compatible with fsspec.

    See `AsyncMapper` for details.
    """

    order_preserving = True

    def __init__(
        self,
        func: Callable[[InputT], Awaitable[ResultT]],
        iterable: Iterable[InputT],
        *,
        workers: int = ASYNC_WORKERS,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__(func, iterable, workers=workers, loop=loop)
        self._order = _OrderState(window=2 * workers)

    def _push_result(self, i: int, result: ResultT | None) -> None:
        if i in self._order.getters:
            future = self._order.getters.pop(i)
            future.set_result(result)
        else:
            heappush(self._order.heap, (i, result))

    async def worker(self) -> None:
        while (item := await self._queues.work.get()) is not None:
            i = self._order.items_seen
            self._order.items_seen += 1
            if i >= self._order.next_yield + self._order.window:
                event = self._order.waiters[i - self._order.window] = asyncio.Event()
                await event.wait()
            result = await self.func(item)
            self._push_result(i, result)
            self._queues.work.task_done()

    async def init(self) -> None:
        self._queues = _QueueState(work=asyncio.Queue(2 * self.workers))

    async def _pop_result(self) -> ResultT | None:
        if self._order.heap and self._order.heap[0][0] == self._order.next_yield:
            _i, out = heappop(self._order.heap)
        else:
            self._order.getters[self._order.next_yield] = get_value = (
                self.loop.create_future()
            )
            out = await get_value
        if self._order.next_yield in self._order.waiters:
            event = self._order.waiters.pop(self._order.next_yield)
            event.set()
        self._order.next_yield += 1
        return out

    async def _end_iteration(self) -> None:
        self._push_result(self._order.next_yield + len(self._order.heap), None)

    async def _break_iteration(self) -> None:
        self._order.heap = []
        self._push_result(self._order.next_yield, None)


def iter_over_async(ait: AsyncIterable[T], loop) -> Iterator[T]:
    """Wrap an asynchronous iterator into a synchronous one"""
    ait = ait.__aiter__()

    # helper async fn that just gets the next element from the async iterator
    async def get_next():
        try:
            obj = await ait.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None

    # actual sync iterator
    while True:
        done, obj = asyncio.run_coroutine_threadsafe(get_next(), loop).result()
        if done:
            break
        yield obj
