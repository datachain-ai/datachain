import contextlib
import traceback
from collections.abc import Iterable, Sequence
from itertools import chain
from multiprocessing import cpu_count
from queue import Empty
from sys import stdin
from time import monotonic, sleep
from typing import TYPE_CHECKING, Literal

import multiprocess
from cloudpickle import load, loads
from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from multiprocess.context import Process
from multiprocess.queues import Queue as MultiprocessQueue

from datachain.catalog import Catalog
from datachain.catalog.catalog import clone_catalog_with_cache
from datachain.catalog.loader import DISTRIBUTED_IMPORT_PATH, get_udf_distributor_class
from datachain.lib.model_store import ModelStore
from datachain.lib.udf import UdfRunError, _get_cache
from datachain.query.dataset import (
    get_download_callback,
    get_generated_callback,
    get_processed_callback,
    process_udf_outputs,
)
from datachain.query.queue import get_from_queue, put_into_queue
from datachain.query.udf import UdfInfo
from datachain.utils import batched, flatten, safe_closing

if TYPE_CHECKING:
    from sqlalchemy import Select, Table

    from datachain.data_storage import AbstractMetastore, AbstractWarehouse
    from datachain.lib.udf import UDFAdapter
    from datachain.query.batch import RowsOutput

DEFAULT_BATCH_SIZE = 10000
STOP_SIGNAL = "STOP"
OK_STATUS = "OK"
FINISHED_STATUS = "FINISHED"
FAILED_STATUS = "FAILED"
NOTIFY_STATUS = "NOTIFY"


def get_n_workers_from_arg(n_workers: int | None = None) -> int:
    if not n_workers:
        return cpu_count()
    if n_workers < 1:
        raise RuntimeError("Must use at least one worker for parallel UDF execution!")
    return n_workers


def udf_entrypoint() -> int:
    """Parallel processing (faster for more CPU-heavy UDFs)."""
    # Load UDF info from stdin
    udf_info: UdfInfo = load(stdin.buffer)

    query = udf_info["query"]
    if "sys__id" not in query.selected_columns:
        raise RuntimeError("sys__id column is required in UDF query")

    batching = udf_info["batching"]
    is_generator = udf_info["is_generator"]

    download_cb = get_download_callback()
    processed_cb = get_processed_callback()
    generated_cb = get_generated_callback(is_generator)

    wh_cls, wh_args, wh_kwargs = udf_info["warehouse_clone_params"]
    warehouse: AbstractWarehouse = wh_cls(*wh_args, **wh_kwargs)

    with contextlib.closing(
        batching(
            warehouse.dataset_select_paginated,
            query,
            id_col=query.selected_columns.sys__id,
        )
    ) as udf_inputs:
        try:
            UDFDispatcher(udf_info).run_udf(
                udf_inputs,
                download_cb=download_cb,
                processed_cb=processed_cb,
                generated_cb=generated_cb,
            )
        finally:
            download_cb.close()
            processed_cb.close()
            generated_cb.close()

    return 0


def udf_worker_entrypoint() -> int:
    if not (udf_distributor_class := get_udf_distributor_class()):
        raise RuntimeError(
            f"{DISTRIBUTED_IMPORT_PATH} import path is required "
            "for distributed UDF processing."
        )

    return udf_distributor_class.run_udf()


class UDFDispatcher:
    _catalog: Catalog | None = None
    task_queue: MultiprocessQueue | None = None
    done_queue: MultiprocessQueue | None = None

    def __init__(self, udf_info: UdfInfo, buffer_size: int = DEFAULT_BATCH_SIZE):
        self.udf_data = udf_info["udf_data"]
        self.catalog_init_params = udf_info["catalog_init"]
        self.metastore_clone_params = udf_info["metastore_clone_params"]
        self.warehouse_clone_params = udf_info["warehouse_clone_params"]
        self.query = udf_info["query"]
        self.table = udf_info["table"]
        self.udf_fields = udf_info["udf_fields"]
        self.cache = udf_info["cache"]
        self.is_generator = udf_info["is_generator"]
        self.is_batching = udf_info["batching"].is_batching
        self.processes = udf_info["processes"]
        self.rows_total = udf_info["rows_total"]
        self.batch_size = udf_info["batch_size"]
        self.buffer_size = buffer_size
        self.task_queue = None
        self.done_queue = None
        self.ctx = multiprocess.get_context("spawn")

    @property
    def catalog(self) -> "Catalog":
        if not self._catalog:
            ms_cls, ms_args, ms_kwargs = self.metastore_clone_params
            metastore: AbstractMetastore = ms_cls(*ms_args, **ms_kwargs)
            ws_cls, ws_args, ws_kwargs = self.warehouse_clone_params
            warehouse: AbstractWarehouse = ws_cls(*ws_args, **ws_kwargs)
            self._catalog = Catalog(metastore, warehouse, **self.catalog_init_params)
        return self._catalog

    def _create_worker(self) -> "UDFWorker":
        udf: UDFAdapter = loads(self.udf_data)
        # Ensure all registered DataModels have rebuilt schemas in worker processes.
        ModelStore.rebuild_all()
        return UDFWorker(
            self.catalog,
            udf,
            self.task_queue,
            self.done_queue,
            self.query,
            self.table,
            self.cache,
            self.is_batching,
            self.batch_size,
            self.udf_fields,
        )

    def _run_worker(self) -> None:
        try:
            worker = self._create_worker()
            worker.run()
        except (Exception, KeyboardInterrupt) as e:
            if self.done_queue:
                # We put the exception into the done queue so the main process
                # can handle it appropriately. We include the stacktrace to propagate
                # it to the main process and show it to the user.
                put_into_queue(
                    self.done_queue,
                    {
                        "status": FAILED_STATUS,
                        "exception": e,
                        "stacktrace": traceback.format_exc(),
                    },
                )
            if isinstance(e, KeyboardInterrupt):
                return
            raise

    def run_udf(
        self,
        input_rows: Iterable["RowsOutput"],
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
        generated_cb: Callback = DEFAULT_CALLBACK,
    ) -> None:
        n_workers = self.processes
        if n_workers is True:
            n_workers = None  # Use default number of CPUs (cores)
        elif not n_workers or n_workers < 1:
            n_workers = 1  # Single-threaded (on this worker)
        n_workers = get_n_workers_from_arg(n_workers)

        if n_workers == 1:
            # no need to spawn worker processes if we are running in a single process
            self.run_udf_single(input_rows, download_cb, processed_cb, generated_cb)
        else:
            if self.buffer_size < n_workers:
                raise RuntimeError(
                    "Parallel run error: buffer size is smaller than "
                    f"number of workers: {self.buffer_size} < {n_workers}"
                )

            self.run_udf_parallel(
                n_workers, input_rows, download_cb, processed_cb, generated_cb
            )

    def run_udf_single(
        self,
        input_rows: Iterable["RowsOutput"],
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
        generated_cb: Callback = DEFAULT_CALLBACK,
    ) -> None:
        udf: UDFAdapter = loads(self.udf_data)
        # Rebuild schemas in single process too for consistency (cheap, idempotent).
        ModelStore.rebuild_all()

        if not self.is_batching:
            input_rows = flatten(input_rows)

        def get_inputs() -> Iterable["RowsOutput"]:
            warehouse = self.catalog.warehouse.clone()
            for ids in batched(input_rows, DEFAULT_BATCH_SIZE):
                yield from warehouse.dataset_rows_select_from_ids(
                    self.query, ids, self.is_batching
                )

        prefetch = udf.prefetch
        with _get_cache(self.catalog.cache, prefetch, use_cache=self.cache) as _cache:
            udf_results = udf.run(
                self.udf_fields,
                get_inputs(),
                self.catalog,
                self.cache,
                download_cb=download_cb,
                processed_cb=processed_cb,
            )
            with safe_closing(udf_results):
                process_udf_outputs(
                    self.catalog.warehouse.clone(),
                    self.table,
                    udf_results,
                    udf,
                    cb=generated_cb,
                    batch_size=self.batch_size,
                )

    def input_batch_size(self, n_workers: int) -> int:
        input_batch_size = self.rows_total // n_workers
        if input_batch_size == 0:
            input_batch_size = 1
        elif input_batch_size > DEFAULT_BATCH_SIZE:
            input_batch_size = DEFAULT_BATCH_SIZE
        return input_batch_size

    def run_udf_parallel(  # noqa: C901, PLR0912
        self,
        n_workers: int,
        input_rows: Iterable["RowsOutput"],
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
        generated_cb: Callback = DEFAULT_CALLBACK,
    ) -> None:
        self.task_queue = self.ctx.Queue()
        self.done_queue = self.ctx.Queue()

        pool = [
            self.ctx.Process(name=f"Worker-UDF-{i}", target=self._run_worker)
            for i in range(n_workers)
        ]
        for p in pool:
            p.start()

        try:
            # Will be set to True when the input is exhausted
            input_finished = False

            input_rows = batched(
                input_rows if self.is_batching else flatten(input_rows),
                self.input_batch_size(n_workers),
            )

            # Stop all workers after the input rows have finished processing
            input_data = chain(input_rows, [STOP_SIGNAL] * n_workers)

            # Add initial buffer of tasks
            for _ in range(self.buffer_size):
                try:
                    put_into_queue(self.task_queue, next(input_data))
                except StopIteration:
                    input_finished = True
                    break

            # Process all tasks
            while n_workers > 0:
                while True:
                    try:
                        result = self.done_queue.get_nowait()
                        break
                    except Empty:
                        for p in pool:
                            exitcode = p.exitcode
                            if exitcode not in (None, 0):
                                message = (
                                    f"Worker {p.name} exited unexpectedly with "
                                    f"code {exitcode}"
                                )
                                raise RuntimeError(message) from None
                        sleep(0.01)

                if bytes_downloaded := result.get("bytes_downloaded"):
                    download_cb.relative_update(bytes_downloaded)
                if downloaded := result.get("downloaded"):
                    download_cb.increment_file_count(downloaded)
                if processed := result.get("processed"):
                    processed_cb.relative_update(processed)
                if generated := result.get("generated"):
                    generated_cb.relative_update(generated)

                status = result["status"]
                if status in (OK_STATUS, NOTIFY_STATUS):
                    pass  # Do nothing here
                elif status == FINISHED_STATUS:
                    n_workers -= 1  # Worker finished
                else:  # Failed / error
                    n_workers -= 1
                    if exc := result.get("exception"):
                        if isinstance(exc, KeyboardInterrupt):
                            raise exc
                        raise UdfRunError(exc, stacktrace=result.get("stacktrace"))
                    raise RuntimeError("Internal error: Parallel UDF execution failed")

                if status == OK_STATUS and not input_finished:
                    try:
                        put_into_queue(self.task_queue, next(input_data))
                    except StopIteration:
                        input_finished = True
        finally:
            self._shutdown_workers(pool)

    def _shutdown_workers(self, pool: list[Process]) -> None:
        self._terminate_pool(pool)
        self._drain_queue(self.done_queue)
        self._drain_queue(self.task_queue)
        self._close_queue(self.done_queue)
        self._close_queue(self.task_queue)

    def _terminate_pool(self, pool: list[Process]) -> None:
        for proc in pool:
            if proc.is_alive():
                proc.terminate()

        deadline = monotonic() + 1.0
        for proc in pool:
            if not proc.is_alive():
                continue
            remaining = deadline - monotonic()
            if remaining > 0:
                proc.join(remaining)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=0.2)

    def _drain_queue(self, queue: MultiprocessQueue) -> None:
        while True:
            try:
                queue.get_nowait()
            except Empty:
                return
            except (OSError, ValueError):
                return

    def _close_queue(self, queue: MultiprocessQueue) -> None:
        with contextlib.suppress(OSError, ValueError):
            queue.close()
        with contextlib.suppress(RuntimeError, AssertionError, ValueError):
            queue.join_thread()


class DownloadCallback(Callback):
    def __init__(self, queue: MultiprocessQueue) -> None:
        self.queue = queue
        super().__init__()

    def relative_update(self, inc: int = 1) -> None:
        put_into_queue(self.queue, {"status": NOTIFY_STATUS, "bytes_downloaded": inc})

    def increment_file_count(self, inc: int = 1) -> None:
        put_into_queue(self.queue, {"status": NOTIFY_STATUS, "downloaded": inc})


class ProcessedCallback(Callback):
    def __init__(
        self,
        name: Literal["processed", "generated"],
        queue: MultiprocessQueue,
    ) -> None:
        self.name = name
        self.queue = queue
        super().__init__()

    def relative_update(self, inc: int = 1) -> None:
        put_into_queue(self.queue, {"status": NOTIFY_STATUS, self.name: inc})


class UDFWorker:
    def __init__(
        self,
        catalog: "Catalog",
        udf: "UDFAdapter",
        task_queue: MultiprocessQueue,
        done_queue: MultiprocessQueue,
        query: "Select",
        table: "Table",
        cache: bool,
        is_batching: bool,
        batch_size: int,
        udf_fields: Sequence[str],
    ) -> None:
        self.catalog = catalog
        self.udf = udf
        self.task_queue = task_queue
        self.done_queue = done_queue
        self.query = query
        self.table = table
        self.cache = cache
        self.is_batching = is_batching
        self.batch_size = batch_size
        self.udf_fields = udf_fields

        self.download_cb = DownloadCallback(self.done_queue)
        self.processed_cb = ProcessedCallback("processed", self.done_queue)
        self.generated_cb = ProcessedCallback("generated", self.done_queue)

    def run(self) -> None:
        prefetch = self.udf.prefetch
        with _get_cache(self.catalog.cache, prefetch, use_cache=self.cache) as _cache:
            catalog = clone_catalog_with_cache(self.catalog, _cache)
            udf_results = self.udf.run(
                self.udf_fields,
                self.get_inputs(),
                catalog,
                self.cache,
                download_cb=self.download_cb,
                processed_cb=self.processed_cb,
            )
            with safe_closing(udf_results):
                process_udf_outputs(
                    catalog.warehouse,
                    self.table,
                    self.notify_and_process(udf_results),
                    self.udf,
                    cb=self.generated_cb,
                    batch_size=self.batch_size,
                )
        put_into_queue(self.done_queue, {"status": FINISHED_STATUS})

    def notify_and_process(self, udf_results):
        for row in udf_results:
            put_into_queue(self.done_queue, {"status": OK_STATUS})
            yield row

    def get_inputs(self) -> Iterable["RowsOutput"]:
        warehouse = self.catalog.warehouse.clone()
        while (batch := get_from_queue(self.task_queue)) != STOP_SIGNAL:
            for ids in batched(batch, DEFAULT_BATCH_SIZE):
                yield from warehouse.dataset_rows_select_from_ids(
                    self.query, ids, self.is_batching
                )
