import contextlib
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Sequence

import sqlalchemy as sa

from datachain.data_storage.schema import PARTITION_COLUMN_ID

RowsOutputBatch = Sequence[Sequence]
RowsOutput = Sequence | RowsOutputBatch


class BatchingStrategy(ABC):
    """BatchingStrategy provides means of batching UDF executions."""

    is_batching: bool

    @abstractmethod
    def __call__(
        self,
        execute: Callable,
        query: sa.Select,
        id_col: sa.ColumnElement | None = None,
    ) -> Generator[RowsOutput, None, None]:
        """Apply the provided parameters to the UDF."""


class NoBatching(BatchingStrategy):
    """
    NoBatching implements the default batching strategy, which is not to
    batch UDF calls.
    """

    is_batching = False

    def __call__(
        self,
        execute: Callable,
        query: sa.Select,
        id_col: sa.ColumnElement | None = None,
    ) -> Generator[Sequence, None, None]:
        ids_only = False
        if id_col is not None:
            query = query.with_only_columns(id_col)
            ids_only = True

        rows = execute(query)
        yield from (r[0] for r in rows) if ids_only else rows


class Batch(BatchingStrategy):
    """
    Batch implements UDF call batching, where each execution of a UDF
    is passed a sequence of multiple parameter sets.
    """

    is_batching = True

    def __init__(self, count: int):
        self.count = count

    def __call__(
        self,
        execute: Callable,
        query: sa.Select,
        id_col: sa.ColumnElement | None = None,
    ) -> Generator[RowsOutput, None, None]:
        from datachain.data_storage.warehouse import SELECT_BATCH_SIZE

        ids_only = False
        if id_col is not None:
            query = query.with_only_columns(id_col)
            ids_only = True

        # choose page size that is a multiple of the batch size
        page_size = math.ceil(SELECT_BATCH_SIZE / self.count) * self.count

        # select rows in batches
        results = []

        with contextlib.closing(execute(query, page_size=page_size)) as rows:
            for row in rows:
                results.append(row)
                if len(results) >= self.count:
                    batch, results = results[: self.count], results[self.count :]
                    yield [r[0] for r in batch] if ids_only else batch

            if len(results) > 0:
                yield [r[0] for r in results] if ids_only else results


class Partition(BatchingStrategy):
    """
    Partition implements UDF call batching, where each execution of a UDF
    is run on a list of dataset rows grouped by the specified column.
    Dataset rows need to be sorted by the grouping column.
    """

    is_batching = True

    def __call__(
        self,
        execute: Callable,
        query: sa.Select,
        id_col: sa.ColumnElement | None = None,
    ) -> Generator[RowsOutput, None, None]:
        if (partition_col := query.selected_columns.get(PARTITION_COLUMN_ID)) is None:
            raise RuntimeError("partition column not found in query")

        ids_only = False
        if id_col is not None:
            query = query.with_only_columns(id_col, partition_col)
            ids_only = True

        current_partition: int | None = None
        batch: list = []

        query_fields = [str(c.name) for c in query.selected_columns]
        id_column_idx = query_fields.index("sys__id")
        partition_column_idx = query_fields.index(PARTITION_COLUMN_ID)

        ordered_query = query.order_by(None).order_by(
            partition_col,
            *query._order_by_clauses,
        )

        with contextlib.closing(execute(ordered_query)) as rows:
            for row in rows:
                partition = row[partition_column_idx]
                if current_partition != partition:
                    current_partition = partition
                    if len(batch) > 0:
                        yield batch
                        batch = []
                batch.append(row[id_column_idx] if ids_only else row)

            if len(batch) > 0:
                yield batch
