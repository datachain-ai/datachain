from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Sequence
from itertools import groupby
from operator import itemgetter

import sqlalchemy as sa

from datachain.data_storage.schema import PARTITION_COLUMN_ID
from datachain.utils import batched, safe_closing

DEFAULT_BATCH_SIZE = 10_000

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
        if id_col is not None:
            query = query.with_only_columns(id_col)
        with safe_closing(execute(query)) as rows:
            yield from (r[0] for r in rows) if id_col is not None else rows


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
        with safe_closing(NoBatching()(execute, query, id_col)) as rows:
            yield from (list(b) for b in batched(rows, self.count))


class Partition(BatchingStrategy):
    """
    Partition implements UDF call batching, where each execution of a UDF
    is run on a list of dataset rows grouped by the specified column.
    `execute` must return the rows sorted by the partition id.
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
        if id_col is not None:
            query = query.with_only_columns(id_col, partition_col)
        names = list(query.selected_columns.keys())
        partition_of = itemgetter(names.index(PARTITION_COLUMN_ID))
        id_at = names.index("sys__id")
        # execute pages rows in partition order (dataset_select_paginated)
        with safe_closing(execute(query)) as rows:
            for _, group in groupby(rows, partition_of):
                batch = list(group)
                yield [r[id_at] for r in batch] if id_col is not None else batch
