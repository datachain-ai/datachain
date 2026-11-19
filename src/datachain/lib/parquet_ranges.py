import io
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.spec import AbstractBufferedFile

if TYPE_CHECKING:
    from datachain.client import Client
    from datachain.lib.file import File

# Footer sample read upfront so metadata and trailing chunks take one GET.
PARQUET_FOOTER_SAMPLE_SIZE = 1024 * 1024
# Column chunk ranges closer than this are coalesced into one request.
PARQUET_RANGE_GAP = 64 * 1024
# Per-reader cap on prefetched Parquet bytes; larger reads fall back to streaming.
PARQUET_RANGE_MAX_BYTES = 64 * 1024 * 1024
PARQUET_RANGE_THREADS = 8


def _leaf_names(field: pa.Field) -> list[str]:
    if not field.type.num_fields:
        return [field.name]
    return [
        name
        for i in range(field.type.num_fields)
        for name in _leaf_names(field.type.field(i))
    ]


def column_ranges(
    metadata: pq.FileMetaData,
    columns: Sequence[str],
    gap: int = PARQUET_RANGE_GAP,
) -> list[tuple[int, int]]:
    """Byte ranges of the column chunks for the top-level Arrow `columns`."""
    pairs = [
        (field.name, leaf)
        for field in metadata.schema.to_arrow_schema()
        for leaf in _leaf_names(field)
    ]
    wanted = set(columns)
    indices = [i for i, (top, _) in enumerate(pairs) if top in wanted]
    ranges = sorted(
        (start, start + col.total_compressed_size)
        for r in range(metadata.num_row_groups)
        for group in (metadata.row_group(r),)
        for i in indices
        if (col := group.column(i)).total_compressed_size > 0
        for start in (col.dictionary_page_offset or col.data_page_offset,)
    )
    coalesced: list[tuple[int, int]] = []
    for start, end in ranges:
        if coalesced and start - coalesced[-1][1] <= gap:
            coalesced[-1] = (coalesced[-1][0], max(coalesced[-1][1], end))
        else:
            coalesced.append((start, end))
    return coalesced


def read_parquet_parts(
    client: "Client",
    file: "File",
    columns: Sequence[str],
    reserve: Callable[[int], bool],
) -> tuple[int, list[tuple[int, bytes]]]:
    """Read the Parquet footer sample and selected column chunks, then close."""
    with client.open_object(
        file, use_cache=False, cb=file._download_cb, block_size=0
    ) as stream:
        size = stream.seek(0, io.SEEK_END)
        sample_start = max(0, size - PARQUET_FOOTER_SAMPLE_SIZE)
        stream.seek(sample_start)
        sample = stream.read(size - sample_start)
        if len(sample) < 8 or sample[-4:] != b"PAR1":
            return size, [(sample_start, sample)]
        footer_start = size - 8 - int.from_bytes(sample[-8:-4], "little")
        if 0 <= footer_start < sample_start:
            if size - footer_start > PARQUET_RANGE_MAX_BYTES:
                return size, []
            stream.seek(footer_start)
            sample = stream.read(sample_start - footer_start) + sample
            sample_start = footer_start
        try:
            ranges = column_ranges(pq.read_metadata(pa.BufferReader(sample)), columns)
        except (pa.ArrowException, OSError):
            ranges = []
        spans = [
            (start, min(end, sample_start))
            for start, end in ranges
            if start < sample_start
        ]
        span_bytes = sum(end - start for start, end in spans)
        if ranges and reserve(len(sample) + span_bytes):
            raw: Any = getattr(stream, "fileobj", stream)
            cb = file._download_cb

            def fetch(span: tuple[int, int]) -> tuple[int, bytes]:
                data = raw._fetch_range(*span)
                cb.relative_update(len(data))
                return span[0], data

            if len(spans) <= 1:
                parts = [fetch(span) for span in spans]
            else:
                with ThreadPoolExecutor(min(len(spans), PARQUET_RANGE_THREADS)) as pool:
                    parts = list(pool.map(fetch, spans))
            if parts and parts[-1][0] + len(parts[-1][1]) == sample_start:
                parts[-1] = (parts[-1][0], parts[-1][1] + sample)
            else:
                parts.append((sample_start, sample))
            return size, parts
        if reserve(len(sample)):
            return size, [(sample_start, sample)]
        return size, []


class ParquetBufferedFile(AbstractBufferedFile):
    """Serves prefetched Parquet byte ranges, opening the remote file on miss."""

    def __init__(
        self,
        client: "Client",
        file: "File",
        size: int,
        parts: Sequence[tuple[int, bytes]],
    ):
        super().__init__(
            client.fs,
            file.get_fs_path(),
            size=size,
            block_size=0,
            cache_type="none",
        )
        remote: Any = None

        # A closure, so the cache holds no reference to self and parts free on close.
        def fetch(start: int, end: int) -> bytes:
            nonlocal remote
            end = min(end, size)
            for part_start, part in parts:
                if part_start <= start and end <= part_start + len(part):
                    return part[start - part_start : end - part_start]
            if remote is None:
                remote = client.open_object(
                    file, use_cache=False, cb=file._download_cb, block_size=0
                )
            remote.seek(start)
            return remote.read(end - start)

        def close_remote() -> None:
            nonlocal remote
            if remote is not None:
                remote.close()
                remote = None

        self.cache.fetcher = fetch
        self._close_remote = close_remote

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._close_remote()
