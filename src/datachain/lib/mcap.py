import asyncio
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future
from contextlib import contextmanager
from typing import Any

from pydantic import Field, PrivateAttr

from datachain.lib.data_model import DataModel
from datachain.lib.file import PARTIAL_READ_BLOCK_SIZE, File, FileError


class McapTopic(DataModel):
    """A channel of an MCAP file: its topic, message count and encodings."""

    topic: str
    messages: int
    message_encoding: str
    schema_name: str
    schema_encoding: str


class McapSummary(DataModel):
    """Totals from the summary section of an MCAP file.

    Attributes:
        topics: One entry per channel.
        messages: Message count over `topics`.
        start: Log time of the first message in the file, in nanoseconds.
        end: Log time of the last message in the file, in nanoseconds.
        duration: `end - start`, in seconds.
    """

    topics: list[McapTopic]
    messages: int
    start: int
    end: int
    duration: float


class McapMessage(DataModel):
    """An MCAP message; times are in nanoseconds.

    `decoded` holds the decoded message when `messages()` got `decoder_factories`;
    it isn't saved with the dataset.
    """

    topic: str
    log_time: int
    publish_time: int
    data: bytes
    message_encoding: str
    schema_name: str
    schema_encoding: str
    _decoded: Any = PrivateAttr(default=None)

    @property
    def decoded(self) -> Any:
        return self._decoded


def _topic_list(topics: str | Sequence[str] | None) -> list[str] | None:
    if topics is None:
        return None
    return [topics] if isinstance(topics, str) else list(topics)


def _schema_fields(schema) -> dict[str, str]:
    if schema is None:
        return {"schema_name": "", "schema_encoding": ""}
    return {"schema_name": schema.name, "schema_encoding": schema.encoding}


def _reader(stream, decoder_factories=()):
    from mcap.reader import make_reader

    return make_reader(stream, decoder_factories=decoder_factories)


_RANGED_PROTOCOLS = ("s3", "gs", "az")
_MAX_CHUNKS_IN_FLIGHT = 8
_MAX_BYTES_IN_FLIGHT = 64 << 20


def _chunk_spans(summary, topics, start, end) -> list[tuple[int, int]]:
    """Byte ranges of chunks that may hold matches, in the reader's read order."""
    ids = None
    if topics is not None:
        ids = {i for i, ch in summary.channels.items() if ch.topic in topics}
    chunks = sorted(
        (
            c
            for c in summary.chunk_indexes
            if (start is None or c.message_end_time >= start)
            and (end is None or c.message_start_time < end)
            and (
                ids is None
                or not c.message_index_offsets
                or not ids.isdisjoint(c.message_index_offsets)
            )
        ),
        key=lambda c: (c.message_start_time, c.chunk_start_offset),
    )
    return [
        (c.chunk_start_offset, c.chunk_start_offset + c.chunk_length) for c in chunks
    ]


class _ChunkPrefetcher:
    """Seekable view of `f` that serves planned byte ranges from parallel reads.

    Ranges are fetched ahead in plan order, a bounded number at a time; any other
    read goes to `f`.
    """

    def __init__(self, f, fetch: Callable[[int, int], Future]):
        self._f = f
        self._fetch = fetch
        self._todo: deque[tuple[int, int]] = deque()
        self._window: deque[tuple[int, int, Future]] = deque()

    def plan(self, spans: list[tuple[int, int]]) -> None:
        self._todo.extend(spans)
        self._fill()

    def _fill(self) -> None:
        size = sum(e - s for s, e, _ in self._window)
        while self._todo and len(self._window) < _MAX_CHUNKS_IN_FLIGHT:
            s, e = self._todo[0]
            if self._window and size + e - s > _MAX_BYTES_IN_FLIGHT:
                break
            self._todo.popleft()
            self._window.append((s, e, self._fetch(s, e)))
            size += e - s

    def close(self) -> None:
        self._todo.clear()
        while self._window:
            self._window.popleft()[2].cancel()

    def read(self, n: int = -1) -> bytes:
        pos = self._f.tell()
        for i, (s, e, fut) in enumerate(self._window):
            if s <= pos and 0 <= n <= e - pos:
                data = fut.result()[pos - s : pos - s + n]
                for _ in range(i + (pos + n == e)):
                    self._window.popleft()[2].cancel()
                self._fill()
                self._f.seek(pos + n)
                return data
        return self._f.read(n)

    def __getattr__(self, name):
        return getattr(self._f, name)


class McapFile(File):
    """`File` holding an MCAP robotics log (ROS 2, Foxglove)."""

    def summary(self, topics: str | Sequence[str] | None = None) -> McapSummary:
        """Read the summary section only, optionally counting just `topics`."""
        topics = _topic_list(topics)
        with self.open(block_size=PARTIAL_READ_BLOCK_SIZE) as f:
            s = _reader(f).get_summary()
        st = s and s.statistics
        if st is None or (st.channel_count, st.schema_count) != (
            len(s.channels),
            len(s.schemas),
        ):
            raise FileError("MCAP file has no complete summary", self.source, self.path)
        counts = st.channel_message_counts
        channels = [
            McapTopic(
                topic=ch.topic,
                messages=counts.get(ch.id, 0),
                message_encoding=ch.message_encoding,
                **_schema_fields(s.schemas.get(ch.schema_id)),
            )
            for ch in s.channels.values()
            if topics is None or ch.topic in topics
        ]
        start, end = st.message_start_time, st.message_end_time
        return McapSummary(
            topics=channels,
            messages=sum(t.messages for t in channels),
            start=start,
            end=end,
            duration=(end - start) / 1e9,
        )

    def messages(
        self,
        topics: str | Sequence[str] | None = None,
        start: int | None = None,
        end: int | None = None,
        decoder_factories: Sequence[Any] = (),
    ) -> Iterator[McapMessage]:
        """Yield messages in log-time order, logged in `[start, end)` nanoseconds.

        Only the chunks that may hold matching messages are read. With
        `decoder_factories` (e.g. `mcap_protobuf.decoder.DecoderFactory()`), each
        message also carries its `decoded` object.
        """
        names = _topic_list(topics)
        with self._message_reader(names, start, end, decoder_factories) as reader:
            args = {"topics": names, "starting_at": start, "ending_before": end}
            rows = (
                reader.iter_decoded_messages(**args)
                if decoder_factories
                else ((*row, None) for row in reader.iter_messages(**args))
            )
            for schema, channel, msg, decoded in rows:
                message = McapMessage(
                    topic=channel.topic,
                    log_time=msg.log_time,
                    publish_time=msg.publish_time,
                    data=msg.data,
                    message_encoding=channel.message_encoding,
                    **_schema_fields(schema),
                )
                message._decoded = decoded
                yield message

    @contextmanager
    def _message_reader(self, topics, start, end, decoder_factories):
        client = self._catalog.get_client(self.source) if self._catalog else None
        if (
            client
            and client.protocol in _RANGED_PROTOCOLS
            and not (self.location or self._caching_enabled)
            and not client.cache.contains(self)
        ):
            with self.open(block_size=PARTIAL_READ_BLOCK_SIZE) as f:
                # The filesystem is created here, outside its event loop.
                loop = client.fs.loop
                view = _ChunkPrefetcher(
                    f,
                    lambda s, e: asyncio.run_coroutine_threadsafe(
                        client.read_range(self, s, e), loop
                    ),
                )
                reader = _reader(view, decoder_factories)
                summary = reader.get_summary()
                if summary is not None and summary.chunk_indexes:
                    view.plan(_chunk_spans(summary, topics, start, end))
                    try:
                        yield reader
                    finally:
                        view.close()
                    return
        window = start is not None or end is not None
        with self.open(block_size=PARTIAL_READ_BLOCK_SIZE if window else None) as f:
            yield _reader(f, decoder_factories)


class McapSlice(DataModel):
    """A `[start, end)` log-time window of an MCAP file, in nanoseconds.

    Attributes:
        mcap: The MCAP file.
        start: Window start, inclusive.
        end: Window end, exclusive.
        topics: Topics `messages()` returns by default; empty means all.
    """

    mcap: McapFile
    start: int
    end: int
    topics: list[str] = Field(default_factory=list)

    def messages(
        self,
        topics: str | Sequence[str] | None = None,
        decoder_factories: Sequence[Any] = (),
    ) -> Iterator[McapMessage]:
        """Yield the window's messages, of `topics` or else of `self.topics`."""
        if topics is None:
            topics = self.topics or None
        return self.mcap.messages(topics, self.start, self.end, decoder_factories)
