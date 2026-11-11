from collections.abc import Iterator, Sequence
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


class McapFile(File):
    """`File` holding an MCAP robotics log (ROS 2, Foxglove)."""

    def summary(self, topics: str | Sequence[str] | None = None) -> McapSummary:
        """Read the summary section only, optionally counting just `topics`."""
        topics = _topic_list(topics)
        with self._open_read(block_size=PARTIAL_READ_BLOCK_SIZE, cache=False) as f:
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
        window = start is not None or end is not None
        with self.open(block_size=PARTIAL_READ_BLOCK_SIZE if window else None) as f:
            reader = _reader(f, decoder_factories)
            names = _topic_list(topics)
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
