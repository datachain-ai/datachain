from collections.abc import Iterator, Sequence

from pydantic import Field

from datachain.lib.data_model import DataModel
from datachain.lib.file import File, FileError


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
    """A raw MCAP message; times are in nanoseconds."""

    topic: str
    log_time: int
    publish_time: int
    data: bytes
    message_encoding: str
    schema_name: str
    schema_encoding: str


def _topic_list(topics: str | Sequence[str] | None) -> list[str] | None:
    if topics is None:
        return None
    return [topics] if isinstance(topics, str) else list(topics)


def _schema_fields(schema) -> dict[str, str]:
    if schema is None:
        return {"schema_name": "", "schema_encoding": ""}
    return {"schema_name": schema.name, "schema_encoding": schema.encoding}


def _reader(stream):
    try:
        from mcap.reader import make_reader
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies for reading MCAP files.\n"
            "To install run:\n\n"
            "  pip install 'datachain[mcap]'\n"
        ) from exc
    return make_reader(stream)


class McapFile(File):
    """`File` holding an MCAP robotics log (ROS 2, Foxglove)."""

    def summary(self, topics: str | Sequence[str] | None = None) -> McapSummary:
        """Read the summary section only, optionally counting just `topics`."""
        topics = _topic_list(topics)
        with self.open() as f:
            s = _reader(f).get_summary()
        if s is None or s.statistics is None:
            raise FileError(
                "MCAP file has no summary statistics", self.source, self.path
            )
        counts = s.statistics.channel_message_counts
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
        start, end = s.statistics.message_start_time, s.statistics.message_end_time
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
    ) -> Iterator[McapMessage]:
        """Yield messages in log-time order, logged in `[start, end)` nanoseconds.

        Only the chunks that may hold matching messages are read.
        """
        with self.open() as f:
            for schema, channel, msg in _reader(f).iter_messages(
                _topic_list(topics), starting_at=start, ending_before=end
            ):
                yield McapMessage(
                    topic=channel.topic,
                    log_time=msg.log_time,
                    publish_time=msg.publish_time,
                    data=msg.data,
                    message_encoding=channel.message_encoding,
                    **_schema_fields(schema),
                )


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
        self, topics: str | Sequence[str] | None = None
    ) -> Iterator[McapMessage]:
        """Yield the window's messages, of `topics` or else of `self.topics`."""
        if topics is None:
            topics = self.topics or None
        return self.mcap.messages(topics, self.start, self.end)


def to_mcap_file(file: File) -> McapFile:
    mcap = McapFile(**file.model_dump())
    mcap._set_stream(file._catalog, caching_enabled=file._caching_enabled)
    return mcap
