import os
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from datachain.lib.file import File
from datachain.lib.udf import Generator
from datachain.query import Session

if TYPE_CHECKING:
    from .datachain import DataChain


def read_mcap(
    path: str | os.PathLike[str] | list[str] | list[os.PathLike[str]],
    topics: str | Sequence[str] | None = None,
    duration: float | None = None,
    session: Session | None = None,
    settings: dict | None = None,
    **kwargs: Any,
) -> "DataChain":
    """Generate a chain from MCAP robotics logs (ROS 2, Foxglove), `*.mcap` files.

    By default it emits one row per file: `mcap` (`McapFile`) and `summary`
    (`McapSummary`), read from the file's summary section only. With `duration`
    it emits one `mcap` (`McapSlice`) row per `duration` seconds of log time,
    starting at the file's first message.

    Parameters:
        path: Storage path(s) or URI(s), e.g. `s3://bucket/logs/`. Supports globs.
        topics: Topics the summary counts and slice `messages()` return.
            Files with no messages on these topics give no slices.
        duration: Slice length in seconds.
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        kwargs: Parameters to pass to `read_storage`.

    Example:
        ```py
        import datachain as dc

        chain = dc.read_mcap("s3://bucket/logs/", topics="/joint_states", duration=10)
        for (window,) in chain.limit(1).to_iter("mcap"):
            for msg in window.messages():
                print(msg.topic, msg.log_time, len(msg.data))
        ```
    """
    from datachain.lib.mcap import McapFile, McapSlice, McapSummary

    from .datachain import C
    from .storage import read_storage

    step = None if duration is None else round(duration * 1e9)
    if step is not None and step <= 0:
        raise ValueError("duration must be positive")
    if isinstance(topics, str):
        topics = [topics]
    chain = read_storage(path, session=session, settings=settings, **kwargs)
    chain = chain.filter(C("file.path").glob("*.mcap"))

    def rows(file: File) -> Iterator[Any]:
        mcap = McapFile(**file.model_dump())
        mcap._set_stream(file._catalog, caching_enabled=file._caching_enabled)
        summary = mcap.summary(topics)
        if step is None:
            yield mcap, summary
        elif summary.messages:
            for start in range(summary.start, summary.end + 1, step):
                yield McapSlice(
                    mcap=mcap, start=start, end=start + step, topics=list(topics or [])
                )

    output: dict[str, Any] = (
        {"mcap": McapSlice} if step else {"mcap": McapFile, "summary": McapSummary}
    )
    return chain.gen(_SummaryOnly(rows), output=output)


class _SummaryOnly(Generator):
    """Reads only each file's summary section, so large files aren't prefetched."""

    def __init__(self, rows):
        from datachain.lib.arrow import PARTIAL_READ_PREFETCH_MAX_SIZE

        super().__init__()
        self.process = rows  # type: ignore[method-assign]
        self.prefetch_max_size = PARTIAL_READ_PREFETCH_MAX_SIZE
