"""Catalog MCAP topics and log-time ranges, then export a recording selection.

Install: pip install datachain mcap
Run from a checkout: python examples/robotics/mcap_inventory.py --demo
For real data, supply one or more local/cloud URI patterns instead of --demo.
"""

import argparse
import io
from pathlib import Path

from mcap.reader import NonSeekingReader, make_reader
from pydantic import BaseModel

import datachain as dc


class RecordingInfo(BaseModel):
    topics: list[str]
    message_count: int
    first_log_time_ns: int | None
    last_log_time_ns: int | None


def inspect_mcap(file: dc.File) -> RecordingInfo:
    with file.open() as raw, io.BufferedReader(raw) as stream:
        reader = make_reader(stream)
        summary = reader.get_summary()
        stats = summary.statistics if summary is not None else None
        counts = stats.channel_message_counts if stats is not None else {}
        if (
            summary is not None
            and stats is not None
            and sum(counts.values()) == stats.message_count
            and all(channel_id in summary.channels for channel_id in counts)
        ):
            topics = sorted(
                {
                    summary.channels[channel_id].topic
                    for channel_id, count in stats.channel_message_counts.items()
                    if count
                }
            )
            count = stats.message_count
            first = stats.message_start_time if count else None
            last = stats.message_end_time if count else None
        else:
            seen = set()
            count = 0
            first = last = None
            stream.seek(0)
            for _, channel, message in NonSeekingReader(stream).iter_messages(
                log_time_order=False
            ):
                seen.add(channel.topic)
                count += 1
                first = (
                    message.log_time if first is None else min(first, message.log_time)
                )
                last = message.log_time if last is None else max(last, message.log_time)
            topics = sorted(seen)
    return RecordingInfo(
        topics=topics,
        message_count=count,
        first_log_time_ns=first,
        last_log_time_ns=last,
    )


def make_demo(directory: Path) -> str:
    from mcap.writer import Writer

    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "complete.mcap": ["/camera/front", "/joint_states"],
        "missing_joints.mcap": ["/camera/front"],
    }
    for name, topics in files.items():
        stream = io.BytesIO()
        writer = Writer(stream)
        writer.start()
        for topic in topics:
            channel = writer.register_channel(topic, "json", schema_id=0)
            for second in [1, 2]:
                stamp = second * 1_000_000_000
                writer.add_message(channel, stamp, b'{"synthetic":true}', stamp)
        writer.finish()
        data = stream.getvalue()
        path = directory / name
        if path.exists():
            if path.read_bytes() != data:
                raise FileExistsError(f"Demo path contains different data: {path}")
        else:
            path.write_bytes(data)
    return str(directory / "*.mcap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", help="Local/cloud MCAP URI patterns")
    parser.add_argument(
        "--demo", action="store_true", help="Create two tiny synthetic MCAP files"
    )
    parser.add_argument(
        "--required-topic", action="append", help="Repeat for each required topic"
    )
    args = parser.parse_args()
    if args.demo and args.sources:
        parser.error("Choose --demo or source URIs")
    sources = [make_demo(Path("mcap-demo"))] if args.demo else args.sources
    if not sources:
        parser.error("Supply source URIs or --demo")
    required = args.required_topic or ["/camera/front", "/joint_states"]
    recordings = (
        dc.read_storage(sources)
        .settings(parallel=True, prefetch=0)
        .map(recording=inspect_mcap)
        .save("recordings")
    )
    recordings.select("file.path", "recording.topics", "recording.message_count").show()
    selected = recordings
    for topic in required:
        selected = selected.filter(dc.func.array.contains("recording.topics", topic))
    selected = selected.save("training_recordings")
    selected.select(
        "file.source",
        "file.path",
        "file.etag",
        "file.version",
        "recording.first_log_time_ns",
        "recording.last_log_time_ns",
    ).to_parquet("training_manifest.parquet")
    print(f"Catalogued {recordings.count()} recordings; selected {selected.count()}.")


if __name__ == "__main__":
    main()
