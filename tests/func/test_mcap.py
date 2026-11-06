import io
import tarfile

import pytest
from mcap.writer import Writer as McapWriter

import datachain as dc
from datachain.lib.file import FileError
from datachain.lib.mcap import McapFile, McapSlice
from datachain.lib.tar import process_tar

T0 = 1_700_000_000_000_000_000
SEC = 1_000_000_000
# /a: 1 s apart over 0..9 s, /b: 2.5 s apart over 0..7.5 s
A_TIMES = [T0 + i * SEC for i in range(10)]
B_TIMES = [T0 + i * 2_500_000_000 for i in range(4)]


def mcap_bytes(**writer_options) -> bytes:
    buf = io.BytesIO()
    w = McapWriter(buf, chunk_size=64, **writer_options)
    w.start()
    schema = w.register_schema("A", "jsonschema", b"{}")
    a = w.register_channel("/a", "json", schema)
    b = w.register_channel("/b", "raw", 0)
    msgs = [(t, a, b"a%d" % i) for i, t in enumerate(A_TIMES)]
    msgs += [(t, b, b"b%d" % i) for i, t in enumerate(B_TIMES)]
    for t, ch, data in sorted(msgs):
        w.add_message(ch, log_time=t, data=data, publish_time=t + 1)
    w.finish()
    return buf.getvalue()


@pytest.fixture
def mcap_dir(tmp_dir):
    (tmp_dir / "ep1.mcap").write_bytes(mcap_bytes())
    (tmp_dir / "ep2.mcap").write_bytes(mcap_bytes())
    (tmp_dir / "notes.txt").write_text("not mcap")
    return tmp_dir


@pytest.fixture
def mcap_file(mcap_dir, test_session):
    return (
        dc.read_mcap(mcap_dir.as_uri(), session=test_session)
        .order_by("mcap.path")
        .to_values("mcap")[0]
    )


def times(messages):
    return [(m.topic, m.log_time - T0) for m in messages]


def test_written_file_has_several_chunks(mcap_file):
    from mcap.reader import make_reader

    with mcap_file.open() as f:
        assert make_reader(f).get_summary().statistics.chunk_count > 3


def test_summary(mcap_file):
    summary = mcap_file.summary()
    assert [tuple(t.model_dump().values()) for t in summary.topics] == [
        ("/a", 10, "json", "A", "jsonschema"),
        ("/b", 4, "raw", "", ""),
    ]
    assert (summary.messages, summary.start, summary.end, summary.duration) == (
        14,
        T0,
        T0 + 9 * SEC,
        9.0,
    )


@pytest.mark.parametrize("topics", ["/b", ["/b"], ("/b", "/missing")])
def test_summary_of_topics(mcap_file, topics):
    summary = mcap_file.summary(topics)
    assert [t.topic for t in summary.topics] == ["/b"]
    assert summary.messages == 4
    assert (summary.start, summary.end) == (T0, T0 + 9 * SEC)


def test_summary_of_no_topics(mcap_file):
    summary = mcap_file.summary([])
    assert (summary.topics, summary.messages) == ([], 0)


def as_mcap(file):
    mcap = McapFile(**file.model_dump())
    mcap._set_stream(file._catalog)
    return mcap


@pytest.mark.parametrize(
    "options", [{"use_statistics": False}, {"repeat_channels": False}]
)
def test_summary_needs_complete_summary_section(tmp_dir, test_session, options):
    (tmp_dir / "bare.mcap").write_bytes(mcap_bytes(**options))
    (file,) = dc.read_storage(tmp_dir.as_uri(), session=test_session).to_values("file")
    with pytest.raises(FileError, match="no complete summary"):
        as_mcap(file).summary()


def test_mcap_inside_tar(tmp_dir, test_session):
    data = mcap_bytes()
    with tarfile.open(tmp_dir / "logs.tar", "w") as tar:
        info = tarfile.TarInfo("ep.mcap")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    chain = dc.read_storage((tmp_dir / "logs.tar").as_uri(), session=test_session)
    (member,) = chain.gen(file=process_tar).to_values("file")
    mcap = as_mcap(member)
    assert mcap.summary().messages == 14
    assert [m.data for m in mcap.messages("/b")] == [b"b0", b"b1", b"b2", b"b3"]


def test_messages_in_log_time_order(mcap_file):
    msgs = list(mcap_file.messages())
    assert len(msgs) == 14
    assert [m.log_time for m in msgs] == sorted(A_TIMES + B_TIMES)
    first_a, first_b = msgs[0], msgs[1]
    assert first_a.model_dump() == {
        "topic": "/a",
        "log_time": T0,
        "publish_time": T0 + 1,
        "data": b"a0",
        "message_encoding": "json",
        "schema_name": "A",
        "schema_encoding": "jsonschema",
    }
    assert (first_b.topic, first_b.data, first_b.schema_name) == ("/b", b"b0", "")


def test_messages_by_topic_and_time(mcap_file):
    assert times(mcap_file.messages("/b")) == [
        ("/b", 0),
        ("/b", 2_500_000_000),
        ("/b", 5 * SEC),
        ("/b", 7_500_000_000),
    ]
    assert times(mcap_file.messages(start=T0 + 5 * SEC, end=T0 + 7 * SEC)) == [
        ("/a", 5 * SEC),
        ("/b", 5 * SEC),
        ("/a", 6 * SEC),
    ]
    assert times(mcap_file.messages(["/a"], T0 + 8 * SEC + 1)) == [("/a", 9 * SEC)]
    assert list(mcap_file.messages(end=T0)) == []
    assert list(mcap_file.messages([])) == []


def test_read_mcap_one_row_per_file(mcap_dir, test_session):
    chain = dc.read_mcap(mcap_dir.as_uri(), session=test_session)
    assert set(chain.schema) == {"mcap", "summary"}
    rows = chain.order_by("mcap.path").to_list("mcap", "summary")
    assert [type(m) for m, _ in rows] == [McapFile, McapFile]
    assert [m.path for m, _ in rows] == ["ep1.mcap", "ep2.mcap"]
    assert [(s.messages, s.duration) for _, s in rows] == [(14, 9.0), (14, 9.0)]
    assert len(list(rows[1][0].messages())) == 14


def test_read_mcap_topics_filter_summary(mcap_dir, test_session):
    chain = dc.read_mcap(mcap_dir.as_uri(), topics="/a", session=test_session)
    assert chain.to_values("summary.messages") == [10, 10]


@pytest.mark.parametrize(
    "duration,starts",
    [(4, [0, 4, 8]), (3, [0, 3, 6, 9]), (9, [0, 9]), (9.5, [0]), (100, [0])],
)
def test_read_mcap_slices(mcap_dir, test_session, duration, starts):
    chain = dc.read_mcap(
        f"{mcap_dir.as_uri()}/ep1.mcap", duration=duration, session=test_session
    )
    slices = chain.order_by("mcap.start").to_values("mcap")
    step = round(duration * SEC)
    assert [(s.start - T0, s.end - T0) for s in slices] == [
        (t * SEC, t * SEC + step) for t in starts
    ]
    assert all(type(s) is McapSlice and s.topics == [] for s in slices)
    msgs = [m for s in slices for m in s.messages()]
    assert [m.log_time for m in msgs] == sorted(A_TIMES + B_TIMES)


def test_read_mcap_slice_messages(mcap_dir, test_session):
    chain = dc.read_mcap(
        f"{mcap_dir.as_uri()}/ep1.mcap", duration=4, session=test_session
    )
    middle = chain.order_by("mcap.start").to_values("mcap")[1]
    assert times(middle.messages()) == [
        ("/a", 4 * SEC),
        ("/a", 5 * SEC),
        ("/b", 5 * SEC),
        ("/a", 6 * SEC),
        ("/a", 7 * SEC),
        ("/b", 7_500_000_000),
    ]
    assert times(middle.messages("/b")) == [("/b", 5 * SEC), ("/b", 7_500_000_000)]


def test_read_mcap_slices_of_topics(mcap_dir, test_session):
    chain = dc.read_mcap(
        mcap_dir.as_uri(), topics=["/b"], duration=4, session=test_session
    )
    slices = chain.order_by("mcap.mcap.path", "mcap.start").to_values("mcap")
    assert len(slices) == 6
    assert all(s.topics == ["/b"] for s in slices)
    assert times(slices[1].messages()) == [("/b", 5 * SEC), ("/b", 7_500_000_000)]
    assert times(slices[2].messages()) == []
    assert len(list(slices[1].messages(["/a"]))) == 4


def test_read_mcap_slices_of_missing_topic(mcap_dir, test_session):
    chain = dc.read_mcap(
        mcap_dir.as_uri(), topics="/missing", duration=4, session=test_session
    )
    assert chain.count() == 0


@pytest.mark.parametrize("duration", [0, -1, 1e-10])
def test_read_mcap_rejects_non_positive_duration(mcap_dir, duration):
    with pytest.raises(ValueError, match="duration must be positive"):
        dc.read_mcap(mcap_dir.as_uri(), duration=duration)


def test_read_mcap_slices_survive_save(mcap_dir, test_session):
    dc.read_mcap(mcap_dir.as_uri(), topics="/a", duration=5, session=test_session).save(
        "mcap_slices"
    )
    slices = (
        dc.read_dataset("mcap_slices", session=test_session)
        .order_by("mcap.mcap.path", "mcap.start")
        .to_values("mcap")
    )
    assert [(s.mcap.path, s.start - T0, s.end - T0) for s in slices] == [
        ("ep1.mcap", 0, 5 * SEC),
        ("ep1.mcap", 5 * SEC, 10 * SEC),
        ("ep2.mcap", 0, 5 * SEC),
        ("ep2.mcap", 5 * SEC, 10 * SEC),
    ]
    assert type(slices[0].mcap) is McapFile
    assert [m.data for m in slices[3].messages()] == [b"a5", b"a6", b"a7", b"a8", b"a9"]


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("version_aware", [False], indirect=True)
def test_read_mcap_from_cloud_without_cache(cloud_test_catalog, cloud_prefix):
    ctc = cloud_test_catalog
    (ctc.src / cloud_prefix / "ep.mcap").write_bytes(mcap_bytes())
    uri = f"{ctc.src_uri}/{cloud_prefix}"

    chain = dc.read_mcap(
        uri, duration=4, session=ctc.session, settings={"cache": False, "prefetch": 0}
    )
    counts = chain.order_by("mcap.start").map(
        n=lambda mcap: len(list(mcap.messages())), output=int
    )
    assert counts.to_values("n") == [6, 6, 2]

    (summary,) = dc.read_mcap(uri, session=ctc.session).to_values("summary")
    assert summary.messages == 14


def test_summary_and_windows_read_small_blocks(mcap_file, mocker):
    from datachain.lib.file import PARTIAL_READ_BLOCK_SIZE

    spy = mocker.spy(McapFile, "open")
    mcap_file.summary()
    list(mcap_file.messages(start=T0))
    list(mcap_file.messages())

    sizes = [call.kwargs.get("block_size") for call in spy.call_args_list]
    assert sizes == [PARTIAL_READ_BLOCK_SIZE, PARTIAL_READ_BLOCK_SIZE, None]


def test_read_mcap_does_not_prefetch_large_files():
    from datachain.lib.arrow import PARTIAL_READ_PREFETCH_MAX_SIZE
    from datachain.lib.dc.mcap import _SummaryOnly

    assert _SummaryOnly(lambda file: iter(())).prefetch_max_size == (
        PARTIAL_READ_PREFETCH_MAX_SIZE
    )


class UpperDecoder:
    def decoder_for(self, message_encoding, schema):
        return bytes.upper if message_encoding == "json" else None


def test_messages_decoded_by_factories(mcap_file):
    factories = [UpperDecoder()]
    msgs = list(mcap_file.messages("/a", end=T0 + 2 * SEC, decoder_factories=factories))
    assert [m.decoded for m in msgs] == [b"A0", b"A1"]
    assert [m.decoded for m in mcap_file.messages("/a", end=T0 + SEC)] == [None]


def many_chunks_mcap(**writer_options) -> bytes:
    buf = io.BytesIO()
    w = mcap_writer.Writer(buf, chunk_size=256, **writer_options)
    w.start()
    a = w.register_channel("/a", "json", w.register_schema("A", "jsonschema", b"{}"))
    b = w.register_channel("/b", "raw", 0)
    msgs = [(T0 + i * SEC // 10, a, b"a%d" % i) for i in range(200)]
    msgs += [(T0 + i * SEC, b, b"b%d" % i * 20) for i in range(20)]
    msgs.sort()
    # Reversed runs make chunks overlap in log time.
    for k in range(0, len(msgs), 7):
        for t, ch, data in reversed(msgs[k : k + 7]):
            w.add_message(ch, log_time=t, data=data, publish_time=t + 1)
    w.finish()
    return buf.getvalue()


def expected_messages(data, topics, start, end, decoder_factories=()):
    from mcap.reader import make_reader

    reader = make_reader(io.BytesIO(data), decoder_factories=decoder_factories)
    args = {"starting_at": start, "ending_before": end}
    if decoder_factories:
        return [
            (c.topic, m.log_time, m.publish_time, m.data, d)
            for _, c, m, d in reader.iter_decoded_messages(topics, **args)
        ]
    return [
        (c.topic, m.log_time, m.publish_time, m.data, None)
        for _, c, m in reader.iter_messages(topics, **args)
    ]


def as_tuples(messages):
    return [(m.topic, m.log_time, m.publish_time, m.data, m.decoded) for m in messages]


def cloud_mcap(ctc, prefix, data):
    (ctc.src / prefix / "ep.mcap").write_bytes(data)
    chain = dc.read_storage(f"{ctc.src_uri}/{prefix}", session=ctc.session)
    (file,) = chain.to_values("file")
    return as_mcap(file)


def spy_range_reads(mocker, mcap):
    client = mcap._catalog.get_client(mcap.source)
    return mocker.spy(type(client), "read_range")


def read_ranges(spy):
    return sorted(call.args[2:] for call in spy.call_args_list)


WINDOWS = [
    (None, None, None),
    ("/b", None, None),
    (["/a"], T0 + 3 * SEC, T0 + 7 * SEC),
    (["/a", "/b"], T0 + 5 * SEC, T0 + 5 * SEC + 1),
    ("/b", T0 + 11 * SEC + 1, T0 + 13 * SEC),
    (None, T0 + 19 * SEC, None),
    (None, None, T0),
    ([], None, None),
    ("/missing", T0, T0 + 5 * SEC),
]


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("version_aware", [False], indirect=True)
def test_cloud_messages_match_sequential_reader(
    cloud_test_catalog, cloud_prefix, mocker
):
    data = many_chunks_mcap()
    mcap = cloud_mcap(cloud_test_catalog, cloud_prefix, data)
    spy = spy_range_reads(mocker, mcap)

    for topics, start, end in WINDOWS:
        names = [topics] if isinstance(topics, str) else topics
        expected = expected_messages(data, names, start, end)
        assert as_tuples(mcap.messages(topics, start, end)) == expected
    assert spy.call_count > 0

    factories = [UpperDecoder()]
    msgs = mcap.messages("/a", T0 + SEC, T0 + 2 * SEC, decoder_factories=factories)
    expected = expected_messages(data, ["/a"], T0 + SEC, T0 + 2 * SEC, factories)
    assert as_tuples(msgs) == expected


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [False], indirect=True)
@pytest.mark.parametrize(
    "options",
    [
        {"compression": mcap_writer.CompressionType.NONE},
        {"index_types": mcap_writer.IndexType.CHUNK},
        {"index_types": mcap_writer.IndexType.NONE},
        {"use_chunking": False},
    ],
)
def test_cloud_messages_of_other_layouts(
    cloud_test_catalog, cloud_prefix, mocker, options
):
    data = many_chunks_mcap(**options)
    mcap = cloud_mcap(cloud_test_catalog, cloud_prefix, data)
    for topics, start, end in WINDOWS:
        names = [topics] if isinstance(topics, str) else topics
        expected = expected_messages(data, names, start, end)
        assert as_tuples(mcap.messages(topics, start, end)) == expected


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [False], indirect=True)
def test_cloud_messages_fetch_only_needed_chunks(
    cloud_test_catalog, cloud_prefix, mocker
):
    from mcap.reader import make_reader

    data = many_chunks_mcap()
    mcap = cloud_mcap(cloud_test_catalog, cloud_prefix, data)
    summary = make_reader(io.BytesIO(data)).get_summary()
    b_id = next(i for i, ch in summary.channels.items() if ch.topic == "/b")
    start, end = T0 + 5 * SEC, T0 + 9 * SEC
    needed = sorted(
        (c.chunk_start_offset, c.chunk_start_offset + c.chunk_length)
        for c in summary.chunk_indexes
        if c.message_end_time >= start
        and c.message_start_time < end
        and b_id in c.message_index_offsets
    )
    assert 0 < len(needed) < len(summary.chunk_indexes) / 4

    spy = spy_range_reads(mocker, mcap)
    assert [m.data for m in mcap.messages("/b", start, end)] == [
        b"b%d" % i * 20 for i in range(5, 9)
    ]
    assert read_ranges(spy) == needed

    spy.reset_mock()
    assert len(list(mcap.messages())) == 220
    assert len(read_ranges(spy)) == len(summary.chunk_indexes)


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [False], indirect=True)
def test_cloud_messages_fetch_a_bounded_window_ahead(
    cloud_test_catalog, cloud_prefix, mocker
):
    mocker.patch("datachain.lib.mcap._MAX_CHUNKS_IN_FLIGHT", 2)
    mcap = cloud_mcap(cloud_test_catalog, cloud_prefix, many_chunks_mcap())
    spy = spy_range_reads(mocker, mcap)

    messages = mcap.messages()
    assert next(messages).log_time == T0
    messages.close()
    assert 2 <= spy.call_count <= 4


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [False], indirect=True)
def test_cloud_messages_of_cached_file_read_the_cache(
    cloud_test_catalog, cloud_prefix, mocker
):
    data = many_chunks_mcap()
    mcap = cloud_mcap(cloud_test_catalog, cloud_prefix, data)
    mcap._set_stream(mcap._catalog, caching_enabled=True)
    spy = spy_range_reads(mocker, mcap)

    assert as_tuples(mcap.messages("/b")) == expected_messages(data, ["/b"], None, None)
    assert mcap.get_local_path()
    assert spy.call_count == 0


@pytest.mark.parametrize("cloud_type", ["s3", "gs"], indirect=True)
@pytest.mark.parametrize("version_aware", [True], indirect=True)
def test_cloud_messages_read_the_listed_version(cloud_test_catalog, cloud_prefix):
    old = many_chunks_mcap()
    mcap = cloud_mcap(cloud_test_catalog, cloud_prefix, old)
    (cloud_test_catalog.src / cloud_prefix / "ep.mcap").write_bytes(mcap_bytes())

    assert mcap.version
    expected = expected_messages(old, ["/b"], T0 + 3 * SEC, None)
    assert as_tuples(mcap.messages("/b", T0 + 3 * SEC)) == expected
