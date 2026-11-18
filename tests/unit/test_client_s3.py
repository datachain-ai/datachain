import inspect
import pickle
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock, MagicMock

import botocore.session
import pytest
from botocore.exceptions import ClientError
from botocore.utils import parse_timestamp
from fsspec.asyn import get_loop, sync

from datachain.client import s3 as s3_mod
from datachain.client.s3 import (
    _FIELDS,
    ClientS3,
    _fields,
    _handle_200_error,
    _parse_timestamp,
    _rows_page,
)
from datachain.client.writeconfig import WriteConfig
from datachain.node import DirType, Node
from datachain.nodes_thread_pool import NodeChunk


@pytest.mark.parametrize("streaming", [True, False])
def test_write_kwargs_maps_all_fields(streaming):
    # s3fs forwards extra kwargs into s3_additional_kwargs (boto3 put_object /
    # multipart args), including the raw write_options escape hatch.
    cfg = WriteConfig(
        content_type="application/pdf",
        content_disposition="attachment",
        cache_control="max-age=3600",
        content_encoding="gzip",
        metadata={"a": "b"},
        write_options={"ACL": "public-read"},
    )
    assert ClientS3._write_kwargs(cfg, streaming=streaming) == {
        "ContentType": "application/pdf",
        "ContentDisposition": "attachment",
        "CacheControl": "max-age=3600",
        "ContentEncoding": "gzip",
        "Metadata": {"a": "b"},
        "ACL": "public-read",
    }


def test_write_kwargs_empty_config_maps_nothing():
    assert ClientS3._write_kwargs(WriteConfig(), streaming=False) == {}


@pytest.fixture
def nodes():
    return iter(
        [
            make_size_node(11, DirType.DIR, "a/f1", 100),
            make_size_node(12, DirType.FILE, "b/f2", 100),
            make_size_node(13, DirType.FILE, "c/f3", 100),
            make_size_node(14, DirType.FILE, "d/", 100),
            make_size_node(15, DirType.FILE, "e/f5", 100),
            make_size_node(16, DirType.DIR, "f/f6", 100),
            make_size_node(17, DirType.FILE, "g/f7", 100),
        ]
    )


def make_size_node(node_id, dir_type, path, size):
    return Node(
        node_id,
        dir_type=dir_type,
        path=path,
        size=size,
    )


class FakeCache:
    def contains(self, _):
        return False


def make_chunks(nodes, *args, **kwargs):
    return NodeChunk(FakeCache(), "s3://foo", nodes, *args, **kwargs)


def test_node_bucket_the_only_item():
    bkt = make_chunks(iter([make_size_node(20, DirType.FILE, "file.csv", 100)]), 201)

    result = next(bkt)
    assert len(result) == 1
    assert next(bkt, None) is None


def test_node_bucket_the_only_item_over_limit():
    bkt = make_chunks(iter([make_size_node(20, DirType.FILE, "file.csv", 100)]), 1)

    result = next(bkt)
    assert len(result) == 1
    assert next(bkt, None) is None


def test_node_bucket_the_last_one():
    bkt = make_chunks(iter([make_size_node(20, DirType.FILE, "file.csv", 100)]), 1)

    next(bkt)
    with pytest.raises(StopIteration):
        next(bkt)


def test_node_bucket_basic(nodes):
    bkt = list(make_chunks(nodes, 201))

    assert len(bkt) == 2
    assert len(bkt[0]) == 3
    assert len(bkt[1]) == 1


def test_node_bucket_full_split(nodes):
    bkt = list(make_chunks(nodes, 0))

    assert len(bkt) == 4
    assert len(bkt[0]) == 1
    assert len(bkt[1]) == 1
    assert len(bkt[2]) == 1
    assert len(bkt[3]) == 1


@pytest.mark.parametrize(
    "value",
    [
        "2026-09-30T23:00:00.000Z",
        "2026-09-30T23:00:00Z",
        "Wed, 30 Sep 2026 23:00:00 GMT",
        1790809200,
    ],
)
def test_parse_timestamp_matches_botocore(value):
    assert _parse_timestamp(value) == parse_timestamp(value)


LISTING = (
    '<?xml version="1.0" encoding="UTF-8"?><ListVersionsResult '
    'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>b</Name><Prefix>p/</Prefix>'
    "<KeyMarker></KeyMarker><VersionIdMarker></VersionIdMarker><MaxKeys>2</MaxKeys>"
    "<EncodingType>url</EncodingType><IsTruncated>true</IsTruncated>"
    "<NextKeyMarker>p%2Fb</NextKeyMarker><NextVersionIdMarker>v2</NextVersionIdMarker>"
    "<Version><Key>p%2Fa</Key><VersionId>v1</VersionId><IsLatest>true</IsLatest>"
    "<LastModified>2026-09-30T23:00:00.000Z</LastModified><ETag>&quot;e&quot;</ETag>"
    "<ChecksumAlgorithm>CRC32</ChecksumAlgorithm><Size>3</Size>"
    "<StorageClass>STANDARD</StorageClass><Owner><ID>o</ID></Owner>"
    "<RestoreStatus><IsRestoreInProgress>false</IsRestoreInProgress>"
    "<RestoreExpiryDate>2026-10-01T00:00:00.000Z</RestoreExpiryDate></RestoreStatus>"
    "</Version><DeleteMarker><Key>p%2Fb</Key><VersionId>v2</VersionId>"
    "<IsLatest>false</IsLatest><LastModified>2026-09-30T23:00:00.000Z</LastModified>"
    "</DeleteMarker><CommonPrefixes><Prefix>p%2Fc%2F</Prefix></CommonPrefixes>"
    "</ListVersionsResult>"
)


@pytest.mark.parametrize("operation", ["ListObjectVersions", "ListObjectsV2"])
def test_listing_parser_matches_botocore(operation):
    from aiobotocore.parsers import AioRestXMLParser
    from botocore.session import get_session
    from fsspec.asyn import get_loop, sync

    from datachain.client.s3 import _ListingParser

    body = LISTING.encode()
    if operation == "ListObjectsV2":
        body = (
            body.replace(b"ListVersionsResult", b"ListBucketResult")
            .replace(b"<Version>", b"<Contents>")
            .replace(b"</Version>", b"</Contents>")
        )
    shape = (
        get_session().get_service_model("s3").operation_model(operation).output_shape
    )

    async def parse(parser):
        response = {"status_code": 200, "headers": {}, "body": body}
        parsed = parser.parse(response, shape)
        return await parsed if inspect.isawaitable(parsed) else parsed

    expected = sync(get_loop(), parse, AioRestXMLParser())
    assert sync(get_loop(), parse, _ListingParser()) == expected


@pytest.mark.parametrize(
    "operation,body,status",
    [
        ("ListObjectsV2", b"<Error><Code>InternalError</Code></Error>", 200),
        ("CompleteMultipartUpload", b"<Error><Code>InternalError</Code></Error>", 500),
        ("CompleteMultipartUpload", b"<CompleteMultipartUploadResult/>", 200),
    ],
)
def test_200_error_check(operation, body, status):
    model = botocore.session.get_session().get_service_model("s3")
    response = {"status_code": 200, "body": body, "headers": {}}
    _handle_200_error(model.operation_model(operation), response)
    assert response["status_code"] == status


@pytest.mark.parametrize(
    "bucket, in_order", [("bucket", True), ("bucket--usw2-az1--x-s3", False)]
)
def test_directory_buckets_list_out_of_key_order(bucket, in_order):
    assert ClientS3(bucket, {}, None)._lists_in_key_order() is in_order


def test_r2_lists_out_of_key_order(mocker):
    client = ClientS3("bucket", {}, None)
    endpoint = "https://a.r2.cloudflarestorage.com"
    client._fs = mocker.Mock(_s3=mocker.Mock(meta=mocker.Mock(endpoint_url=endpoint)))
    assert not client._lists_in_key_order()


@pytest.mark.parametrize("versions", [False, True])
@pytest.mark.parametrize("requester_pays", [False, True])
def test_listing_pays_as_requester_when_asked(versions, requester_pays):
    s3 = MagicMock()
    s3.list_objects_v2 = s3.list_object_versions = AsyncMock(return_value={})
    client = ClientS3("bucket", {}, None)
    client._fs = MagicMock(
        version_aware=versions,
        req_kw={"RequestPayer": "requester"} if requester_pays else {},
        set_session=AsyncMock(),
        get_s3=AsyncMock(return_value=s3),
    )

    sync(get_loop(), client._list_page, "", "", None, lambda _: True)

    sent = s3.list_objects_v2.call_args.kwargs
    assert sent.get("RequestPayer") == ("requester" if requester_pays else None)


def test_listing_fields_are_kept_once_per_shape_name():
    def shape():
        model = botocore.session.get_session().get_service_model("s3")
        return model.operation_model("ListObjectVersions").output_shape

    _fields(shape(), "")
    kept = len(_FIELDS)
    _fields(shape(), "")

    assert len(_FIELDS) == kept


PAGE = (
    b'<?xml version="1.0" encoding="UTF-8"?>\n<ListBucketResult '
    b'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>b</Name>'
    b"<KeyCount>1</KeyCount><MaxKeys>1000</MaxKeys><IsTruncated>false</IsTruncated>"
    b"<Contents><Key>a</Key><LastModified>2026-09-30T23:00:00.000Z</LastModified>"
    b"<ETag>&quot;e&quot;</ETag><Size>1</Size></Contents></ListBucketResult>"
)
ERROR_200 = b"<Error><Code>InternalError</Code><Message>retry</Message></Error>"
MALFORMED = PAGE.replace(b"</Key>", b"</Kee>")
TRUNCATED = PAGE[:-40]


@pytest.fixture
def s3_bodies(mocker):
    """An S3 endpoint that answers every list request with the next body."""
    bodies: list[bytes] = []
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            body = bodies.pop(0) if len(bodies) > 1 else bodies[0]
            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    mocker.patch("botocore.retryhandler.random.random", return_value=0.0)  # no backoff
    fs = ClientS3.create_fs(
        key="k",
        secret="s",  # noqa: S106
        client_kwargs={"endpoint_url": f"http://127.0.0.1:{server.server_port}"},
        config_kwargs={"retries": {"mode": "legacy"}},
        skip_instance_cache=True,
    )

    def page(*answers, versions=False, delimiter=""):
        bodies[:] = answers
        requests.clear()
        client = ClientS3("b", {"version_aware": versions}, None)
        client._fs = fs
        fs.version_aware = versions
        return (
            sync(get_loop(), client._list_page, "", "", None, delimiter=delimiter),
            len(requests),
        )

    def listed(*answers, versions=False):
        listed_page, count = page(*answers, versions=versions)
        return [r.path for r in listed_page.rows], count

    listed.page = page
    listed.fs = fs
    listed.bodies = bodies
    yield listed
    server.shutdown()
    server.server_close()


def _as_versions_xml(body: bytes) -> bytes:
    version = b"<Version><VersionId>v1</VersionId><IsLatest>true</IsLatest>"
    return (
        body.replace(b"ListBucketResult", b"ListVersionsResult")
        .replace(b"<Contents>", version)
        .replace(b"</Contents>", b"</Version>")
    )


NBSP_TAG = PAGE.replace(b"</Size>", "</Size><StorageClass\u00a0/>".encode())


@pytest.mark.parametrize("versions", [False, True])
@pytest.mark.parametrize(
    "answers, keys, requests",
    [
        ([PAGE], ["a"], 1),
        ([PAGE + b"<!-- trailing comment -->"], ["a"], 1),
        ([b""], [], 1),
        ([MALFORMED, PAGE], ["a"], 2),
        ([TRUNCATED, PAGE], ["a"], 2),
        ([ERROR_200, PAGE], ["a"], 2),
        ([NBSP_TAG, PAGE], ["a"], 2),
        ([PAGE.replace(b"</Key>", b"\x0b</Key>"), PAGE], ["a"], 2),
        ([PAGE.replace(b"</Key>", "\ufffe".encode() + b"</Key>"), PAGE], ["a"], 2),
        ([PAGE.replace(b"version", b"VERSION"), PAGE], ["a"], 2),
        ([PAGE.replace(b"s3.amazonaws.com/doc/", b"doc.s3.amazonaws.com/")], ["a"], 1),
    ],
    ids=[
        "valid",
        "comment",
        "empty",
        "malformed",
        "truncated",
        "error",
        "nbsp-tag",
        "control",
        "nonchar",
        "upper-decl",
        "gcs-xml",
    ],
)
def test_s3_listing_retries_only_a_bad_200(
    s3_bodies, versions, answers, keys, requests
):
    bodies = [_as_versions_xml(b) if versions else b for b in answers]
    assert s3_bodies(*bodies, versions=versions) == (keys, requests)


@pytest.mark.parametrize(
    "body", [MALFORMED, TRUNCATED, ERROR_200], ids=["malformed", "truncated", "error"]
)
def test_s3_listing_gives_up_on_a_bad_200_at_the_retry_limit(s3_bodies, body):
    with pytest.raises(ClientError) as e:
        s3_bodies(body)
    assert e.value.response["ResponseMetadata"]["HTTPStatusCode"] == 500


def test_s3_filesystem_pickles():
    fs = ClientS3.create_fs(
        anon=True,
        client_kwargs={"region_name": "us-east-1"},
        skip_instance_cache=True,
    )
    fs.connect()

    assert isinstance(pickle.loads(pickle.dumps(fs)), type(fs))  # noqa: S301


def test_s3_sessions_parse_listings_and_take_new_credentials(monkeypatch):
    from datachain.client.s3 import _ListingParser

    def credentials(region):
        fs = ClientS3.create_fs(client_kwargs={"region_name": region})
        fs.connect()
        parser = fs.s3._endpoint._response_parser_factory.create_parser("rest-xml")
        return fs.s3._request_signer._credentials.access_key, parser

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "KEY_A")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    first, parser = credentials("us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "KEY_B")
    second, _ = credentials("us-west-2")

    assert (first, second) == ("KEY_A", "KEY_B")
    assert isinstance(parser, _ListingParser)


@pytest.mark.parametrize("refresh", ["explicit", "closed_http_sessions"])
def test_s3_sessions_keep_the_listing_parser_on_refresh(refresh):
    fs = ClientS3.create_fs(
        anon=True,
        client_kwargs={"region_name": "us-east-1"},
        skip_instance_cache=True,
    )
    fs.connect()
    if refresh == "explicit":
        fs.connect(refresh=True)
    else:
        sync(get_loop(), fs.s3._endpoint.http_session.close)
        fs.connect()

    factory = fs.s3._endpoint._response_parser_factory
    assert type(factory).__name__ == "_ParserFactory"


def test_s3_injected_session_refreshes():
    from aiobotocore.session import AioSession

    session = AioSession()
    fs = ClientS3.create_fs(
        anon=True,
        session=session,
        client_kwargs={"region_name": "us-east-1"},
        skip_instance_cache=True,
    )
    fs.connect()
    fs.connect(refresh=True)

    assert fs.session is session


def test_listing_parser_hooks_only_the_synchronous_parse():
    from datachain.client.s3 import _ListingParser

    # aiobotocore before 2.22 parses synchronously: an async override breaks it
    assert "parse" not in vars(_ListingParser)
    assert not inspect.iscoroutinefunction(_ListingParser._do_parse)


def _listing_client(versions, s3):
    client = ClientS3("bucket", {}, None)
    client._fs = MagicMock(
        version_aware=versions,
        req_kw={},
        set_session=AsyncMock(),
        get_s3=AsyncMock(return_value=s3),
    )
    return client


@pytest.mark.parametrize(
    "versions, token, code",
    [
        (True, None, "AccessDenied"),
        (True, ("k", "v"), "NotImplemented"),
        (False, None, "NotImplemented"),
    ],
    ids=["other-error", "after-first-page", "non-versioned"],
)
def test_list_page_errors(versions, token, code):
    error = ClientError({"Error": {"Code": code}}, "ListObjectVersions")
    s3 = MagicMock(
        list_object_versions=AsyncMock(side_effect=error),
        list_objects_v2=AsyncMock(side_effect=error),
    )
    client = _listing_client(versions, s3)

    with pytest.raises(ClientError, match=code):
        sync(get_loop(), client._list_page, "", "", token)
    assert not s3.list_objects_v2.call_count if versions else True


def test_list_page_falls_back_to_current_objects_without_version_support():
    error = ClientError({"Error": {"Code": "NotImplemented"}}, "ListObjectVersions")
    contents = [
        {"Key": "a", "Size": 1, "ETag": '"e"', "LastModified": parse_timestamp(0)}
    ]
    s3 = MagicMock(
        list_object_versions=AsyncMock(side_effect=error),
        list_objects_v2=AsyncMock(
            return_value={"Contents": contents, "IsTruncated": False}
        ),
    )
    client = _listing_client(True, s3)

    for _ in range(2):  # the second page goes straight to the current objects
        page = sync(get_loop(), client._list_page, "", "", None)
        assert [(r.path, r.version, r.is_latest) for r in page.rows] == [
            ("a", "", True)
        ]
    assert s3.list_object_versions.call_count == 1


WHEN = _parse_timestamp("2026-09-30T23:00:00.000Z")
FOLDER = b"<CommonPrefixes><Prefix>p/z+1%26/</Prefix></CommonPrefixes>"


def _page(body: bytes, op: str = "ListObjectsV2"):
    model = botocore.session.get_session().get_service_model("s3")
    shape = model.operation_model(op).output_shape
    return _rows_page(body, shape, "s3://b")


ENCODED = (
    PAGE.replace(b"<Key>a</Key>", b"<Key>p/a+b%26c</Key>")
    .replace(
        b"<IsTruncated>false</IsTruncated>",
        b"<EncodingType>url</EncodingType><IsTruncated>true</IsTruncated>"
        b"<NextContinuationToken>t+1</NextContinuationToken>",
    )
    .replace(b"</Contents>", b"</Contents>" + FOLDER)
)
VERSIONS = (
    b'<?xml version="1.0" encoding="UTF-8"?>\n<ListVersionsResult '
    b'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>b</Name>'
    b"<MaxKeys>1000</MaxKeys><EncodingType>url</EncodingType>"
    b"<IsTruncated>true</IsTruncated><NextKeyMarker>p/d+1%26</NextKeyMarker>"
    b"<NextVersionIdMarker>v2</NextVersionIdMarker>"
    b"<Version><Key>p/a+b%26c</Key><VersionId>v1</VersionId><IsLatest>true</IsLatest>"
    b"<LastModified>2026-09-30T23:00:00.000Z</LastModified>"
    b"<ETag>&quot;e&quot;</ETag><Size>1</Size></Version>"
    b"<DeleteMarker><Key>p/d+1%26</Key><VersionId>v2</VersionId><IsLatest>true</IsLatest>"
    b"<LastModified>2026-09-30T23:00:00.000Z</LastModified></DeleteMarker>"
    + FOLDER
    + b"</ListVersionsResult>"
)


def test_patterns_read_a_page_as_s3_writes_it(s3_bodies):
    assert _page(ENCODED) is not None
    assert _page(VERSIONS, "ListObjectVersions") is not None

    page, count = s3_bodies.page(ENCODED, versions=False, delimiter="/")
    versions, v_count = s3_bodies.page(VERSIONS, versions=True, delimiter="/")

    assert (count, v_count) == (1, 1)
    assert [tuple(r) for r in page.rows] == [
        ("s3://b", "p/a b&c", 1, "", "e", True, WHEN, None)
    ]
    assert (page.folders, page.last, page.token) == (("p/z 1&/",), "p/z 1&/", "t+1")
    assert [tuple(r) for r in versions.rows] == [
        ("s3://b", "p/a b&c", 1, "v1", "e", True, WHEN, None)
    ]
    assert (versions.folders, versions.last, versions.token) == (
        ("p/z 1&/",),
        "p/z 1&/",
        ("p/d 1&", "v2"),
    )


@pytest.mark.parametrize("versions", [False, True])
@pytest.mark.parametrize(
    "old, new, key",
    [
        (b"<Name>", b"<!-- c --><Name>", "a"),
        (b"<Size>1</Size>", b"<Size>1</Size><Extra/>", "a"),
        (b"</Key>", b"&#13;</Key>", "a\r"),
        (b"</Key>", b"\r</Key>", "a\n"),
        (b'"1.0" encoding="UTF-8"', b"'1.0' encoding='utf-8'", "a"),
    ],
    ids=["comment", "unknown-field", "char-reference", "cr", "single-quoted-decl"],
)
def test_elementtree_reads_a_page_the_patterns_do_not(
    s3_bodies, versions, old, new, key
):
    body = (_as_versions_xml(PAGE) if versions else PAGE).replace(old, new)
    op = "ListObjectVersions" if versions else "ListObjectsV2"

    assert _page(body, op) is None
    assert s3_bodies(body, versions=versions) == ([key], 1)


def test_client_stops_trying_patterns_after_three_declined_pages(
    s3_bodies, mocker, caplog
):
    declined = PAGE.replace(b"<Size>1</Size>", b"<Size>1</Size><Extra/>")
    spy = mocker.spy(s3_mod, "_rows_page")
    client = ClientS3("b", {"version_aware": False}, None)
    client._fs = s3_bodies.fs
    s3_bodies.fs.version_aware = False
    s3_bodies.bodies[:] = [declined, declined, declined, PAGE]

    with caplog.at_level("DEBUG", logger="datachain"):
        pages = [sync(get_loop(), client._list_page, "", "", None) for _ in range(4)]

    assert [[r.path for r in p.rows] for p in pages] == [["a"]] * 4
    assert spy.call_count == 3
    assert "reading pages with ElementTree" in caplog.text


def test_patterns_agree_with_elementtree_or_decline(s3_bodies):
    entry_a = (
        b"<Contents><Key>a&amp;lt;b&amp;c&lt;d&gt;&apos;&#34;&#39;</Key>"
        b"<LastModified>2026-09-30T23:00:00.000Z</LastModified>"
        b"<ETag>&quot;e1&quot;</ETag><ChecksumAlgorithm>CRC32</ChecksumAlgorithm>"
        b"<ChecksumType>FULL_OBJECT</ChecksumType><Size>1</Size>"
        b"<StorageClass>STANDARD</StorageClass>"
        b"<Owner><ID>o</ID><DisplayName>d</DisplayName></Owner></Contents>"
    )
    entry_b = (
        b"<Contents><Key>b</Key>"
        b"<LastModified>2026-09-30T23:00:00.000Z</LastModified>"
        b"<ETag>&quot;e2&quot;</ETag><Size>2</Size>"
        b"<StorageClass \t\n/><Owner><ID/><DisplayName/></Owner></Contents>"
    )
    accepted = [
        PAGE.replace(b"UTF-8", b"utf-8"),
        PAGE.replace(
            b"<Contents><Key>a</Key><LastModified>2026-09-30T23:00:00.000Z</LastModified>"
            b"<ETag>&quot;e&quot;</ETag><Size>1</Size></Contents>",
            b"",
        ),
        PAGE.replace(
            b"<Contents><Key>a</Key><LastModified>2026-09-30T23:00:00.000Z</LastModified>"
            b"<ETag>&quot;e&quot;</ETag><Size>1</Size></Contents>",
            entry_a + entry_b + FOLDER,
        ),
        ENCODED,
        VERSIONS,
    ]
    declined = [
        # gap between entries: must decline so the second entry is not dropped
        PAGE.replace(b"</Contents>", b"</Contents><Extra/>" + entry_b),
        # trailing entry with reordered fields: must decline so it is not dropped
        PAGE.replace(
            b"</Contents>",
            b"</Contents><Contents><Size>2</Size><Key>b</Key>"
            b"<LastModified>2026-09-30T23:00:00.000Z</LastModified>"
            b"<ETag>&quot;e2&quot;</ETag></Contents>",
        ),
        # nested entry inside another tag: must decline so it is not treated as root's
        PAGE.replace(b"<Contents>", b"<Extra><Contents>").replace(
            b"</Contents>", b"</Contents></Extra>"
        ),
        # unread entry before matched entries: must decline so it is not skipped
        PAGE.replace(
            b"<Contents>",
            b"<CommonPrefixes><Extra/><Prefix>z/</Prefix></CommonPrefixes><Contents>",
        ),
    ]
    for body in accepted:
        versions = b"<ListVersionsResult" in body
        op = "ListObjectVersions" if versions else "ListObjectsV2"
        assert _page(body, op) is not None
        fast, _ = s3_bodies.page(body, versions=versions, delimiter="/")
        s3_bodies.bodies[:] = [body]
        client = ClientS3("b", {"version_aware": versions}, None)
        client._fs = s3_bodies.fs
        client._declined_pages = 3
        slow = sync(get_loop(), client._list_page, "", "", None, delimiter="/")
        assert fast == slow

    for body in declined:
        assert _page(body) is None
        fast, _ = s3_bodies.page(body, delimiter="/")
        s3_bodies.bodies[:] = [body]
        client = ClientS3("b", {"version_aware": False}, None)
        client._fs = s3_bodies.fs
        client._declined_pages = 3
        slow = sync(get_loop(), client._list_page, "", "", None, delimiter="/")
        assert fast == slow

    for bad in (
        PAGE + b"\xff",
        PAGE.replace(b"<Name>b</Name>", b"<Name>b</Tag>"),
        PAGE.replace(b"2026-09-30T23:00:00.000Z", b"not-a-date"),
        VERSIONS.replace(b"2026-09-30T23:00:00.000Z", b"not-a-date"),
        VERSIONS.replace(
            b"<LastModified>2026-09-30T23:00:00.000Z</LastModified></DeleteMarker>",
            b"<LastModified></LastModified></DeleteMarker>",
        ),
    ):
        op = "ListObjectVersions" if b"<ListVersionsResult" in bad else "ListObjectsV2"
        assert _page(bad, op) is None
