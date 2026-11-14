import inspect
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import quote_plus

import pytest
from botocore import handlers
from botocore.session import get_session
from fsspec.asyn import get_loop, sync
from hypothesis import given, settings
from hypothesis import strategies as st

from datachain.client import s3
from datachain.client.ranges import Page
from datachain.client.s3 import ClientS3, _ListingParser, _rows_page
from datachain.client.s3_pages import LeanError, LeanS3, LeanSpec

PAGES = Path(__file__).parent / "s3_pages"
NS = 'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"'
DATE = "2026-09-30T23:00:00.000Z"
QUOTE = "&quot;"


_ENTRY = (
    b"<Contents><Key>x</Key><LastModified>2026-09-30T23:00:00.000Z</LastModified>"
    b"<Size>1</Size></Contents>"
)


def _shape(versions: bool):
    operation = "ListObjectVersions" if versions else "ListObjectsV2"
    return get_session().get_service_model("s3").operation_model(operation).output_shape


async def _parsed(body: bytes, versions: bool, source: str | None) -> dict:
    token = s3._ROWS_OF.set(source)
    try:
        response = {"status_code": 200, "headers": {}, "body": body}
        parsed = _ListingParser().parse(response, _shape(versions))
        return await parsed if inspect.isawaitable(parsed) else parsed
    finally:
        s3._ROWS_OF.reset(token)


def _parse(body: bytes, versions: bool, source: str | None) -> dict:
    return sync(get_loop(), _parsed, body, versions, source)


def _list_page(
    body: bytes, versions: bool, fast: bool, start_after="", hi=None, auto_set=True
):
    """``_list_page`` of ``body`` as botocore hands it over: parsed, then
    URL-decoded by botocore's handler and by ``_decode_rows``. ``auto_set``:
    botocore asked for the URL encoding itself, as the request didn't set
    EncodingType. Also whether the patterns read the page."""
    read_fast = [False]

    async def respond(**_):
        source = s3._ROWS_OF.get() if fast else None
        parsed = await _parsed(body, versions, source)
        read_fast[0] = "FileRows" in parsed
        decode = (
            handlers.decode_list_object_versions
            if versions
            else handlers.decode_list_object_v2
        )
        context = {"encoding_type_auto_set": True} if auto_set else {}
        decode(parsed, context)
        s3._decode_rows(parsed, context)
        return parsed

    client = ClientS3("bucket", {}, None)  # type: ignore[arg-type]
    endpoint = MagicMock(list_objects_v2=respond, list_object_versions=respond)
    client._fs = MagicMock(
        version_aware=versions,
        req_kw={},
        set_session=AsyncMock(),
        get_s3=AsyncMock(return_value=endpoint),
    )
    try:
        page = sync(get_loop(), client._list_page, "", start_after, None, hi)
    except Exception as e:  # noqa: BLE001
        return type(e), read_fast[0]
    return page, read_fast[0]


def _lean_page(body: bytes, versions: bool, start_after="", hi=None) -> Page | None:
    """The page as a listing child reads it without botocore; None when it
    leaves the page to botocore."""
    spec = LeanSpec(
        uri="s3://bucket",
        url="http://127.0.0.1/bucket",
        region=None,
        credentials=None,
        headers={},
        versions=versions,
        verify=True,
        in_key_order=True,
    )
    try:
        page = LeanS3(spec)._page(body, start_after, hi)
    except LeanError:
        return None
    return page._replace(rows=[tuple(r) for r in page.rows])


def _assert_lean_reads_as_botocore(body, versions, expected, start_after="", hi=None):
    """A child reads a page as ``_list_page`` does, or leaves it to botocore;
    a page botocore fails, it fails too."""
    lean = _lean_page(body, versions, start_after, hi)
    if lean is not None:
        assert isinstance(expected, Page)
        assert lean == expected._replace(rows=[tuple(r) for r in expected.rows])
    return lean


def _assert_lists_as_elementtree(body: bytes, versions=False, fast=True, **kwargs):
    page, read_fast = _list_page(body, versions, True, **kwargs)
    expected, _ = _list_page(body, versions, False, **kwargs)
    assert page == expected
    assert read_fast is fast
    if kwargs.get("auto_set", True):  # a child asks for URL encoding itself
        hi = kwargs.get("hi")
        start = kwargs.get("start_after", "")
        _assert_lean_reads_as_botocore(body, versions, expected, start, hi)
    return page


def _element(tag: str, value) -> str:
    if value is None:
        return f"<{tag}/>"
    if isinstance(value, str):
        return f"<{tag}>{value}</{tag}>"
    return f"<{tag}>" + "".join(_element(t, v) for t, v in value) + f"</{tag}>"


def _escaped(text: str, quote='"', gt=">") -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", gt)
    return text.replace('"', quote)


def _object(key, size="1", etag=f"{QUOTE}e{QUOTE}", modified=DATE, **more):
    fields = [("Key", key), ("LastModified", modified), ("ETag", etag)]
    return [*fields, ("Size", size), *more.items()]


def _version(key, version="v", latest="true", **more):
    fields = _object(key, **more)
    return [fields[0], ("VersionId", version), ("IsLatest", latest), *fields[1:]]


def _marker(key, version="d", latest="true"):
    return [
        ("Key", key),
        ("VersionId", version),
        ("IsLatest", latest),
        ("LastModified", DATE),
    ]


def _page(
    entries=(),
    *,
    versions=False,
    url=False,
    prefixes=(),
    truncated=False,
    space="",
    declaration='<?xml version="1.0" encoding="UTF-8"?>\n',
) -> bytes:
    """A listing page. ``entries`` are (tag, fields) pairs; ``Contents`` is
    renamed for a version listing."""
    root = "ListVersionsResult" if versions else "ListBucketResult"
    top = [("Name", "bucket"), ("Prefix", ""), ("MaxKeys", "1000")]
    if url:
        top.append(("EncodingType", "url"))
    top.append(("IsTruncated", "true" if truncated else "false"))
    if truncated:
        top += (
            [("NextKeyMarker", "z"), ("NextVersionIdMarker", "zv")]
            if versions
            else [("NextContinuationToken", "t&amp;n")]
        )
    elements = [_element(t, v) for t, v in top]
    for tag, fields in entries:
        tag = "Version" if versions and tag == "Contents" else tag
        inner = space.join(_element(t, v) for t, v in fields)
        elements.append(f"<{tag}>{space}{inner}{space}</{tag}>")
    elements += [_element("CommonPrefixes", [("Prefix", p)]) for p in prefixes]
    inner = space.join(elements)
    return f"{declaration}<{root} {NS}>{space}{inner}{space}</{root}>".encode()


@pytest.mark.parametrize(
    "name, versions",
    [
        ("truncated", False),
        ("folders", False),
        ("empty", False),
        ("owner", False),
        ("versions", True),
    ],
)
def test_real_pages_list_as_elementtree_reads_them(name, versions):
    body = (PAGES / f"{name}.xml").read_bytes()
    _assert_lists_as_elementtree(body, versions)


@pytest.mark.parametrize(
    "name,versions",
    [("truncated", False), ("versions", True), ("owner", False), ("folders", False)],
)
def test_children_read_real_pages_without_botocore(name, versions):
    body = (PAGES / f"{name}.xml").read_bytes()
    expected, _ = _list_page(body, versions, False)

    assert _assert_lean_reads_as_botocore(body, versions, expected) is not None


def test_real_page_rows_hold_every_field():
    body = (PAGES / "versions.xml").read_bytes()
    page = _assert_lists_as_elementtree(body, versions=True)
    row = page.rows[0]
    assert row.path.endswith("_c20220010003587.nc")
    assert (row.size, row.version, row.etag, row.is_latest) == (
        2781690,
        "",
        "a30b2e4d13206f3c4da4a5d86ba780b9",
        True,
    )
    assert row.last_modified.isoformat() == "2022-01-01T00:04:15+00:00"
    assert page.token == (page.rows[-1].path, "null")


@pytest.mark.parametrize(
    "key, path",
    [
        ("a%2Bb+c", "a+b c"),
        ("%25", "%"),
        ("%2525", "%25"),
        ("%C3%A9t%C3%A9", "été"),
        ("a%26b%3Cc%3E%22%27", "a&b<c>\"'"),
        ("plain/key.txt", "plain/key.txt"),
        ("100%", "100%"),
    ],
)
def test_url_encoded_keys_are_decoded_once(key, path):
    body = _page([("Contents", _object(key))], url=True)
    page = _assert_lists_as_elementtree(body)
    assert [r.path for r in page.rows] == [path]


@pytest.mark.parametrize(
    "key, path, fast",
    [
        ("a>b", "a>b", True),
        ("say &quot;hi&quot;", 'say "hi"', True),
        ("say &#34;hi&#34;", 'say "hi"', True),
        ("it&apos;s &#39;x&#39;", "it's 'x'", True),
        ("raw \"quotes\" 'too'", "raw \"quotes\" 'too'", True),
        ("a+b %41", "a+b %41", True),
        ("spaces  and\ttab\nline", "spaces  and\ttab\nline", True),
        ("ünï cødé/日本/😀", "ünï cødé/日本/😀", True),
        ("a&amp;b", "a&b", True),
        ("&lt;&gt;", "<>", True),
        ("&amp;quot;", "&quot;", True),
        ("&#233;", "é", False),
    ],
)
def test_keys_without_url_encoding_read_as_xml_has_them(key, path, fast):
    page = _assert_lists_as_elementtree(_page([("Contents", _object(key))]), fast=fast)
    assert [r.path for r in page.rows] == [path]


@pytest.mark.parametrize("versions", [False, True])
def test_empty_page(versions):
    page = _assert_lists_as_elementtree(_page(versions=versions), versions)
    assert page.rows == []
    assert (page.first, page.last, page.token) == (None, None, None)


@pytest.mark.parametrize("versions", [False, True])
def test_truncated_page_resumes_at_its_token(versions):
    entry = _version("a") if versions else _object("a")
    body = _page([("Contents", entry)], versions=versions, truncated=True)
    page = _assert_lists_as_elementtree(body, versions)
    assert page.token == (("z", "zv") if versions else "t&n")


def test_versions_interleave_with_delete_markers():
    entries = [
        ("Contents", _version("a", "v2", "true")),
        ("Contents", _version("a", "v1", "false")),
        ("DeleteMarker", _marker("b")),
        ("Contents", _version("b", "null", "false")),
        ("DeleteMarker", _marker("c", "null", "true")),
    ]
    page = _assert_lists_as_elementtree(_page(entries, versions=True), versions=True)
    assert [(r.path, r.version, r.is_latest) for r in page.rows] == [
        ("a", "v2", True),
        ("a", "v1", False),
        ("b", "", False),
    ]
    assert page.last == "c"


def test_delete_markers_only():
    entries = [("DeleteMarker", _marker("a")), ("DeleteMarker", _marker("b"))]
    page = _assert_lists_as_elementtree(_page(entries, versions=True), versions=True)
    assert (page.rows, page.first, page.last) == ([], None, "b")


def test_version_fields_left_out():
    fields = [f for f in _version("a") if f[0] not in ("VersionId", "IsLatest", "ETag")]
    page = _assert_lists_as_elementtree(
        _page([("Contents", fields)], versions=True), versions=True, fast=False
    )
    assert [(r.version, r.is_latest, r.etag) for r in page.rows] == [("", True, "")]


@pytest.mark.parametrize(
    "latest, fast",
    [("false", True), ("", False), ("TRUE", False), ("1", False), (" true", False)],
)
def test_is_latest_is_true_only_when_written_true(latest, fast):
    body = _page([("Contents", _version("a", latest=latest))], versions=True)
    page = _assert_lists_as_elementtree(body, versions=True, fast=fast)
    assert page.rows[0].is_latest is False


@pytest.mark.parametrize(
    "prefixes, url, fast",
    [
        (["a%2Fb%2B%2F/", "c+d/"], True, True),
        (["a&quot;b/", "c d/"], False, True),
        (["a&amp;b/", "c d/"], False, True),
    ],
)
def test_folders_only(prefixes, url, fast):
    page = _assert_lists_as_elementtree(_page(prefixes=prefixes, url=url), fast=fast)
    assert (page.rows, page.last) == ([], "c d/")


@pytest.mark.parametrize(
    "more",
    [
        {"ChecksumAlgorithm": "CRC64NVME", "ChecksumType": "FULL_OBJECT"},
        {"Owner": [("ID", "o"), ("DisplayName", "n")]},
        {"Owner": [("DisplayName", "n"), ("ID", "o")]},
        {"Owner": []},
        {"Owner": None},
        {"Owner": [("ID", "o"), ("DisplayName", None)]},
        {"StorageClass": None, "ChecksumType": None},
        {"StorageClass": "A", "ChecksumType": "B"} | {"Owner": [("ID", "o")] * 2},
    ],
    ids=[
        "checksum",
        "owner",
        "owner-reordered",
        "owner-empty",
        "owner-self-closing",
        "display-name-self-closing",
        "self-closing",
        "repeated-owner-id",
    ],
)
def test_optional_fields(more):
    _assert_lists_as_elementtree(_page([("Contents", _object("a", **more))]))


def test_self_closing_top_level_field():
    body = _page([("Contents", _object("a"))]).replace(
        b"<MaxKeys>", b"<Delimiter/><StartAfter/><MaxKeys>"
    )
    _assert_lists_as_elementtree(body)


@pytest.mark.parametrize(
    "more",
    [
        {"RestoreStatus": [("IsRestoreInProgress", "true")]},
        {"Owner": [("ID", [("Deeper", "x")])]},
        {"Unknown": "x"},
        {"Owner": [("ID", "o"), ("Other", "x")]},
        {"StorageClass": "&amp;"},
    ],
    ids=["restore-status", "nested-owner", "unknown", "unknown-in-owner", "entity"],
)
def test_fields_not_read_may_hold_anything(more):
    entries = [("Contents", _object(k)) for k in "0z"]
    entries.insert(1, ("Contents", _object("a", **more)))
    page = _assert_lists_as_elementtree(_page(entries), fast=False)
    assert [r.path for r in page.rows] == ["0", "a", "z"]


def test_repeated_checksum_algorithms():
    fields = [
        *_object("a"),
        ("ChecksumAlgorithm", "CRC32"),
        ("ChecksumAlgorithm", "SHA1"),
    ]
    _assert_lists_as_elementtree(_page([("Contents", fields)]))


def test_fields_in_another_order_go_to_elementtree():
    fields = _object("a", StorageClass="STANDARD", ChecksumAlgorithm="CRC32")
    entries = [("Contents", fields[::-1]), ("Contents", _object("b"))]
    page = _assert_lists_as_elementtree(_page(entries), fast=False)
    assert [r.path for r in page.rows] == ["a", "b"]


@pytest.mark.parametrize(
    "first",
    [
        ("Size", "5"),
        ("Size", "corrupt"),
        ("Key", "&undefined;"),
        ("Key", "&#0;"),
        ("LastModified", "never"),
        ("ETag", "&#x22;e&#x22;"),
    ],
)
def test_a_read_field_written_twice_goes_to_elementtree(first):
    body = _page([("Contents", [first, *_object("a")])])
    assert _rows_page(body, _shape(False), "s3://bucket") is None
    _assert_lists_as_elementtree(body, fast=False)


def test_an_unread_field_written_twice():
    fields = [*_object("a"), ("StorageClass", "A"), ("StorageClass", "B")]
    _assert_lists_as_elementtree(_page([("Contents", fields)]))


@pytest.mark.parametrize(
    "etag, fast",
    [
        (None, False),
        ("&#x22;e&#x22;", False),
        ("&#X22;e", False),
        ("&amp;quot;e", True),
    ],
)
def test_etags(etag, fast):
    _assert_lists_as_elementtree(
        _page([("Contents", _object("a", etag=etag))]), fast=fast
    )


def test_delete_marker_dates_are_checked():
    marker = [*_marker("b")[:3], ("LastModified", "never")]
    body = _page([("Contents", _version("a")), ("DeleteMarker", marker)], versions=True)
    page, read_fast = _list_page(body, True, True)
    assert (page, read_fast) == (_list_page(body, True, False)[0], False)


@pytest.mark.parametrize("versions", [False, True])
def test_keys_stay_encoded_when_the_request_asked_for_url_encoding(versions):
    entry = _version("a%2Bb+c") if versions else _object("a%2Bb+c")
    entries = [("Contents", entry), ("DeleteMarker", _marker("d%25"))]
    body = _page(
        entries[: 2 if versions else 1],
        versions=versions,
        url=True,
        prefixes=["e%2B/"],
        truncated=True,
    )
    page = _assert_lists_as_elementtree(body, versions, auto_set=False)
    assert [r.path for r in page.rows] == ["a%2Bb+c"]
    assert page.last == "e%2B/"


def test_pretty_printed_page_goes_to_elementtree():
    entries = [
        ("Contents", _object("a", Owner=[("ID", "o")])),
        ("Contents", _object("b")),
    ]
    body = _page(entries, prefixes=["c/"], space="\n  \t")
    _assert_lists_as_elementtree(body, fast=False)


@pytest.mark.parametrize(
    "declaration", ["", '<?xml version="1.0" encoding="utf-8"?>\n']
)
def test_page_without_declaration_or_in_lower_case(declaration):
    body = _page([("Contents", _object("café"))], declaration=declaration)
    _assert_lists_as_elementtree(body)


@pytest.mark.parametrize(
    "body",
    [
        _page([("Contents", _object("a"))], space="\r\n"),
        _page([("Contents", _object("a"))]).replace(b"<Key>", b"<!-- c --><Key>"),
        _page([("Contents", _object("a"))]).replace(b"</Name>", b"</Name><!-- c -->"),
        _page([("Contents", _object("a"))]).replace(
            b"<Name>",
            b"<!-- <Contents><Key>b</Key><Size>1</Size>"
            b"<LastModified>2026-09-30T23:00:00.000Z</LastModified></Contents> -->"
            b"<Name>",
        ),
        _page([("Contents", _object("<![CDATA[a]]>"))]),
        _page([("Contents", _object("a"))]).replace(b"<Contents>", b"<Contents >"),
        _page([("Contents", _object("a"))]).replace(b"<Name>", b"<?pi x?><Name>"),
        _page(
            [("Contents", _object("a"))],
            declaration='<?xml version="1.0" encoding="ISO-8859-1"?>',
        ),
        _page([("Contents", _object("a", modified="&#50;026-09-30T23:00:00Z"))]),
        _page([("Contents", _object("a", size="&#49;"))]),
    ],
    ids=[
        "crlf",
        "comment-in-entry",
        "comment-at-top",
        "commented-out-entry",
        "cdata",
        "space-in-tag",
        "instruction",
        "latin-1",
        "entity-in-date",
        "entity-in-size",
    ],
)
def test_pages_elementtree_reads(body):
    _assert_lists_as_elementtree(body, fast=False)


@pytest.mark.parametrize(
    "body",
    [
        _page([("Contents", _object("a\x01"))]),
        _page([("Contents", _object("a\x0b"))]),
        _page([("Contents", _object("a￾"))]),
        _page([("Contents", _object("a]]>"))]),
        _page([("Contents", _object("a&foo;"))]),
        _page([("Contents", _object("a&#0;"))]),
        _page([("Contents", _object("a"))])[:-30],
        _page([("Contents", _object("a"))])[: -len(b"</ListBucketResult>")],
        _page([("Contents", _object("a"))]).replace(b"</Key>", b"</Kee>"),
        _page([("Contents", _object("a"))]).replace(b"<Name>", b"<Na<Contents>"),
        _page([("Contents", _object("a"))]).replace(
            b"<Name>", b"<Name " + _ENTRY + b" >"
        ),
        b"\xff" + _page([("Contents", _object("a"))]),
        b"<Error><Code>InternalError</Code></Error>",
    ],
    ids=[
        "control",
        "vertical-tab",
        "noncharacter",
        "cdata-end",
        "undefined-entity",
        "null-reference",
        "cut-mid-entry",
        "cut-after-entries",
        "mismatched-tag",
        "entry-inside-a-tag",
        "entry-inside-tag-space",
        "not-utf-8",
        "error",
    ],
)
def test_malformed_pages_are_retried_as_elementtree_finds_them(body):
    assert _rows_page(body, _shape(False), "s3://bucket") is None
    parsed = _parse(body, False, "s3://bucket")
    assert parsed["ResponseMetadata"]["HTTPStatusCode"] == 500


@pytest.mark.parametrize("missing", ["Key", "Size", "LastModified"])
def test_entries_without_a_required_field_fail_as_elementtree_reads_them(missing):
    fields = [f for f in _object("a") if f[0] != missing]
    page, read_fast = _list_page(_page([("Contents", fields)]), False, True)
    assert (page, read_fast) == (KeyError, False)


def test_is_latest_with_an_entity():
    body = _page([("Contents", _version("a", latest="&#116;rue"))], versions=True)
    page = _assert_lists_as_elementtree(body, versions=True, fast=False)
    assert page.rows[0].is_latest is True


def test_date_with_quote_entities():
    modified = "&#39;2026-09-30T23:00:00.000Z&#39;"
    page = _assert_lists_as_elementtree(
        _page([("Contents", _object("a", modified=modified))])
    )
    assert page.rows[0].last_modified.isoformat() == "2026-09-30T23:00:00+00:00"


@pytest.mark.parametrize(
    "name",
    [
        b"<Name>b" + _ENTRY + b"</Name>",
        b"<Name>b " + _ENTRY + b"</Name>",
        b"<Name>" + _ENTRY + b"b</Name>",
        b"<Name>" + _ENTRY + b"</Name>",
        b"<CommonPrefixes>" + _ENTRY + b"<Prefix>p/</Prefix></CommonPrefixes>",
        b"<Owner><ID>o</ID>" + _ENTRY + b"</Owner>",
    ],
    ids=["after-text", "after-space", "before-text", "alone", "in-folder", "in-owner"],
)
def test_an_entry_inside_another_element_is_no_row(name):
    body = _page([("Contents", _object("a"))]).replace(b"<Name>bucket</Name>", name)
    page = _assert_lists_as_elementtree(body, fast=False)
    assert [r.path for r in page.rows] == ["a"]


@pytest.mark.parametrize(
    "entry",
    [b"<Contents/>", b"<Contents>x</Contents>", b"<CommonPrefixes/>"],
)
def test_an_entry_with_text_only_goes_to_elementtree(entry):
    body = _page([("Contents", _object("a"))]).replace(b"<Name>", entry + b"<Name>")
    _assert_lists_as_elementtree(body, fast=False)


def test_a_prefixed_root_goes_to_elementtree():
    body = _page([("Contents", _object("a"))]).replace(
        b"<ListBucketResult xmlns=", b"<s3:ListBucketResult xmlns:s3="
    )
    body = body.replace(b"</ListBucketResult>", b"</s3:ListBucketResult>")
    page = _assert_lists_as_elementtree(body, fast=False)
    assert page.rows == []


def test_entries_past_hi_need_no_size():
    entries = [("Contents", _object("a")), ("Contents", _object("b")[:2])]
    page = _assert_lists_as_elementtree(_page(entries), fast=False, hi="a")
    assert ([r.path for r in page.rows], page.past_hi) == (["a"], "b")


def test_listing_parser_reads_rows_only_for_list_page():
    body = _page([("Contents", _object("a"))])
    assert "FileRows" not in _parse(body, False, None)
    assert "Contents" not in _parse(body, False, "s3://bucket")


_KEY_CHARS = st.characters(
    exclude_categories=["Cs"],
    exclude_characters="\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f"
    "\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\r\ufffe\uffff",
)
_TEXT = st.text(_KEY_CHARS, max_size=8)
_ENTITIES = re.compile("&(?:amp|lt|gt|quot|#34|apos|#39);")
_QUOTES = st.sampled_from(['"', "&quot;", "&#34;", "&#x22;"])
_OPTIONAL = {
    "StorageClass": st.sampled_from(["STANDARD", "GLACIER", None]),
    "ChecksumAlgorithm": st.sampled_from(["CRC32", "CRC64NVME"]),
    "ChecksumType": st.sampled_from(["FULL_OBJECT", None]),
    "Owner": st.none()
    | st.lists(
        st.tuples(
            st.sampled_from(["ID", "DisplayName"]), st.sampled_from(["o", "n", None])
        ),
        max_size=2,
    ),
}
_SECOND_READS = st.sampled_from(
    [("Size", "7"), ("Size", "x"), ("Key", "&bad;"), ("LastModified", "never")]
)


def _in_s3_order(fields, versions) -> bool:
    """Whether fields are in the order S3 writes them: the read ones first,
    checksums before the size, the rest after it."""
    names = [n for n, _ in fields]
    reads = ["Key", "VersionId", "IsLatest"][: 3 if versions else 1]
    reads += ["LastModified", "ETag"]
    if names[: len(reads)] != reads or "Size" not in names:
        return False
    size = names.index("Size")
    between = names[len(reads) : size]
    return all(n.startswith("Checksum") for n in between) and not (
        {"Key", "VersionId", "IsLatest", "LastModified", "ETag", "Size"}
        & set(names[size + 1 :])
    )


@st.composite
def _pages(draw):
    versions, url = draw(st.booleans()), draw(st.booleans())
    quote, gt = draw(_QUOTES), draw(st.sampled_from([">", "&gt;"]))
    plain = True  # nothing the patterns leave to ElementTree

    def note(text):
        nonlocal plain
        plain = plain and "&" not in _ENTITIES.sub("", text) and "]]>" not in text
        return text

    def written(text):
        return note(quote_plus(text, safe="/") if url else _escaped(text, quote, gt))

    entries = []
    for key in draw(st.lists(_TEXT, max_size=6)):
        if versions and draw(st.booleans()):
            entries.append(("DeleteMarker", _marker(written(key))))
            continue
        more = {k: draw(v) for k, v in _OPTIONAL.items() if draw(st.booleans())}
        etag = note(quote + draw(st.text("0123456789abcdef", max_size=4)) + quote)
        size = str(draw(st.integers(0, 2**63)))
        if versions:
            version = draw(st.sampled_from(["null", "v1", "x&quot;y"]))
            latest = draw(st.sampled_from(["true", "false"]))
            fields = _version(
                written(key), version, latest, etag=etag, size=size, **more
            )
        else:
            fields = _object(written(key), etag=etag, size=size, **more)
        if draw(st.integers(0, 9)) == 0:
            fields = [draw(_SECOND_READS), *fields]
            plain = False
        if draw(st.booleans()):
            shuffled = draw(st.permutations(fields))
            plain = plain and _in_s3_order(shuffled, versions)
            fields = shuffled
        entries.append(("Contents", fields))
    prefixes = [written(p) + "/" for p in draw(st.lists(_TEXT, max_size=3))]
    space = draw(st.sampled_from(["", "\n", "\n  "]))
    body = _page(
        entries,
        versions=versions,
        url=url,
        prefixes=prefixes,
        truncated=draw(st.booleans()),
        space=space,
    )
    # space inside or between entries; the root's other children may have it
    plain = plain and not (space and (entries or len(prefixes) > 1))
    return body, versions, plain, draw(_TEXT), draw(st.none() | _TEXT)


@settings(max_examples=300, deadline=None)
@given(_pages(), st.booleans())
def test_generated_pages_list_as_elementtree_reads_them(page, auto_set):
    body, versions, plain, start_after, hi = page
    fast, read_fast = _list_page(body, versions, True, start_after, hi, auto_set)
    expected, _ = _list_page(body, versions, False, start_after, hi, auto_set)
    assert fast == expected
    assert read_fast is plain
    if auto_set:
        _assert_lean_reads_as_botocore(body, versions, expected, start_after, hi)


@pytest.fixture
def s3_endpoint():
    """An S3 endpoint that answers every request with ``body``."""
    served = {"body": b"", "paths": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            served["paths"].append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(served["body"])))
            self.end_headers()
            self.wfile.write(served["body"])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    served["fs"] = ClientS3.create_fs(
        key="k",
        secret="s",  # noqa: S106
        client_kwargs={"endpoint_url": f"http://127.0.0.1:{server.server_port}"},
        cache_regions=False,
        skip_instance_cache=True,
    )
    yield served
    server.shutdown()
    server.server_close()


def test_list_page_asks_for_url_encoding_and_decodes_each_key_once(s3_endpoint):
    s3_endpoint["body"] = _page(
        [("Contents", _object("a%2Bb+c%25")), ("DeleteMarker", _marker("d%25"))],
        versions=True,
        url=True,
        prefixes=["e%2B/"],
    )
    client = ClientS3("bucket", {}, None)
    client._fs = s3_endpoint["fs"]

    page = sync(get_loop(), client._list_page, "", "", None)

    assert "encoding-type=url" in s3_endpoint["paths"][-1]
    assert [r.path for r in page.rows] == ["a+b c%"]
    assert page.last == "e+/"
    assert s3._ROWS_OF.get() is None


def test_other_listings_read_full_entries(s3_endpoint):
    s3_endpoint["body"] = _page([("Contents", _object("a", StorageClass="GLACIER"))])
    fs = s3_endpoint["fs"]

    async def listed():
        await fs.set_session()
        endpoint = await fs.get_s3()
        return await endpoint.list_objects_v2(Bucket="bucket")

    page = sync(get_loop(), listed)

    assert page["Contents"][0]["StorageClass"] == "GLACIER"
    assert "FileRows" not in page


@pytest.mark.parametrize("versions", [False, True])
@pytest.mark.parametrize(
    "tail",
    [
        b"<StorageClass>STANDARD</Wrong>",
        b"<Owner><ID>x</Owner>",
        b'<StorageClass attr="unterminated>STANDARD</StorageClass>',
        b"<ChecksumAlgorithm>CRC32</ChecksumType>",
    ],
)
def test_malformed_tail_falls_back_in_parent_and_child(versions, tail):
    tag, fields = ("Version", _version("a")) if versions else ("Contents", _object("a"))
    body = _page([(tag, fields)], versions=versions)
    closing = f"</{tag}>".encode()
    body = body.replace(closing, tail + closing)
    assert _rows_page(body, _shape(versions), "s3://bucket") is None
    assert (
        _parse(body, versions, "s3://bucket")["ResponseMetadata"]["HTTPStatusCode"]
        == 500
    )
    assert _lean_page(body, versions) is None
