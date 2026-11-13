import asyncio
import datetime as dt
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple
from unittest import mock

import pytest
from botocore.auth import S3SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from fsspec.asyn import get_loop, sync
from hypothesis import given
from hypothesis import strategies as st

import datachain
from datachain.client import s3_pages
from datachain.client.fsspec import RangeListedClient
from datachain.client.listing_worker import Rows
from datachain.client.ranges import Keep, Page, code_points, utf16_units
from datachain.client.s3 import ClientS3
from datachain.client.s3_pages import LeanS3, LeanSpec, _quote

ROOT = os.path.dirname(os.path.dirname(datachain.__file__))
PAGES = Path(__file__).parent / "s3_pages"


def _spec(**changes) -> LeanSpec:
    spec = LeanSpec(
        uri="s3://bucket",
        url="https://bucket.s3.us-east-1.amazonaws.com/",
        region="us-east-1",
        credentials=("AKIDEXAMPLE", "secret/key+x", "token=="),
        headers={"x-amz-request-payer": "requester"},
        versions=False,
        verify=True,
        in_key_order=True,
    )
    return spec._replace(**changes)


@pytest.mark.parametrize(
    "url", ["https://bucket.s3.us-east-1.amazonaws.com/", "http://127.0.0.1:9000/bkt"]
)
@pytest.mark.parametrize("token", ["token==", None])
def test_requests_are_signed_as_botocore_signs_them(url, token, monkeypatch):
    now = dt.datetime(2026, 11, 12, 1, 2, 3, tzinfo=dt.UTC)
    monkeypatch.setattr(s3_pages.time, "gmtime", now.timetuple)
    monkeypatch.setattr("botocore.auth.get_current_datetime", lambda: now)
    key, secret = "AKIDEXAMPLE", "secret/key+x"
    lean = LeanS3(_spec(url=url, credentials=(key, secret, token)))
    query = {
        "list-type": "2",
        "prefix": "a b/ü+%~*",
        "max-keys": "1000",
        "encoding-type": "url",
        "continuation-token": "1mbNf1+/=",
    }
    canonical = "&".join(f"{_quote(k)}={_quote(v)}" for k, v in sorted(query.items()))

    headers = lean._signed(canonical)

    request = AWSRequest(
        method="GET",
        url=f"{url}?{canonical}",
        headers={"x-amz-request-payer": "requester"},
    )
    S3SigV4Auth(Credentials(key, secret, token), "s3", "us-east-1").add_auth(request)
    assert headers["authorization"] == request.headers["Authorization"]
    assert headers["x-amz-date"] == request.headers["X-Amz-Date"]
    assert headers.get("x-amz-security-token") == request.headers.get(
        "X-Amz-Security-Token"
    )


def test_unsigned_requests_carry_no_signature():
    headers = LeanS3(_spec(region=None, credentials=None))._signed("prefix=")

    assert headers == {"x-amz-request-payer": "requester"}


class _Request(NamedTuple):
    url: str
    method: str
    headers: dict


def _client_after(request: _Request, signer=None, config=None):
    """A client whose botocore just sent ``request``."""
    client = ClientS3("bucket", {}, None)  # type: ignore[arg-type]
    client._fs = mock.MagicMock(config_kwargs=config or {}, client_kwargs={})
    model = mock.MagicMock()
    s3 = mock.MagicMock(_request_signer=mock.MagicMock(_credentials=signer))
    s3.meta.service_model.operation_model.return_value.output_shape = model
    return sync(get_loop(), client._lean_spec, s3, [request], False)


SIGNED = (
    "AWS4-HMAC-SHA256 Credential=AKID/20261112/eu-west-2/s3/aws4_request, "
    "SignedHeaders=host;x-amz-date, Signature=abc"
)


class _Frozen(NamedTuple):
    access_key: str
    secret_key: str
    token: str | None


class _Signer:
    def __init__(self, expiry=None):
        self._expiry_time = expiry

    async def get_frozen_credentials(self):
        return _Frozen("AKID", "secret", "tok")


def test_a_lean_spec_takes_the_url_region_and_credentials_botocore_used():
    sent = _Request(
        "https://bucket.s3.eu-west-2.amazonaws.com/?list-type=2&prefix=a",
        "GET",
        {
            "Authorization": SIGNED,
            "X-Amz-Date": "x",
            "x-amz-request-payer": "requester",
        },
    )

    spec = _client_after(sent, _Signer())

    assert spec.url == "https://bucket.s3.eu-west-2.amazonaws.com/"
    assert spec.region == "eu-west-2"
    assert spec.credentials == ("AKID", "secret", "tok")
    assert spec.headers == {"x-amz-request-payer": "requester"}


def test_an_unsigned_request_makes_an_unsigned_spec():
    spec = _client_after(_Request("http://127.0.0.1:9000/bkt?list-type=2", "GET", {}))

    assert (spec.url, spec.region, spec.credentials) == (
        "http://127.0.0.1:9000/bkt",
        None,
        None,
    )


@pytest.mark.parametrize(
    "request_,signer,config",
    [
        # a directory bucket's session token: not a plain request
        (_Request("https://b/", "GET", {"x-amz-s3session-token": "t"}), None, None),
        # SigV4A, as for multi-region access points
        (
            _Request(
                "https://b/", "GET", {"Authorization": "AWS4-ECDSA-P256-SHA256 x"}
            ),
            None,
            None,
        ),
        (_Request("https://b/", "GET", {"Authorization": SIGNED}), None, None),
        # credentials that expire within the listing
        (
            _Request("https://b/", "GET", {"Authorization": SIGNED}),
            _Signer(dt.datetime.now(dt.UTC)),
            None,
        ),
        (_Request("https://b/", "GET", {}), None, {"proxies": {"https": "p"}}),
        (_Request("ftp://b/", "GET", {}), None, None),
    ],
)
def test_children_use_botocore_unless_the_request_was_plain(request_, signer, config):
    assert _client_after(request_, signer, config) is False


def test_lean_children_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("DATACHAIN_LEAN_LISTING", "0")

    assert _client_after(_Request("https://b/", "GET", {})) is False


class _Key(NamedTuple):
    path: str


_KEYS = st.lists(st.text("ab/é\U0001f600", max_size=4), max_size=12)


@given(
    _KEYS,
    st.booleans(),
    st.text("ab/", max_size=3),
    st.none() | st.text("ab/", max_size=3),
)
def test_a_page_is_kept_as_its_keys_are_one_by_one(keys, ordered, lo, hi):
    keys = sorted(keys) if ordered else keys
    rows = [_Key(k) for k in keys]
    for order in (code_points, utf16_units):
        client = mock.Mock(
            key_order=order, _is_valid_key=RangeListedClient._is_valid_key
        )
        one, whole = Keep(client, lo, hi), Keep(client, lo, hi)

        expected = [r for r in rows if one(r.path)]

        assert whole.take(rows, keys) == expected
        assert (whole.past_lo, whole.past_hi, whole.in_order) == (
            one.past_lo,
            one.past_hi,
            one.in_order,
        )


def test_rows_from_a_child_unpickle_where_they_are_read():
    rows = Rows(pickle.dumps([("s3://b", "k", 1, "", "e", True, None, None)]), 1)

    assert len(rows) == 1
    assert rows._rows is None  # counted, not read
    assert [r.path for r in rows] == ["k"]
    assert rows[0].size == 1


class _FolderClient(RangeListedClient):
    """Keys in folders; a delimiter request names the folders, as S3 does."""

    PREFIX = "fake://"
    protocol = "fake"
    LIST_PAGE_SIZE = 4
    LISTS_LEVELS = True
    NAMES_FOLDERS = True

    def __init__(self, keys: list[str], names_folders: bool = True) -> None:
        super().__init__("bucket", {}, None)  # type: ignore[arg-type]
        self.keys = sorted(keys)
        self.names_folders = names_folders
        self.starts: list[str] = []
        self.requests = 0

    def info_to_file(self, v, path):
        raise NotImplementedError

    async def _list_page(
        self, prefix, start_after, token, hi=None, page_size=None, delimiter=""
    ) -> Page:
        self.requests += 1
        await asyncio.sleep(0)
        keys = [k for k in self.keys if k.startswith(prefix)]
        if delimiter:
            folders = sorted(
                {
                    k[: k.index("/", len(prefix)) + 1]
                    for k in keys
                    if "/" in k[len(prefix) :]
                }
            )
            files = [k for k in keys if "/" not in k[len(prefix) :]]
            keep = Keep(self, "", None)
            return Page(
                [_Key(k) for k in files if keep(k)],
                files[0] if files else None,
                max(files + folders, default=None),
                None,
                folders=tuple(folders) if self.names_folders else None,
            )
        keep = Keep(self, start_after, hi)
        if token is None:
            self.starts.append(start_after)
            token = next((i for i, k in enumerate(keys) if k > start_after), len(keys))
        chunk = keys[token : token + (page_size or self.LIST_PAGE_SIZE)]
        end = token + len(chunk)
        return Page(
            [_Key(k) for k in chunk if keep(k)],
            chunk[0] if chunk else None,
            chunk[-1] if chunk else None,
            end if end < len(keys) else None,
            keep.past_lo,
            keep.past_hi,
        )


def _paths(client) -> list[str]:
    async def paths():
        return [r.path async for page in client.scandir("") for r in page]

    with mock.patch("datachain.client.listing_worker.LIST_PROCESSES", 0):
        return sync(get_loop(), paths)


def test_ranges_start_at_the_folders_past_the_first_page():
    keys = [f"{d}/{i:02}" for d in "abcdef" for i in range(10)] + ["top"]
    client = _FolderClient(keys)

    assert sorted(_paths(client)) == sorted(keys)
    assert {"b/", "c/", "d/", "e/", "f/"} <= set(client.starts)
    assert "a/" not in client.starts  # read already by the first page


def test_a_listing_of_one_page_takes_one_request():
    client = _FolderClient(["a/1", "b/2", "z"])

    assert sorted(_paths(client)) == ["a/1", "b/2", "z"]
    assert client.requests == 1


def test_an_empty_prefix_is_not_found():
    client = _FolderClient(["a/1"])

    async def listed():
        return [r async for page in client.scandir("b") for r in page]

    with pytest.raises(FileNotFoundError):
        sync(get_loop(), listed)


def test_a_client_that_names_no_folders_lists_from_one_range():
    keys = [f"{d}/{i:02}" for d in "abc" for i in range(10)]
    client = _FolderClient(keys, names_folders=False)

    assert sorted(_paths(client)) == keys
    assert client.starts[0] == ""


def test_a_storage_client_imports_only_its_own_library():
    code = (
        "import sys; from datachain.client import Client; "
        "Client.get_implementation('s3://b/k'); "
        "print(sorted({'gcsfs', 'adlfs', 'huggingface_hub'} & set(sys.modules)))"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": ROOT},
    )

    assert out.stdout.strip() == "[]"


def test_a_listing_child_starts_without_botocore():
    code = (
        "import sys, datachain.client.listing_child, datachain.client.s3_pages; "
        "heavy = {'botocore', 's3fs', 'pydantic', 'sqlalchemy'}; "
        "print(sorted(heavy & set(sys.modules)))"
    )
    started = time.monotonic()
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": ROOT},
    )

    assert out.stdout.strip() == "[]"
    assert time.monotonic() - started < 30


def test_s3_clients_list_in_code_point_order():
    assert ClientS3.key_order is code_points  # Keep checks their pages whole


@pytest.mark.parametrize(
    "url,host",
    [
        ("https://Bucket.S3.amazonaws.com:443/", "bucket.s3.amazonaws.com"),
        ("http://minio:80/b", "minio"),
        ("http://MinIO.local:9000/b", "minio.local:9000"),
        ("http://[::1]:9000/b", "[::1]:9000"),
    ],
)
def test_the_signed_host_is_the_one_aiohttp_sends(url, host):
    assert LeanS3(_spec(url=url)).host == host


@pytest.mark.parametrize("name,versions", [("truncated", False), ("versions", True)])
def test_real_pages_read_with_patterns(name, versions):
    from datachain.client.s3_pages import split_page

    split = split_page((PAGES / f"{name}.xml").read_bytes(), versions)

    assert split is not None
    assert split[2]


def test_a_200_error_is_retried(monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    bodies = [
        b"<?xml version='1.0'?><Error><Code>InternalError</Code></Error>",
        b"<ok/>",
    ]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = bodies.pop(0)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(s3_pages.random, "random", lambda: 0)
    lean = LeanS3(_spec(url=f"http://127.0.0.1:{server.server_port}/b", region=None))
    try:
        assert sync(get_loop(), lean._get, {"prefix": ""}) == b"<ok/>"
    finally:
        sync(get_loop(), lean.session.close)
        server.shutdown()
        server.server_close()
