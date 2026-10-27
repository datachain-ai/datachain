import os
import sys
from pathlib import Path
from typing import NamedTuple

import pytest
from fsspec.asyn import sync
from hypothesis import given
from hypothesis import strategies as st

from datachain.asyn import get_loop
from datachain.client import Client
from datachain.client.fsspec import RangeListedClient
from datachain.client.local import FileClient
from datachain.client.ranges import (
    Keep,
    KeyRange,
    Page,
    code_points,
    key_midpoint,
    split_points,
)
from datachain.client.writeconfig import WriteConfig


def test_bad_protocol():
    with pytest.raises(NotImplementedError):
        Client.get_implementation("bogus://bucket")


def test_write_kwargs_base_default_ignores_content_settings_and_metadata():
    # Backends that don't override _write_kwargs (e.g. HF) fall back to the base,
    # which has no native mapping for content settings or metadata and drops them.
    cfg = WriteConfig(
        content_type="application/pdf",
        content_disposition="attachment",
        metadata={"a": "b"},
    )
    assert Client._write_kwargs(cfg, streaming=True) == {}


def test_write_kwargs_base_default_rejects_write_options():
    # The base rejects the raw escape hatch rather than crash or silently drop it.
    cfg = WriteConfig(write_options={"foo": "bar"})
    with pytest.raises(NotImplementedError, match="write_options"):
        Client._write_kwargs(cfg, streaming=True)


def test_win_paths_are_recognized():
    if sys.platform != "win32":
        pytest.skip()

    assert Client.get_implementation("file://C:/bucket") == FileClient
    assert Client.get_implementation("file://C:\\bucket") == FileClient
    assert Client.get_implementation("file://\\bucket") == FileClient
    assert Client.get_implementation("file:///bucket") == FileClient
    assert Client.get_implementation("C://bucket") == FileClient
    assert Client.get_implementation("C:\\bucket") == FileClient
    assert Client.get_implementation("\bucket") == FileClient


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
def test_parse_file_path_ends_with_slash(cloud_type):
    uri, rel_part = Client.parse_url("./animals/".replace("/", os.sep))
    assert uri == (Path().absolute() / Path("animals")).as_uri()
    assert rel_part == ""


@given(st.text(min_size=1), st.text(min_size=1), st.text())
def test_key_midpoint_is_strictly_inside(a, b, alphabet):
    lo, hi = sorted([a, b])
    mid = key_midpoint(lo, hi, alphabet)
    assert mid is None or lo < mid < hi
    mid = key_midpoint(lo, None, alphabet)
    assert mid is None or lo < mid


def test_no_split_point_starts_with_a_mark_after_the_prefix():
    # Azure lists nothing from a start key like "p/" + U+0301
    assert key_midpoint("p/z", None, "a\u0301\u0410", prefix="p/") is None
    assert key_midpoint("p/a\u0301", "p/b", "a\u0301\u0410", prefix="p/") == "p/a\u0410"


@pytest.mark.parametrize(
    "lo,hi,alphabet,expected",
    [
        ("a", "c", "abc", "b"),
        ("img/0001.jpg", "img/9999.jpg", "img/.jpg0123456789", "img/4"),
        ("a", "a\0", "a", None),
    ],
)
def test_key_midpoint(lo, hi, alphabet, expected):
    assert key_midpoint(lo, hi, alphabet) == expected


def test_key_midpoint_stays_near_dense_keys():
    alphabet = "img/.jpg0123456789"
    assert key_midpoint("img/0999.jpg", None, alphabet) > "img/z"
    assert (
        "img/0999.jpg"
        < key_midpoint("img/0999.jpg", None, alphabet, first="img/0000.jpg")
        < "img/5"
    )


def test_key_midpoint_leaves_two_steps():
    alphabet = "img/.jpg0123456789"
    first = "img/0002.jpg"

    assert key_midpoint("img/0003.jpg", "img/0005.jpg", alphabet, first) == "img/0004"
    assert key_midpoint("img/0003.jpg", "img/0004.jpg", alphabet, first) is None


def _range(lo, hi, age):
    return KeyRange(lo, hi, code_points, age)


def test_young_ranges_split_close_ahead():
    (mid,) = split_points(
        _range("", None, 1), "img/0003.jpg", "img/0000.jpg", "img/.jpg0123"
    )

    assert "img/0003.jpg" < mid < "img/1"


def test_old_ranges_split_far_by_their_start_and_near_by_their_last_page():
    # folders of a page each: the range starts in the gap before folder f2
    far, near = split_points(
        _range("f1/9", None, 9), "f2/0999", "f2/0000", "f/0123456789"
    )

    assert far > "f3"
    assert "f2/0999" < near < far


def test_the_first_range_splits_far_at_its_midpoint():
    assert split_points(_range("", "c", 9), "a", "a", "abc")[0] == "b"


@pytest.mark.parametrize("age", [1, 9])
def test_splits_stay_inside_the_prefix(age):
    mids = split_points(
        _range("n2/a", None, age), "n2/b.txt", "n2/b.txt", "n2/.btx", "n2/"
    )

    assert mids
    assert all(mid.startswith("n2/") for mid in mids)


@pytest.fixture(autouse=True)
def _one_process(mocker):
    mocker.patch("datachain.client.listing_worker.LIST_PROCESSES", 0)  # fakes stay here


class _Row(NamedTuple):
    path: str


class _ReorderingClient(RangeListedClient):
    """Keys ``000``..``039`` in pages of 4; pages from ``reversed_from`` on come
    back in reverse order."""

    PREFIX = "fake://"
    protocol = "fake"
    LIST_PAGE_SIZE = 4

    def __init__(self, reversed_from: str) -> None:
        super().__init__("bucket", {}, None)  # type: ignore[arg-type]
        self.keys = [f"{i:03}" for i in range(40)]
        self.reversed_from = reversed_from
        self.starts: list[str] = []

    def info_to_file(self, v, path):
        raise NotImplementedError

    async def _list_page(
        self, prefix, start_after, token, hi=None, page_size=None, delimiter=""
    ) -> Page:
        keep = Keep(self, start_after, hi)
        if token is None:
            self.starts.append(start_after)
            token = next(
                (i for i, k in enumerate(self.keys) if k > start_after), len(self.keys)
            )
        chunk = self.keys[token : token + (page_size or self.LIST_PAGE_SIZE)]
        end = token + len(chunk)
        if chunk and chunk[0] >= self.reversed_from:
            chunk = chunk[::-1]
        return Page(
            [_Row(k) for k in chunk if keep(k)],
            chunk[0] if chunk else None,
            max(chunk, default=None),
            end if end < len(self.keys) else None,
            keep.past_lo,
            keep.past_hi,
            keep.in_order,
        )


def _listed_paths(client: Client) -> list[str]:
    async def paths():
        return [r.path async for page in client.scandir("") for r in page]

    return sync(get_loop(), paths)


def test_keys_out_of_order_turn_splitting_off(mocker):
    mocker.patch("datachain.client.ranges.LIST_WORKERS", 4)
    client = _ReorderingClient(reversed_from="")

    assert sorted(_listed_paths(client)) == client.keys
    assert client.starts == [""]  # one range


def test_keys_out_of_order_after_a_split_fail_the_listing(mocker):
    mocker.patch("datachain.client.ranges.LIST_WORKERS", 4)
    client = _ReorderingClient(reversed_from="020")

    with pytest.raises(RuntimeError, match="out of order"):
        _listed_paths(client)
    assert len(client.starts) > 1


class _OverSeekingClient(RangeListedClient):
    """Keys ``a/0``..``j/3`` in pages of 4. A start key skips the key after it, as
    SeaweedFS skips a whole folder that a start key names. With ``empty_first``,
    a range's first page is empty but not the last."""

    PREFIX = "fake://"
    protocol = "fake"
    LIST_PAGE_SIZE = 4

    def __init__(self, empty_first: bool = False, churn: bool = False) -> None:
        super().__init__("bucket", {}, None)  # type: ignore[arg-type]
        self.keys = [f"{c}/{j}" for c in "abcdefghij" for j in range(4)]
        self.starts: list[str] = []
        self.empty_first = empty_first
        self.churn = churn  # keys change between the reads instead

    def info_to_file(self, v, path):
        raise NotImplementedError

    async def _list_page(
        self, prefix, start_after, token, hi=None, page_size=None, delimiter=""
    ) -> Page:
        keep = Keep(self, start_after, hi)
        keys = [k for k in self.keys if k.startswith(prefix)]
        first = token is None and start_after and not page_size
        if token is None:
            self.starts.append(start_after)
            token = next((i for i, k in enumerate(keys) if k > start_after), len(keys))
            token += bool(start_after) and not self.churn
            if start_after and self.empty_first:
                return Page([], None, None, token)
        chunk = keys[token : token + (page_size or self.LIST_PAGE_SIZE)]
        end = token + len(chunk)
        if self.churn and first and chunk:
            # once a range above has read, its first key goes and a new one comes
            self.keys.remove(chunk[0])
            self.keys = sorted([*self.keys, f"{start_after}!"])
        return Page(
            [_Row(k) for k in chunk if keep(k)],
            chunk[0] if chunk else None,
            chunk[-1] if chunk else None,
            end if end < len(keys) else None,
            keep.past_lo,
            keep.past_hi,
            keep.in_order,
        )


@pytest.mark.parametrize("empty_first", [False, True])
def test_keys_skipped_past_a_start_key_fail_the_listing(mocker, empty_first):
    mocker.patch("datachain.client.ranges.LIST_WORKERS", 4)
    client = _OverSeekingClient(empty_first)

    with pytest.raises(RuntimeError, match="skipped keys"):
        _listed_paths(client)
    assert len(client.starts) > 1


def test_range_listing_can_be_turned_off(mocker, monkeypatch):
    mocker.patch("datachain.client.ranges.LIST_WORKERS", 4)
    monkeypatch.setenv("DATACHAIN_RANGE_LISTING", "0")
    client = _OverSeekingClient()

    assert sorted(_listed_paths(client)) == client.keys
    assert client.starts == [""]


class _HiddenInversionClient(RangeListedClient):
    """Keys ``000``..``039`` in pages of 4. Once the listing has split, one page
    inside a range comes back with the first key past the range second: out of
    order, though that key never becomes a row."""

    PREFIX = "fake://"
    protocol = "fake"
    LIST_PAGE_SIZE = 4

    def __init__(self) -> None:
        super().__init__("bucket", {}, None)  # type: ignore[arg-type]
        self.keys = [f"{i:03}" for i in range(40)]
        self.inverted = False

    def info_to_file(self, v, path):
        raise NotImplementedError

    async def _list_page(
        self, prefix, start_after, token, hi=None, page_size=None, delimiter=""
    ) -> Page:
        keep = Keep(self, start_after, hi)
        if token is None:
            token = next(
                (i for i, k in enumerate(self.keys) if k > start_after), len(self.keys)
            )
        chunk = self.keys[token : token + (page_size or self.LIST_PAGE_SIZE)]
        end = token + len(chunk)
        after = next((k for k in self.keys if hi is not None and k > hi), None)
        if not self.inverted and chunk and after and chunk[-1] < hi:
            chunk = [chunk[0], after, *chunk[1:]]
            self.inverted = True
        return Page(
            [_Row(k) for k in chunk if keep(k)],
            chunk[0] if chunk else None,
            max(chunk, default=None),
            end if end < len(self.keys) else None,
            keep.past_lo,
            keep.past_hi,
            keep.in_order,
        )


def test_keys_out_of_order_past_a_range_end_fail_the_listing(mocker):
    mocker.patch("datachain.client.ranges.LIST_WORKERS", 4)
    client = _HiddenInversionClient()

    with pytest.raises(RuntimeError, match="out of order"):
        _listed_paths(client)
    assert client.inverted


def test_keys_written_during_the_listing_do_not_fail_it(mocker):
    mocker.patch("datachain.client.ranges.LIST_WORKERS", 4)
    client = _OverSeekingClient(churn=True)
    stable = set(client.keys)

    listed = _listed_paths(client)
    assert len(listed) == len(set(listed))
    assert len(client.starts) > 1
    assert {k for k in stable if k in client.keys} <= set(listed)
