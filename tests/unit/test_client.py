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
from datachain.client.ranges import Page, key_midpoint
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
    assert lo < key_midpoint(lo, None, alphabet)


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


def test_splits_stay_inside_the_prefix():
    mid = key_midpoint("n2/b.txt", None, "n2/.btx", prefix="n2/")

    assert mid is not None
    assert mid.startswith("n2/")


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
        self, prefix, start_after, token, keep, page_size=None, hi=None
    ) -> Page:
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
