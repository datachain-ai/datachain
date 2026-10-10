import os
import sys
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from datachain.client import Client
from datachain.client.fsspec import key_midpoint
from datachain.client.local import FileClient
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
