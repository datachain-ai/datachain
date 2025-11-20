import posixpath
import re
from datetime import datetime
from struct import pack
from time import sleep
from typing import Any

import msgpack
import pytest

import datachain as dc
from datachain.cli import ls
from datachain.config import Config, ConfigLevel
from datachain.lib.listing import LISTING_PREFIX


@pytest.fixture
def studio_config(global_config_dir):
    with Config(ConfigLevel.GLOBAL).edit() as conf:
        conf["studio"] = {"token": "isat_access_token", "team": "team_name"}


def same_lines(lines1, lines2):
    def _split_lines(lines):
        return [line.strip() for line in sorted(lines.split("\n"))]

    return _split_lines(lines1) == _split_lines(lines2)


def test_ls_no_args(cloud_test_catalog, cloud_type, capsys):
    session = cloud_test_catalog.session
    catalog = session.catalog
    src = cloud_test_catalog.src_uri

    dc.read_storage(src, session=session).exec()
    ls([], catalog=catalog)
    captured = capsys.readouterr()
    assert captured.out == f"{src}/@v1.0.0\n"


def test_ls_root(cloud_test_catalog, cloud_type, capsys):
    src = cloud_test_catalog.src_uri
    root = src[: src.find("://") + 3]
    src_name = src.replace(root, "", 1)
    ls([root], catalog=cloud_test_catalog.catalog)
    captured = capsys.readouterr()
    if cloud_type == "file":
        assert not captured.out
    else:
        buckets = captured.out.split("\n")
        assert src_name in buckets


def ls_sources_output(src):
    return """\
cats/
description
dogs/
dogs:
dog1
dog2
dog3

dogs/others:
dog4
    """


def test_ls_sources(cloud_test_catalog, cloud_type, capsys):
    src = cloud_test_catalog.src_uri
    ls([src], catalog=cloud_test_catalog.catalog)
    ls([f"{src}/dogs/*"], catalog=cloud_test_catalog.catalog)
    captured = capsys.readouterr()
    assert same_lines(captured.out, ls_sources_output(src))


def test_ls_not_found(cloud_test_catalog):
    src = cloud_test_catalog.src_uri
    with pytest.raises(FileNotFoundError):
        ls([f"{src}/cats/bogus*"], catalog=cloud_test_catalog.catalog)


# TODO return file test when https://github.com/datachain-ai/datachain/issues/318
# is done
@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
def test_ls_not_a_directory(cloud_test_catalog, capsys):
    src = cloud_test_catalog.src_uri
    with pytest.raises(FileNotFoundError):
        ls([f"{src}/description/"], catalog=cloud_test_catalog.catalog)


def ls_glob_output(src, cloud_type):
    if cloud_type == "file":
        return """\
dog1
dog2
dog3

others:
dog4
    """

    return """\
dogs/others:
dog4

dogs:
dog1
dog2
dog3
    """


def test_ls_glob_sub(cloud_test_catalog, cloud_type, capsys):
    src = cloud_test_catalog.src_uri
    ls([f"{src}/dogs/*"], catalog=cloud_test_catalog.catalog)
    captured = capsys.readouterr()
    assert same_lines(captured.out, ls_glob_output(src, cloud_type))


def list_dataset_name(uri, path):
    return f"{LISTING_PREFIX}{uri}/{posixpath.join(path, '').lstrip('/')}"


def test_ls_partial_indexing(cloud_test_catalog, cloud_type, capsys):
    catalog = cloud_test_catalog.catalog
    src = cloud_test_catalog.src_uri

    ls([f"{src}/dogs/others/"], catalog=cloud_test_catalog.catalog)
    # These sleep calls are here to ensure that capsys can fully capture the output
    # and to avoid any flaky tests due to multithreading generating output out of order
    sleep(0.05)
    captured = capsys.readouterr()
    assert "Listing" in captured.err
    assert captured.out == "dog4\n"

    ls([f"{src}/cats/"], catalog=cloud_test_catalog.catalog)
    sleep(0.05)
    captured = capsys.readouterr()
    assert sorted(ls.name for ls in catalog.listings()) == [
        list_dataset_name(src, "cats/"),
        list_dataset_name(src, "dogs/others/"),
    ]
    assert "Listing" in captured.err
    assert same_lines("cat1\ncat2\n", captured.out)

    ls([f"{src}/dogs/"], catalog=cloud_test_catalog.catalog)
    sleep(0.05)
    captured = capsys.readouterr()
    assert sorted(ls.name for ls in catalog.listings()) == [
        list_dataset_name(src, "cats/"),
        list_dataset_name(src, "dogs/"),
        list_dataset_name(src, "dogs/others/"),
    ]
    assert "Listing" in captured.err
    assert same_lines("others/\ndog1\ndog2\ndog3\n", captured.out)

    ls([f"{src}/cats/"], catalog=cloud_test_catalog.catalog)
    sleep(0.05)
    captured = capsys.readouterr()
    assert sorted(ls.name for ls in catalog.listings()) == [
        list_dataset_name(src, "cats/"),
        list_dataset_name(src, "dogs/"),
        list_dataset_name(src, "dogs/others/"),
    ]
    assert "Listing" not in captured.err
    assert same_lines("cat1\ncat2\n", captured.out)

    ls([f"{src}/"], catalog=cloud_test_catalog.catalog)
    sleep(0.05)
    captured = capsys.readouterr()
    assert sorted(ls.name for ls in catalog.listings()) == [
        list_dataset_name(src, ""),
        list_dataset_name(src, "cats/"),
        list_dataset_name(src, "dogs/"),
        list_dataset_name(src, "dogs/others/"),
    ]
    assert "Listing" in captured.err
    assert same_lines("cats/\ndogs/\ndescription\n", captured.out)


class MockResponse:
    def __init__(self, content, status_code, ok=True):
        self.content = content
        self.ok = ok
        self.status_code = status_code


def mock_post(method, url, data=None, json=None, **kwargs):
    source = json["source"]
    path = re.sub(r"\w+://[^/]+/?", "", source).rstrip("/")
    data = [
        {
            **d,
            "path_str": d["path_str"].format(src=path),
            "path": d["path_str"].format(src=path).split("/"),
        }
        for d in REMOTE_DATA[path]
    ]
    return MockResponse(
        content=msgpack.packb({"data": data}, default=_pack_extended_types),
        status_code=200,
    )


def _pack_extended_types(obj):
    if isinstance(obj, datetime):
        if obj.tzinfo:
            data = (obj.timestamp(), int(obj.utcoffset().total_seconds()))
            return msgpack.ExtType(42, pack("!dl", *data))
        data = (obj.timestamp(),)
        return msgpack.ExtType(42, pack("!d", *data))
    raise TypeError(f"Unknown type: {type(obj)}")


ls_remote_sources_output = """\
{src}:
cats/
dogs/
description

{src}/dogs/others:
dog4

{src}/dogs:
dog1
dog2
dog3
"""


def test_ls_remote_sources(cloud_type, capsys, monkeypatch, studio_config):
    src = f"{cloud_type}://bucket"
    with monkeypatch.context() as m:
        m.setattr("requests.request", mock_post)
        ls([src, f"{src}/dogs/others", f"{src}/dogs"], studio=True)
    captured = capsys.readouterr()
    assert captured.out == ls_remote_sources_output.format(src=src)


REMOTE_DATA: dict[str, list[dict[str, Any]]] = {
    "": [
        {
            "id": 816,
            "dir_type": 1,
            "name": "cats",
            "etag": "",
            "version": "",
            "is_latest": True,
            "last_modified": datetime(2023, 1, 17, 21, 39, 0, 88564),
            "size": 0,
            "path_str": "{src}/cats",
            "path": [],
        },
        {
            "id": 825,
            "dir_type": 1,
            "name": "dogs",
            "etag": "",
            "version": "",
            "is_latest": True,
            "last_modified": datetime(2023, 1, 17, 21, 39, 0, 88567),
            "size": 0,
            "path_str": "{src}/dogs",
            "path": [],
        },
        {
            "id": None,
            "dir_type": 0,
            "name": "description",
            "etag": "20664550afa2654017377ceb266a1f82",
            "version": "",
            "is_latest": True,
            "last_modified": datetime(2022, 2, 10, 3, 39, 9),
            "size": 350496,
            "path_str": "{src}/description",
            "path": [],
        },
    ],
    "dogs/others": [
        {
            "id": None,
            "dir_type": 0,
            "name": "dog4",
            "etag": "c4e42ce24d92bb5b4c4be9a99b237502",
            "version": "",
            "is_latest": True,
            "last_modified": datetime(2022, 6, 28, 22, 39, 1),
            "size": 32975,
            "path_str": "{src}/dogs/others/dog4",
            "path": [],
        },
    ],
    "dogs": [
        {
            "id": None,
            "dir_type": 0,
            "name": "dog1",
            "etag": "44a632238558e0aa4c54bdb901bf9cff",
            "version": "",
            "is_latest": True,
            "last_modified": datetime(2022, 6, 28, 22, 39, 1),
            "size": 101,
            "path_str": "{src}/dogs/dog1",
            "path": [],
        },
        {
            "id": None,
            "dir_type": 0,
            "name": "dog2",
            "etag": "76556c960239c50e5a8f8569daf85355",
            "version": "",
            "is_latest": True,
            "last_modified": datetime(2022, 6, 28, 22, 39, 1),
            "size": 29759,
            "path_str": "{src}/dogs/dog2",
            "path": [],
        },
        {
            "id": None,
            "dir_type": 0,
            "name": "dog3",
            "etag": "b1c99fedcf77bf5fa62984e93db1955c",
            "version": "",
            "is_latest": True,
            "last_modified": datetime(2022, 6, 28, 22, 39, 1),
            "size": 102,
            "path_str": "{src}/dogs/dog3",
            "path": [],
        },
    ],
}
