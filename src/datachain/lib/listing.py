import glob
import logging
import os
import posixpath
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, TypeVar

from fsspec.asyn import get_loop
from sqlalchemy.sql.expression import true

import datachain.fs.utils as fsutils
from datachain.asyn import iter_over_async
from datachain.client import Client
from datachain.error import ClientError
from datachain.lib.file import File
from datachain.query.schema import Column
from datachain.sql.functions import path as pathfunc
from datachain.utils import uses_glob

if TYPE_CHECKING:
    from datachain.lib.dc import DataChain
    from datachain.query.session import Session

LISTING_TTL = 4 * 60 * 60  # cached listing lasts 4 hours
LISTING_PREFIX = "lst__"  # listing datasets start with this name

D = TypeVar("D", bound="DataChain")

# Disable warnings for remote errors in clients
logging.getLogger("aiobotocore.credentials").setLevel(logging.CRITICAL)
logging.getLogger("gcsfs").setLevel(logging.CRITICAL)


def listing_dataset_expired(lst_ds) -> bool:
    """Function that checks if listing dataset is expired or not"""
    lst_version = lst_ds.versions[-1]
    if not lst_version.finished_at:
        return False

    expires = lst_version.finished_at + timedelta(seconds=LISTING_TTL)
    return datetime.now(timezone.utc) > expires


def list_bucket(uri: str, cache, client_config=None) -> Callable:
    """
    Function that returns another generator function that yields File objects
    from bucket where each File represents one bucket entry.
    """

    def list_func() -> Iterator[File]:
        config = client_config or {}
        client = Client.get_client(uri, cache, **config)  # type: ignore[arg-type]
        _, path = Client.parse_url(uri)
        for entries in iter_over_async(client.scandir(path.rstrip("/")), get_loop()):
            yield from entries

    list_func.__name__ = "read_storage"

    return list_func


def get_file_info(uri: str, cache, client_config=None) -> File:
    """
    Wrapper to return File object by its URI
    """
    client = Client.get_client(uri, cache, **(client_config or {}))  # type: ignore[arg-type]
    _, path = Client.parse_url(uri)
    return client.get_file_info(path)


def ls(
    dc: D,
    path: str,
    recursive: bool | None = True,
    column="file",
) -> D:
    """
    Return files by some path from DataChain instance which contains bucket listing.
    Path can have globs.
    If recursive is set to False, only first level children will be returned by
    specified path
    """

    def _file_c(name: str) -> Column:
        return Column(f"{column}.{name}")

    dc = dc.filter(_file_c("is_latest") == true())

    if recursive:
        if not path or path == "/":
            # root of a bucket, returning all latest files from it
            return dc

        if not uses_glob(path):
            # path is not glob, so it's pointing to some directory or a specific
            # file and we are adding proper filter for it
            return dc.filter(
                (_file_c("path") == path)
                | (_file_c("path").glob(path.rstrip("/") + "/*"))
            )

        # path has glob syntax so we are returning glob filter
        return dc.filter(_file_c("path").glob(path))
    # returning only first level children by path
    return dc.filter(pathfunc.parent(_file_c("path")) == path.lstrip("/").rstrip("/*"))


def parse_listing_uri(uri: str) -> tuple[str, str, str]:
    """
    Parsing uri and returns listing dataset name, listing uri and listing path
    """
    storage_uri, path = Client.parse_url(uri)
    if uses_glob(path):
        lst_uri_path = posixpath.dirname(path)
    else:
        storage_uri, path = Client.parse_url(f"{uri.rstrip('/')}/")
        lst_uri_path = path

    lst_uri = f"{storage_uri}/{lst_uri_path.lstrip('/')}"
    ds_name = (
        f"{LISTING_PREFIX}{storage_uri}/{posixpath.join(lst_uri_path, '').lstrip('/')}"
    )

    # we should remove dots from the name
    ds_name = ds_name.replace(".", "_")

    return ds_name, lst_uri, path


def is_listing_dataset(name: str) -> bool:
    """Returns True if it's special listing dataset"""
    return name.startswith(LISTING_PREFIX)


def listing_uri_from_name(dataset_name: str) -> str:
    """Returns clean storage URI from listing dataset name"""
    if not is_listing_dataset(dataset_name):
        raise ValueError(f"Dataset {dataset_name} is not a listing")
    return dataset_name.removeprefix(LISTING_PREFIX)


@contextmanager
def _reraise_as_client_error() -> Iterator[None]:
    try:
        yield
    except Exception as e:
        raise ClientError(message=str(e), error_code=getattr(e, "code", None)) from e


def get_listing(
    uri: str | os.PathLike[str], session: "Session", update: bool = False
) -> tuple[str | None, str, str, bool]:
    """Returns correct listing dataset name that must be used for saving listing
    operation. It takes into account existing listings and reusability of those.
    It also returns boolean saying if returned dataset name is reused / already
    exists or not (on update it always returns False - just because there was no
    reason to complicate it so far). And it returns correct listing path that should
    be used to find rows based on uri.
    """
    from datachain.client.local import FileClient
    from datachain.telemetry import telemetry

    catalog = session.catalog
    cache = catalog.cache
    client_config = catalog.client_config

    client = Client.get_client(uri, cache, **client_config)
    telemetry.log_param("client", client.PREFIX)
    if not isinstance(uri, str):
        uri = os.fspath(uri)

    # we don't want to use cached dataset (e.g. for a single file listing)
    isfile = _reraise_as_client_error()(fsutils.isfile)
    if not glob.has_magic(uri) and not uri.endswith("/") and isfile(client.fs, uri):
        _, path = Client.parse_url(uri)
        return None, uri, path, False

    ds_name, list_uri, list_path = parse_listing_uri(uri)
    listing = None
    listings = [
        ls for ls in catalog.listings() if not ls.is_expired and ls.contains(ds_name)
    ]

    # if no need to update - choosing the most recent one;
    # otherwise, we'll using the exact original `ds_name`` in this case:
    # - if a "bigger" listing exists, we don't want to update it, it's better
    #   to create a new "smaller" one on "update=True"
    # - if an exact listing exists it will have the same name as `ds_name`
    #   anyway below
    if listings and not update:
        listing = sorted(listings, key=lambda ls: ls.created_at)[-1]

    # for local file system we need to fix listing path / prefix
    # if we are reusing existing listing
    if isinstance(client, FileClient) and listing and listing.name != ds_name:
        list_path = f"{ds_name.strip('/').removeprefix(listing.name)}/{list_path}"

    ds_name = listing.name if listing else ds_name
    return ds_name, list_uri, list_path, bool(listing)
