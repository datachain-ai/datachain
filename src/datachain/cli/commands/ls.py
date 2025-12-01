import shlex
from collections.abc import Iterable, Iterator
from itertools import chain
from typing import TYPE_CHECKING

from datachain.cli.utils import determine_flavors
from datachain.config import Config
from datachain.query.session import Session

if TYPE_CHECKING:
    from datachain.catalog import Catalog


def ls(
    sources,
    long: bool = False,
    studio: bool = False,
    local: bool = False,
    all: bool = True,
    team: str | None = None,
    **kwargs,
):
    token = Config().read().get("studio", {}).get("token")
    all, local, studio = determine_flavors(studio, local, all, token)

    if all or local:
        ls_local(sources, long=long, **kwargs)

    if (all or studio) and token:
        ls_remote(sources, long=long, team=team)


def ls_local(
    sources,
    long: bool = False,
    catalog=None,
    client_config=None,
    **kwargs,
):
    from datachain import listings

    if sources:
        session = Session.get(catalog=catalog, client_config=client_config)
        catalog = session.catalog
        actual_sources = list(ls_urls(sources, catalog=catalog, long=long, **kwargs))
        if len(actual_sources) == 1:
            for _, entries in actual_sources:
                for entry in entries:
                    print(format_ls_entry(entry))
        else:
            first = True
            for source, entries in actual_sources:
                # print a newline between directory listings
                if first:
                    first = False
                else:
                    print()
                if source:
                    print(f"{source}:")
                for entry in entries:
                    print(format_ls_entry(entry))
    else:
        # Collect results in a list here to prevent interference from `tqdm` and `print`
        listing = listings().to_list("listing")
        for (ls,) in listing:
            print(format_ls_entry(f"{ls.uri}@v{ls.version}"))  # type: ignore[union-attr]


def format_ls_entry(entry: str) -> str:
    if entry.endswith("/") or not entry:
        entry = shlex.quote(entry[:-1])
        return f"{entry}/"
    return shlex.quote(entry)


def ls_remote(
    paths: Iterable[str],
    long: bool = False,
    team: str | None = None,
):
    from datachain.node import long_line_str
    from datachain.remote.studio import StudioClient

    client = StudioClient(team=team)
    first = True
    for path, response in client.ls(paths):
        if not first:
            print()
        if not response.ok or response.data is None:
            print(f"{path}:\n  Error: {response.message}\n")
            continue

        print(f"{path}:")
        if long:
            for row in response.data:
                entry = long_line_str(
                    row["name"] + ("/" if row["dir_type"] else ""),
                    row["last_modified"],
                )
                print(format_ls_entry(entry))
        else:
            for row in response.data:
                entry = row["name"] + ("/" if row["dir_type"] else "")
                print(format_ls_entry(entry))
        first = False


def ls_urls(
    sources,
    catalog: "Catalog",
    long: bool = False,
    **kwargs,
) -> Iterator[tuple[str, Iterator[str]]]:
    curr_dir = None
    value_iterables = []
    for next_dir, values in _ls_urls_flat(sources, long, catalog, **kwargs):
        if curr_dir is None or next_dir == curr_dir:  # type: ignore[unreachable]
            value_iterables.append(values)
        else:
            yield curr_dir, chain(*value_iterables)  # type: ignore[unreachable]
            value_iterables = [values]
        curr_dir = next_dir
    if curr_dir is not None:
        yield curr_dir, chain(*value_iterables)


def _node_data_to_ls_values(row, long_format=False):
    from datachain.node import DirType, long_line_str

    name = row[0]
    is_dir = row[1] == DirType.DIR
    ending = "/" if is_dir else ""
    value = name + ending
    if long_format:
        last_modified = row[2]
        timestamp = last_modified if not is_dir else None
        return long_line_str(value, timestamp)
    return value


def _ls_urls_flat(
    sources,
    long: bool,
    catalog: "Catalog",
    **kwargs,
) -> Iterator[tuple[str, Iterable[str]]]:
    from datachain.client import Client
    from datachain.node import long_line_str

    for source in sources:
        client_cls = Client.get_implementation(source)
        if client_cls.is_root_url(source):
            buckets = client_cls.ls_buckets(**catalog.client_config)
            if long:
                values: Iterable[str] = (
                    long_line_str(b.name, b.created) for b in buckets
                )
            else:
                values = (b.name for b in buckets)
            yield source, values
        else:
            found = False
            fields = ["name", "dir_type"]
            if long:
                fields.append("last_modified")
            for data_source, results in catalog.ls([source], fields=fields, **kwargs):
                values = [_node_data_to_ls_values(r, long) for r in results]
                found = True
                yield data_source.dirname(), values
            if not found:
                raise FileNotFoundError(f"No such file or directory: {source}")
