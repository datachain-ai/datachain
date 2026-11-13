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
    all: bool = False,
    team: str | None = None,
    **kwargs,
):
    token = Config().read().get("studio", {}).get("token")
    all, local, studio = determine_flavors(studio, local, all, token)

    show_local = all or local
    show_studio = all or studio
    label_sections = show_local and show_studio

    if show_local:
        if label_sections:
            print("Local:")
        ls_local(sources, long=long, **kwargs)

    if show_studio:
        if label_sections:
            print("\nStudio:")
        ls_remote(sources, long=long, team=team)


def ls_local(
    sources,
    long: bool = False,
    catalog=None,
    client_config=None,
    **kwargs,
):
    from datachain import listings

    catalog = catalog or Session.get().catalog
    if sources:
        actual_sources = list(
            ls_urls(
                sources,
                catalog=catalog,
                long=long,
                client_config=client_config,
                **kwargs,
            )
        )
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
        listing = listings(session=catalog.session).to_list("listing")
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
    client_config: dict | None = None,
    **kwargs,
) -> Iterator[tuple[str, Iterable[str]]]:
    from datachain.client import Client
    from datachain.node import long_line_str

    config = catalog.client_config if client_config is None else client_config
    _prepared_listings: set[str] = set()
    for source in sources:
        client_cls = Client.get_implementation(source)
        if client_cls.is_root_url(source):
            buckets = client_cls.ls_buckets(**config)
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
            for data_source, results in catalog.ls(
                [source],
                fields=fields,
                client_config=client_config,
                _prepared_listings=_prepared_listings,
                **kwargs,
            ):
                values = [_node_data_to_ls_values(r, long) for r in results]
                found = True
                yield data_source.dirname(), values
            if not found:
                raise FileNotFoundError(f"No such file or directory: {source}")
