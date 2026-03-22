from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datachain.catalog import Catalog


def show(
    catalog: "Catalog",
    name: str,
    version: str | None = None,
    limit: int = 10,
    offset: int = 0,
    columns: Sequence[str] = (),
    no_collapse: bool = False,
    schema: bool = False,
    include_hidden: bool = False,
) -> None:
    from datachain import Session, read_dataset

    session = Session.get(catalog=catalog)
    dc = read_dataset(name=name, version=version, session=session)
    if columns:
        dc = dc.select(*columns)
    if offset:
        dc = dc.offset(offset)

    dc.show(limit=limit, flatten=no_collapse, include_hidden=include_hidden)

    if schema:
        print("\nSchema:")
        dc.print_schema()
