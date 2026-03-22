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
    script: bool = False,
) -> None:
    from datachain import Session, read_dataset
    from datachain.query.dataset import DatasetQuery
    from datachain.utils import show_records

    dataset = catalog.get_dataset(name, include_incomplete=False)
    dataset_version = dataset.get_version(version or dataset.latest_version)

    if script:
        print(dataset_version.query_script)
        return

    if include_hidden:
        hidden_fields = []
    else:
        hidden_fields = SignalSchema.get_flatten_hidden_fields(
            dataset_version.feature_schema
        )

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
