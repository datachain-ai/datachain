from typing import TYPE_CHECKING

from datachain.lib.listing import LISTING_PREFIX, ls
from datachain.lib.listing_info import ListingInfo
from datachain.lib.settings import Settings
from datachain.lib.signal_schema import SignalSchema
from datachain.query import Session
from datachain.query.dataset import DatasetQuery, QueryStep, step_result

from .values import read_values

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    from datachain.dataset import DatasetVersion
    from datachain.query.dataset import StepResult

    from .datachain import DataChain

    P = ParamSpec("P")


class ReadOnlyQueryStep(QueryStep):
    """
    This step is used to read the dataset in read-only mode.
    It is used to avoid the need to read the table metadata from the warehouse.
    This is useful when we want to list the files in the dataset.
    """

    def apply(self) -> "StepResult":
        import sqlalchemy as sa

        def q(*columns):
            return sa.select(*columns)

        table_name = self.catalog.warehouse.dataset_table_name(
            self.dataset, self.dataset_version
        )
        dataset_row_cls = self.catalog.warehouse.schema.dataset_row_cls
        table = dataset_row_cls.new_table(
            table_name,
            columns=(
                [
                    *dataset_row_cls.sys_columns(),
                    *dataset_row_cls.listing_columns(),
                ]
            ),
        )

        return step_result(
            q, table.columns, dependencies=[(self.dataset, self.dataset_version)]
        )


def listings(
    session: Session | None = None,
    in_memory: bool = False,
    column: str = "listing",
    **kwargs,
) -> "DataChain":
    """Generate chain with list of cached listings.
    Listing is a special kind of dataset which has directory listing data of
    some underlying storage (e.g S3 bucket).

    Example:
        ```py
        import datachain as dc
        dc.listings().show()
        ```
    """
    session = Session.get(session, in_memory=in_memory)
    catalog = kwargs.get("catalog") or session.catalog

    return read_values(
        session=session,
        in_memory=in_memory,
        output={column: ListingInfo},
        **{column: catalog.listings()},  # type: ignore[arg-type]
    )


def read_listing_dataset(
    name: str,
    version: str | None = None,
    path: str = "",
    session: Session | None = None,
    settings: dict | None = None,
) -> tuple["DataChain", "DatasetVersion"]:
    """Read a listing dataset and return a DataChain and listing version.

    Args:
        name: Name of the dataset
        version: Version of the dataset
        path: Path within the listing to read. Path can have globs.
        session: Optional Session object to use for reading
        settings: Optional settings dictionary to use for reading

    Returns:
        tuple[DataChain, DatasetVersion]: A tuple containing:
            - DataChain configured for listing files
            - DatasetVersion object for the specified listing version

    Example:
        ```py
        import datachain as dc
        chain, listing_version = dc.read_listing_dataset(
            "lst__s3://my-bucket/my-path", version="1.0.0", path="my-path"
        )
        chain.show()
        ```
    """
    # Configure and return a DataChain for reading listing dataset files
    # Uses ReadOnlyQueryStep to avoid warehouse metadata lookups
    from datachain.lib.dc import Sys
    from datachain.lib.file import File

    from .datachain import DataChain

    if not name.startswith(LISTING_PREFIX):
        name = LISTING_PREFIX + name

    session = Session.get(session)
    dataset = session.catalog.get_dataset(name, include_incomplete=False)
    if version is None:
        version = dataset.latest_version

    query = DatasetQuery(name=name, session=session)

    if settings:
        cfg = {**settings}
        if "prefetch" not in cfg:
            cfg["prefetch"] = 0
        _settings = Settings(**cfg)
    else:
        _settings = Settings(prefetch=0)
    signal_schema = SignalSchema({"sys": Sys, "file": File})

    query.starting_step = ReadOnlyQueryStep(query.catalog, dataset, version)
    query.version = version
    # We already know that this is a listing dataset,
    # so we can set the listing function to True
    query.set_listing_fn(lambda: True)

    chain = DataChain(query, _settings, signal_schema)
    chain = ls(chain, path, recursive=True, column="file")

    return chain, dataset.get_version(version)
