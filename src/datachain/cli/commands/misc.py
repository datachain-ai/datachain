from typing import TYPE_CHECKING

import shtab

if TYPE_CHECKING:
    from datachain.catalog import Catalog


def clear_cache(catalog: "Catalog"):
    catalog.cache.clear()


def garbage_collect(catalog: "Catalog", retention_days: int | None = None):
    temp_tables = catalog.get_temp_table_names()
    cleaned_version_ids = catalog.cleanup_failed_dataset_versions(retention_days)

    total_cleaned = len(temp_tables) + len(cleaned_version_ids)

    if total_cleaned == 0:
        print("Nothing to clean up.")
    else:
        if temp_tables:
            print(f"Garbage collecting {len(temp_tables)} temporary tables.")
            catalog.cleanup_tables(temp_tables)

        if cleaned_version_ids:
            print(
                f"Cleaned {len(cleaned_version_ids)} failed/incomplete dataset versions."
            )


def completion(shell: str) -> str:
    from datachain.cli import get_parser

    return shtab.complete(
        get_parser(),
        shell=shell,
    )
