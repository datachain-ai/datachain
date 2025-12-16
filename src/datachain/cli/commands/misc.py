from typing import TYPE_CHECKING

import shtab

if TYPE_CHECKING:
    from datachain.catalog import Catalog


def clear_cache(catalog: "Catalog"):
    catalog.cache.clear()


def garbage_collect(catalog: "Catalog"):
    temp_tables = catalog.get_temp_table_names()
    num_versions_removed = catalog.cleanup_failed_dataset_versions()

    total_cleaned = len(temp_tables) + num_versions_removed

    if total_cleaned == 0:
        print("Nothing to clean up.")
    else:
        if temp_tables:
            print(f"Garbage collecting {len(temp_tables)} tables.")
            catalog.cleanup_tables(temp_tables)

        if num_versions_removed:
            print(f"Cleaned {num_versions_removed} failed/incomplete dataset versions.")


def completion(shell: str) -> str:
    from datachain.cli import get_parser

    return shtab.complete(
        get_parser(),
        shell=shell,
    )
