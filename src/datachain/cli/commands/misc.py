from typing import TYPE_CHECKING

import shtab

if TYPE_CHECKING:
    from datachain.catalog import Catalog


def clear_cache(catalog: "Catalog"):
    catalog.cache.clear()


def garbage_collect(catalog: "Catalog", checkpoint_ttl: int | None = None):
    print("Collecting temporary tables...")
    temp_tables = catalog.get_temp_table_names()
    if temp_tables:
        catalog.cleanup_tables(temp_tables)
        print(f"  Removed {len(temp_tables)} temporary tables.")
    else:
        print("  No temporary tables to clean up.")

    print("Collecting failed dataset versions...")
    num_versions = catalog.cleanup_failed_dataset_versions()
    if num_versions:
        print(f"  Removed {num_versions} failed/incomplete dataset versions.")
    else:
        print("  No failed dataset versions to clean up.")

    print("Collecting outdated checkpoints...")
    num_checkpoints = catalog.cleanup_checkpoints(ttl_seconds=checkpoint_ttl)
    if num_checkpoints:
        print(f"  Removed {num_checkpoints} outdated checkpoints.")
    else:
        print("  No outdated checkpoints to clean up.")


def completion(shell: str) -> str:
    from datachain.cli import get_parser

    return shtab.complete(
        get_parser(),
        shell=shell,
    )
