"""Shapes of the snapshots fed to the knowledge enrichment prompts.

The prompts in `../prompts/` (`enrich_bucket.md`, `enrich.md`) are the authoritative
spec for what each field means and how it renders.
"""

from typing import Any

from typing_extensions import NotRequired, TypedDict


class PreviewData(TypedDict):
    columns: list[str]
    rows: list[list[Any]]
    file_url_prefix: NotRequired[str]


class ExtensionStat(TypedDict):
    ext: str
    count: int
    total_bytes: int
    pct_count: float
    pct_bytes: float


class DirStat(TypedDict):
    path: str
    files: int
    bytes: int
    depth: int


class SizeDistribution(TypedDict):
    min_bytes: NotRequired[int]
    max_bytes: NotRequired[int]
    median_bytes: NotRequired[int]
    p10_bytes: NotRequired[int]
    p90_bytes: NotRequired[int]
    empty_count: NotRequired[int]


class TimeRange(TypedDict):
    oldest: NotRequired[str | None]
    newest: NotRequired[str | None]


class ListingMeta(TypedDict):
    listing_uuid: str | None
    listing_created: str | None
    listing_finished: str | None
    listing_expires: str | None
    listing_expired: bool | None


class BucketSnapshot(TypedDict):
    uri: str
    scheme: str
    bucket: str
    prefix: str
    anon: bool | None
    sampled: bool
    scanned: str | None
    listing_uuid: NotRequired[str | None]
    listing_created: NotRequired[str | None]
    listing_finished: NotRequired[str | None]
    listing_expires: NotRequired[str | None]
    listing_expired: NotRequired[bool | None]
    total_files: int | None
    total_size_bytes: int | None
    max_depth: int | None
    extensions: list[ExtensionStat]
    directories: list[DirStat]
    size_distribution: SizeDistribution | None
    time_range: TimeRange | None
    samples: dict[str, Any]
    file_url_prefix: NotRequired[str | None]
    dataset_name: NotRequired[str]
    warnings: NotRequired[list[str]]


class SchemaEntry(TypedDict):
    type: str
    fields: dict[str, str] | None


class DependencyEntry(TypedDict):
    name: str | None
    version: str | None
    type: str | None
    file_path: NotRequired[str]


class DepRef(TypedDict):
    name: str
    version: str | None


class DepUpdate(TypedDict):
    name: str
    version_from: str | None
    version_to: str | None


class DepChanges(TypedDict):
    deps_added: list[DepRef]
    deps_removed: list[DepRef]
    deps_updated: list[DepUpdate]


class ChangesEntry(DepChanges):
    previous_version: str
    script_changed: bool
    previous_script: str | None


class DatasetVersionEntry(TypedDict):
    version: str
    uuid: str | None
    records: int | None
    updated: str | None
    schema: dict[str, SchemaEntry]
    preview: PreviewData | None
    summary: dict[str, Any] | None
    query_script: str | None
    changes: ChangesEntry | None
    dependencies: list[DependencyEntry]


class DatasetSnapshot(TypedDict):
    name: str
    source: str
    attrs: list[str]
    description: str | None
    versions: list[DatasetVersionEntry]
    warnings: NotRequired[list[str]]
