from collections.abc import Sequence
from copy import copy
from functools import wraps
from typing import TYPE_CHECKING, TypeVar

import datachain
from datachain.dataset import DatasetDependency, DatasetRecord
from datachain.error import (
    DatasetNotFoundError,
    DatasetVersionNotFoundError,
    SchemaDriftError,
)
from datachain.project import Project
from datachain.query.dataset import UnionSchemaMismatchError
from datachain.query.session import Session

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Concatenate

    from typing_extensions import ParamSpec

    from datachain.lib.dc import DataChain
    from datachain.lib.signal_schema import SignalSchema
    from datachain.query.dataset import DatasetQuery, DeltaSpec

    P = ParamSpec("P")


T = TypeVar("T", bound="DataChain")


def _dataset_identity(dataset: DatasetRecord) -> tuple[str, str, str]:
    return (
        dataset.project.namespace.name,
        dataset.project.name,
        dataset.name,
    )


def delta_disabled(
    method: "Callable[Concatenate[T, P], T]",
) -> "Callable[Concatenate[T, P], T]":
    """
    Decorator for disabling DataChain methods (e.g `.agg()` or `.union()`) to
    work with delta updates. It throws `NotImplementedError` if any
    participating chain is delta-enabled without `delta_unsafe=True`.
    """

    @wraps(method)
    def _inner(self: T, *args: "P.args", **kwargs: "P.kwargs") -> T:
        from datachain.lib.dc import DataChain

        if not isinstance(self, DataChain):
            raise TypeError(
                f"delta_disabled can only wrap DataChain methods, got {type(self)!r}"
            )

        operands = [
            value
            for value in (self, *args, *kwargs.values())
            if isinstance(value, DataChain)
        ]
        if not operands:
            raise RuntimeError(
                "delta_disabled expected at least one DataChain operand "
                f"for {method.__name__}"
            )

        if any(operand.delta and not operand.delta_unsafe for operand in operands):
            raise NotImplementedError(
                f"Cannot use {method.__name__} with delta datasets - may cause"
                " inconsistency. Set delta_unsafe=True on every participating"
                " delta source to allow this operation."
            )
        return method(self, *args, **kwargs)

    return _inner


def _rewrite_query_for_delta_replay(
    source_replacements: Sequence[tuple["DatasetQuery", "DataChain"]],
    original_query_chain: "DataChain",
) -> "DataChain":
    # Clone the original query and replace all delta source occurrences with
    # their replay inputs before executing the tree once.
    replay = original_query_chain.clone()
    replay_query = original_query_chain._query
    for source_occurrence_query, replacement_source_chain in source_replacements:
        replay_query = replay_query.replace_source(
            source_occurrence_query, replacement_source_chain._query
        )

    replay._query = replay_query
    replay.signals_schema = original_query_chain.signals_schema
    return replay


def _format_schema_drift_message(
    context: str,
    existing_schema: "SignalSchema",
    updated_schema: "SignalSchema",
) -> tuple[str, bool]:
    missing_cols, new_cols = existing_schema.compare_signals(updated_schema)

    if not new_cols and not missing_cols:
        return "", False

    parts: list[str] = []
    if new_cols:
        parts.append("new columns detected: " + ", ".join(sorted(new_cols)))
    if missing_cols:
        parts.append(
            "columns missing in updated data: " + ", ".join(sorted(missing_cols))
        )

    details = "; ".join(parts)
    message = f"Delta update failed: schema drift detected while {context}: {details}."

    return message, True


def _safe_union(
    left: "DataChain",
    right: "DataChain",
    context: str,
) -> "DataChain":
    try:
        return left.union(right)
    except UnionSchemaMismatchError as exc:
        message, has_drift = _format_schema_drift_message(
            context,
            left.signals_schema,
            right.signals_schema,
        )
        if has_drift:
            raise SchemaDriftError(message) from exc
        raise


def _read_source(
    name: str, project: Project, version: str, current_only: bool
) -> "DataChain":
    from datachain.lib.dc import C

    chain = datachain.read_dataset(
        name,
        namespace=project.namespace.name,
        project=project.name,
        version=version,
    )
    if current_only:
        column = chain.signals_schema.get_file_signal()
        assert column is not None
        chain = chain.filter(C(f"{column}.is_latest"))
    return chain


def _build_source_diff_chain(
    source_ds_name: str,
    source_ds_project: Project,
    source_ds_version: str,
    source_ds_latest_version: str,
    on: str | Sequence[str],
    compare: str | Sequence[str] | None = None,
    current_only: bool = False,
    deleted: bool = False,
) -> "DataChain":
    """Get delta chain for processing changes between versions."""
    source_dc = _read_source(
        source_ds_name,
        source_ds_project,
        source_ds_version,
        current_only,
    )
    source_dc_latest = _read_source(
        source_ds_name,
        source_ds_project,
        source_ds_latest_version,
        current_only,
    )

    # Calculate diff between source versions
    return source_dc_latest.diff(source_dc, on=on, compare=compare, deleted=deleted)


def _build_source_retry_chain(
    result_dataset: "DataChain",
    source_ds_name: str,
    source_ds_project: Project,
    source_ds_version: str,
    on: str | Sequence[str],
    right_on: str | Sequence[str] | None,
    delta_retry: bool | str | None,
    diff_chain: "DataChain | None",
    current_only: bool = False,
) -> "DataChain | None":
    """Get retry chain for processing error records and missing records."""
    # Import here to avoid circular import
    from datachain.lib.dc import C

    retry_chain = None
    source_dc = _read_source(
        source_ds_name,
        source_ds_project,
        source_ds_version,
        current_only,
    )

    # Handle error records if delta_retry is a string (column name)
    if isinstance(delta_retry, str):
        error_records = result_dataset.filter(C(delta_retry) != "")
        retry_chain = source_dc.subtract(
            source_dc.subtract(error_records, on=on, right_on=right_on), on=on
        )

    # Handle missing records if delta_retry is True
    elif delta_retry is True:
        missing_records = source_dc.subtract(result_dataset, on=on, right_on=right_on)
        retry_chain = missing_records

    # Subtract also diff chain since some items might be picked
    # up by `delta=True` itself (e.g. records got modified AND are missing in the
    # result dataset atm)
    on = [on] if isinstance(on, str) else on

    if retry_chain is None:
        return None
    if diff_chain is not None:
        retry_chain = retry_chain.subtract(diff_chain, on=on)
    return retry_chain.distinct(*on)


def _get_source_info(
    source_ds: DatasetRecord,
    result_dependencies: list[DatasetDependency | None],
    catalog,
) -> tuple[
    str | None,
    Project | None,
    str | None,
    str | None,
]:
    """Get source dataset information using already-fetched result dependencies.

    Returns:
        Tuple of (source_name, source_project, source_version, source_latest_version).
        Returns (None, None, None, None) if source dataset was removed.
    """
    source_ds_dep = next(
        (
            result_dependency
            for result_dependency in result_dependencies
            if result_dependency
            and result_dependency.name == source_ds.name
            and result_dependency.project == source_ds.project.name
            and result_dependency.namespace == source_ds.project.namespace.name
        ),
        None,
    )
    if not source_ds_dep:
        # Starting dataset was removed, back off to normal dataset creation
        return None, None, None, None

    # Refresh starting dataset to have new versions if they are created
    source_ds = catalog.get_dataset(
        source_ds.name,
        namespace_name=source_ds.project.namespace.name,
        project_name=source_ds.project.name,
        versions=None,
        include_incomplete=False,
    )

    return (
        source_ds.name,
        source_ds.project,
        source_ds_dep.version,
        source_ds.latest_version,
    )


def _normalize_dependencies(
    dependencies: list[DatasetDependency | None],
    updated_versions: dict[tuple[str, str, str], str],
) -> list[DatasetDependency]:
    normalized = [d for d in copy(dependencies) if d is not None]
    for dep in normalized:
        key = (dep.namespace, dep.project, dep.name)
        if key in updated_versions:
            dep.version = updated_versions[key]
    return normalized


def _get_shared_result_key(
    delta_sources: Sequence["DatasetQuery"],
) -> tuple[str, ...]:
    result_keys = {
        (fields,)
        if isinstance(fields := source.delta_spec.right_on or source.delta_spec.on, str)
        else tuple(fields)
        for source in delta_sources
        if source.delta_spec is not None
    }
    if len(result_keys) != 1:
        raise NotImplementedError(
            "Delta sources in the same query must use the same result key"
        )

    return next(iter(result_keys))


def _can_replay_changed_keys(query: "DatasetQuery", key: Sequence[str]) -> bool:
    # Inputs without delta sources are replayed in full, so only steps on the path
    # of changed rows matter.
    return all(
        step.can_replay_changed_keys(key)
        and all(
            _can_replay_changed_keys(child, key)
            for child in step.sub_queries
            if child.delta_sources()
        )
        for step in query.steps
    )


def _changed_keys_are_aligned(
    changed_keys: Sequence[tuple["DeltaSpec", "DataChain | None"]],
) -> bool:
    (first_spec, first_keys), *others = changed_keys
    return all(
        first_keys is not None
        and keys is not None
        and first_keys.subtract(keys, on=first_spec.on, right_on=spec.on).empty
        and keys.subtract(first_keys, on=spec.on, right_on=first_spec.on).empty
        for spec, keys in others
    )


def _static_input_changed(
    query: "DatasetQuery", dependencies: list[DatasetDependency | None]
) -> bool:
    source = query.starting_step
    # Inline inputs have no recorded dependency and are assumed fixed.
    if (
        query.delta_spec is None
        and source is not None
        and not Session.is_temp_dataset(source.dataset.name)
    ):
        identity = _dataset_identity(source.dataset)
        if not any(
            dep is not None
            and (dep.namespace, dep.project, dep.name) == identity
            and dep.version == source.dataset_version
            for dep in dependencies
        ):
            return True
    return any(
        _static_input_changed(child, dependencies)
        for step in query.steps
        for child in step.sub_queries
    )


def _persist_keys(
    chains: Sequence["DataChain"], on: str | Sequence[str]
) -> "DataChain | None":
    fields = [on] if isinstance(on, str) else on
    keys = None
    for chain in chains:
        selected = chain._evolve(
            signal_schema=chain.signals_schema.clone_without_sys_signals()
        ).select(*fields)
        keys = selected if keys is None else keys.union(selected)
    if keys is None:
        return None
    keys = keys.persist()
    return None if keys.empty else keys


def _build_source_replay_input(
    target_source_query: "DatasetQuery",
    target_source_spec: "DeltaSpec",
    result_dataset: "DataChain",
    result_dependencies: list[DatasetDependency | None],
) -> tuple["DataChain | None", "DataChain | None", str | None, str | None]:
    """Return replay input, persisted replacement keys and source versions.

    No changes produce no input or keys; a missing dependency also has no versions.
    """
    if target_source_query.starting_step is None:  # pragma: no cover
        raise RuntimeError("Source dataset must be resolved before processing delta")

    source_ds_name, source_ds_project, source_ds_version, source_ds_latest_version = (
        _get_source_info(
            target_source_query.starting_step.dataset,
            result_dependencies,
            target_source_query.catalog,
        )
    )

    if source_ds_name is None:
        return None, None, None, None

    assert source_ds_project is not None
    assert source_ds_version is not None
    assert source_ds_latest_version is not None

    current_only = target_source_query.listing_fn is not None
    versions_changed = source_ds_version != source_ds_latest_version
    diff_chain = None
    if versions_changed:
        diff_chain = _build_source_diff_chain(
            source_ds_name,
            source_ds_project,
            source_ds_version,
            source_ds_latest_version,
            target_source_spec.on,
            target_source_spec.compare,
            current_only=current_only,
        )

    changed = []
    if versions_changed:
        # Deleted records carry their old key, so this also covers removals.
        changed.append(
            _build_source_diff_chain(
                source_ds_name,
                source_ds_project,
                source_ds_version,
                source_ds_latest_version,
                target_source_spec.on,
                target_source_spec.compare,
                current_only=current_only,
                deleted=True,
            )
        )

    retry_chain = None
    if target_source_spec.delta_retry:
        retry_chain = _build_source_retry_chain(
            result_dataset,
            source_ds_name,
            source_ds_project,
            source_ds_latest_version,
            target_source_spec.on,
            target_source_spec.right_on,
            target_source_spec.delta_retry,
            diff_chain,
            current_only=current_only,
        )

    source_replay_input = diff_chain
    if retry_chain is not None:
        changed.append(retry_chain)
        source_replay_input = (
            _safe_union(
                diff_chain,
                retry_chain,
                context="combining retry records with delta changes",
            )
            if diff_chain is not None
            else retry_chain
        )

    return (
        source_replay_input,
        _persist_keys(changed, target_source_spec.on),
        source_ds_version,
        source_ds_latest_version,
    )


def delta_retry_update(  # noqa: PLR0911
    dc: "DataChain",
    namespace_name: str,
    project_name: str,
    name: str,
) -> tuple["DataChain | None", list[DatasetDependency] | None, bool]:
    """
    Creates new chain that consists of the last version of current delta dataset
    plus diff from the source with all needed modifications.
    This way we don't need to re-calculate the whole chain from the source again
    (apply all the DataChain methods like filters, mappers, generators etc.)
    but just the diff part which is very important for performance.

    Returns a tuple containing:
        (filtered chain for delta/retry processing, dependencies, found records flag)
    """

    catalog = dc.session.catalog
    dc._query.resolve_all_listings()

    # Check if dataset exists
    try:
        dataset = catalog.get_dataset(
            name,
            namespace_name=namespace_name,
            project_name=project_name,
            versions=None,
            include_incomplete=False,
        )
        latest_version = dataset.latest_version
    except (DatasetNotFoundError, DatasetVersionNotFoundError):
        # First creation of result dataset (dataset missing, or only tombstones left)
        return None, None, True

    delta_sources = dc._query.delta_sources()

    dependencies: list[DatasetDependency | None] = [
        d
        for d in catalog.get_dataset_dependencies(
            name,
            latest_version,
            namespace_name=namespace_name,
            project_name=project_name,
            indirect=False,
        )
        if d is None or not d.removed
    ]
    latest_dataset = datachain.read_dataset(
        name,
        namespace=namespace_name,
        project=project_name,
        version=latest_version,
    )
    if not delta_sources or _static_input_changed(dc._query, dependencies):
        return None, None, True
    updated_versions: dict[tuple[str, str, str], str] = {}
    result_key = _get_shared_result_key(delta_sources)

    source_replacements: list[tuple[DatasetQuery, DataChain]] = []
    source_versions_changed = False
    changed_keys: list[tuple[DeltaSpec, DataChain | None]] = []
    for source in delta_sources:
        target_source_spec = source.delta_spec
        assert target_source_spec is not None
        target_source_step = source.starting_step
        assert target_source_step is not None
        target_source_dataset = target_source_step.dataset

        (
            source_replay_input,
            source_keys,
            source_ds_version,
            source_ds_latest_version,
        ) = _build_source_replay_input(
            target_source_query=source,
            target_source_spec=target_source_spec,
            result_dataset=latest_dataset,
            result_dependencies=dependencies,
        )
        if source_ds_latest_version is None:
            return None, None, True

        updated_versions[_dataset_identity(target_source_dataset)] = (
            source_ds_latest_version
        )
        source_versions_changed |= source_ds_version != source_ds_latest_version
        changed_keys.append((target_source_spec, source_keys))
        if source_replay_input is not None:
            source_replacements.append((source, source_replay_input))

    if all(keys is None for _, keys in changed_keys):
        return None, None, False

    # Partial replay is safe only when every participating source changes the same keys.
    if not _can_replay_changed_keys(
        dc._query, result_key
    ) or not _changed_keys_are_aligned(changed_keys):
        return None, None, True

    spec, keys = changed_keys[0]
    assert keys is not None
    retained_chain = latest_dataset.subtract(keys, on=result_key, right_on=spec.on)

    replayed_delta_chain = _rewrite_query_for_delta_replay(
        source_replacements,
        dc,
    ).persist()
    if (
        not source_versions_changed
        and replayed_delta_chain.empty
        and retained_chain.count() == latest_dataset.count()
    ):
        return None, None, False
    normalized_dependencies = _normalize_dependencies(dependencies, updated_versions)

    retained_chain = retained_chain.subtract(replayed_delta_chain, on=result_key)
    result_chain = _safe_union(
        retained_chain,
        replayed_delta_chain,
        context="merging the delta output with the existing dataset version",
    )
    return result_chain, normalized_dependencies, True
