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
    replay = original_query_chain.clone()
    replay._query = original_query_chain._query.replace_sources(
        {id(source): replacement._query for source, replacement in source_replacements}
    )
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


def _persist_replay_input(chain: "DataChain") -> "DataChain":
    from datachain.lib.dc import Sys
    from datachain.lib.signal_schema import SignalSchema

    schema = chain.signals_schema.clone_without_sys_signals()
    project = chain.session.catalog.metastore.get_project(
        chain.project_name, chain.namespace_name, create=True
    )
    return chain._evolve(
        query=chain._query.save(
            project=project,
            feature_schema=schema.serialize(),
            content_hash=chain._query.hash(),
        ),
        signal_schema=schema | SignalSchema({"sys": Sys}),
    )


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
) -> tuple["DataChain | None", "DataChain | None", str | None]:
    """Return replay input, persisted replacement keys and latest source version."""
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
        return None, None, None

    assert source_ds_project is not None
    assert source_ds_version is not None
    assert source_ds_latest_version is not None

    from datachain.diff import CompareStatus
    from datachain.lib.dc import C

    current_only = target_source_query.listing_fn is not None
    source = _read_source(
        source_ds_name, source_ds_project, source_ds_latest_version, current_only
    )
    versions_changed = source_ds_version != source_ds_latest_version
    diff_chain = None
    changes = None
    changed = []
    if versions_changed:
        previous = _read_source(
            source_ds_name, source_ds_project, source_ds_version, current_only
        )
        status_col = "dc_delta_status"
        names = {name.casefold() for name in source.signals_schema.values}
        while status_col.casefold() in names:
            status_col = "_" + status_col
        changes = _persist_replay_input(
            source.diff(
                previous,
                on=target_source_spec.on,
                compare=target_source_spec.compare,
                status_col=status_col,
            )
        )
        diff_chain = changes.filter(
            C(status_col) != CompareStatus.DELETED
        ).select_except(status_col)
        changed.append(changes)

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
            changes,
            current_only=current_only,
        )

    source_replay_input = diff_chain
    if retry_chain is not None:
        retry_chain = _persist_replay_input(retry_chain)
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
        source_replay_input if source_replay_input is not None else source.limit(0),
        _persist_keys(changed, target_source_spec.on),
        source_ds_latest_version,
    )


def delta_retry_update(
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
    if not delta_sources:
        return None, None, True

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
    updated_versions: dict[tuple[str, str, str], str] = {}
    result_key = _get_shared_result_key(delta_sources)

    source_replacements: list[tuple[DatasetQuery, DataChain]] = []
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
        changed_keys.append((target_source_spec, source_keys))
        assert source_replay_input is not None
        source_replacements.append((source, source_replay_input))

    if all(keys is None for _, keys in changed_keys):
        return None, None, False

    retained_chain = latest_dataset
    for spec, keys in changed_keys:
        if keys is not None:
            retained_chain = retained_chain.subtract(
                keys, on=result_key, right_on=spec.on
            )

    replayed_delta_chain = _rewrite_query_for_delta_replay(
        source_replacements,
        dc,
    ).persist()
    if replayed_delta_chain.empty and all(
        keys is None
        or keys.subtract(latest_dataset, on=spec.on, right_on=result_key).count()
        == keys.count()
        for spec, keys in changed_keys
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
