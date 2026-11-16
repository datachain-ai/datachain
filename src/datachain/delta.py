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


def delta_key_changed_error(key: str, method: str) -> Exception:
    from datachain.lib.utils import DataChainParamsError

    return DataChainParamsError(
        f"{method} can't change delta key {key!r}: write the new value to another"
        " column, or set delta_unsafe=True"
    )


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


def _build_source_retry_chain(
    result_dataset: "DataChain",
    source: "DataChain",
    spec: "DeltaSpec",
    changes: "DataChain | None",
) -> "DataChain | None":
    """Retry failed or missing records that the source diff does not replay."""
    from datachain.lib.dc import C

    on, right_on = spec.on, spec.right_on
    if isinstance(spec.delta_retry, str):
        errors = result_dataset.filter(C(spec.delta_retry) != "")
        # A semi-join via subtract preserves NULL-key matching.
        without_errors = source.subtract(errors, on=on, right_on=right_on)
        retry = source.subtract(without_errors, on=on)
    elif spec.delta_retry is True:
        retry = source.subtract(result_dataset, on=on, right_on=right_on)
    else:
        return None
    if changes is not None:
        retry = retry.subtract(changes, on=on)
    return retry


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


def _validate_result_keys(dc: "DataChain", sources: Sequence["DatasetQuery"]) -> None:
    from datachain.lib.dc.datachain import _paired_key_columns
    from datachain.lib.signal_schema import SignalResolvingError, SignalSchema
    from datachain.lib.utils import DataChainParamsError

    for source in sources:
        spec = source.delta_spec
        assert spec is not None
        on, right_on = spec.on, spec.right_on or spec.on
        source_schema = (
            SignalSchema.deserialize(source.feature_schema)
            if source.feature_schema
            else SignalSchema.from_column_types(source.column_types or {})
        )
        try:
            _paired_key_columns(
                source_schema,
                [on] if isinstance(on, str) else on,
                dc.signals_schema,
                [right_on] if isinstance(right_on, str) else right_on,
            )
        except SignalResolvingError as exc:
            raise DataChainParamsError(
                "Delta keys must be present in the source (delta_on) and result "
                "(delta_result_on). Keep the source key when processing records."
            ) from exc


def _build_source_replay_input(
    source_query: "DatasetQuery",
    spec: "DeltaSpec",
    result_dataset: "DataChain",
    result_dependencies: list[DatasetDependency | None],
    retry_salt: str = "",
) -> tuple["DataChain", list["DataChain"], str] | None:
    """Return replay rows, persisted changes and the latest source version.

    Return None if the source was removed from the result's dependencies.
    """
    assert source_query.starting_step is not None
    name, project, version, latest_version = _get_source_info(
        source_query.starting_step.dataset, result_dependencies, source_query.catalog
    )
    if name is None:
        return None
    assert project is not None
    assert version is not None
    assert latest_version is not None

    from datachain.diff import CompareStatus
    from datachain.lib.dc import C

    current_only = source_query.listing_fn is not None
    source = _read_source(name, project, latest_version, current_only)
    changes = replay = None
    if version != latest_version:
        previous = _read_source(name, project, version, current_only)
        status_col = "dc_delta_status"
        names = {name.casefold() for name in source.signals_schema.values}
        while status_col.casefold() in names:
            status_col = "_" + status_col
        changes = source.diff(
            previous,
            on=spec.on,
            compare=spec.compare,
            status_col=status_col,
        )._persist(preserve_hash=True)
        replay = changes.filter(C(status_col) != CompareStatus.DELETED).select_except(
            status_col
        )
    changed = [changes] if changes is not None and not changes.empty else []
    retry = _build_source_retry_chain(result_dataset, source, spec, changes)
    if retry is not None:
        retry = retry._persist(preserve_hash=True, hash_salt=retry_salt)
        if not retry.empty:
            changed.append(retry)
            replay = retry if replay is None else replay.union(retry)

    return (
        replay if replay is not None else source.limit(0),
        changed,
        latest_version,
    )


def delta_retry_update(
    dc: "DataChain",
    namespace_name: str,
    project_name: str,
    name: str,
    *,
    retry_salt: str = "",
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
    delta_sources = dc._query.delta_sources()
    if not delta_sources:
        return None, None, True

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
        _validate_result_keys(dc, delta_sources)
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

    replacements: dict[int, DatasetQuery] = {}
    prepared: dict[
        tuple[tuple[str, str, str], str, bool],
        tuple[DataChain, list[DataChain], str],
    ] = {}
    changed_rows: list[tuple[DeltaSpec, DataChain]] = []
    for source in delta_sources:
        spec = source.delta_spec
        assert spec is not None
        assert source.starting_step is not None
        identity = _dataset_identity(source.starting_step.dataset)
        key = (
            identity,
            spec.hash(),
            source.listing_fn is not None,
        )
        if key not in prepared:
            replay_input = _build_source_replay_input(
                source, spec, latest_dataset, dependencies, retry_salt
            )
            if replay_input is None:
                return None, None, True
            prepared[key] = replay_input
            changed_rows.extend((spec, rows) for rows in replay_input[1])
        replay_rows, _, latest_source_version = prepared[key]
        updated_versions[identity] = latest_source_version
        replacements[id(source)] = replay_rows._query

    if not changed_rows:
        return None, None, False

    retained_chain = latest_dataset
    for spec, rows in changed_rows:
        retained_chain = retained_chain.subtract(rows, on=result_key, right_on=spec.on)

    replayed_delta_chain = dc._evolve(
        query=dc._query.replace_sources(replacements)
    ).persist()
    if replayed_delta_chain.empty and retained_chain.count() == latest_dataset.count():
        return None, None, False
    normalized_dependencies = _normalize_dependencies(dependencies, updated_versions)

    retained_chain = retained_chain.subtract(replayed_delta_chain, on=result_key)
    result_chain = _safe_union(
        retained_chain,
        replayed_delta_chain,
        context="merging the delta output with the existing dataset version",
    )
    return result_chain, normalized_dependencies, True
