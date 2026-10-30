from collections.abc import Sequence
from copy import copy
from functools import wraps
from typing import TYPE_CHECKING, TypeVar

from sqlalchemy.sql import operators, visitors
from sqlalchemy.sql.elements import (
    BinaryExpression,
    BooleanClauseList,
    ColumnClause,
    ColumnElement,
    Over,
)

import datachain
from datachain.dataset import DatasetDependency, DatasetRecord
from datachain.error import (
    DatasetNotFoundError,
    DatasetVersionNotFoundError,
    SchemaDriftError,
)
from datachain.project import Project
from datachain.query.dataset import (
    SQLDistinct,
    SQLGroupBy,
    SQLJoin,
    SQLLimit,
    SQLMutate,
    SQLOffset,
    SQLSelect,
    Subtract,
    UDFStep,
    UnionSchemaMismatchError,
)

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
    return source_dc_latest.diff(source_dc, on=on, compare=compare, deleted=False)


def _build_source_retry_chain(
    result_dataset: "DataChain",
    source_ds_name: str,
    source_ds_project: Project,
    source_ds_version: str,
    on: str | Sequence[str],
    right_on: str | Sequence[str] | None,
    delta_retry: bool | str | None,
    diff_chain: "DataChain",
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
        error_source_records = source_dc.merge(
            error_records, on=on, right_on=right_on, inner=True
        ).select(
            *list(source_dc.signals_schema.clone_without_sys_signals().values.keys())
        )
        retry_chain = error_source_records

    # Handle missing records if delta_retry is True
    elif delta_retry is True:
        missing_records = source_dc.subtract(result_dataset, on=on, right_on=right_on)
        retry_chain = missing_records

    # Subtract also diff chain since some items might be picked
    # up by `delta=True` itself (e.g. records got modified AND are missing in the
    # result dataset atm)
    on = [on] if isinstance(on, str) else on

    return (
        retry_chain.subtract(diff_chain, on=on).distinct(*on) if retry_chain else None
    )


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
) -> str | Sequence[str]:
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

    result_key = next(iter(result_keys))
    return result_key[0] if len(result_key) == 1 else result_key


def _subtract_source_keys(
    result: "DataChain",
    source: "DataChain",
    on: str | Sequence[str],
    right_on: str | Sequence[str],
) -> "DataChain":
    def columns(schema: "SignalSchema", field: str) -> dict[str, str]:
        prefix = field.replace(".", "__")
        return {
            str(column).removeprefix(prefix): str(column)
            for column in schema.resolve(field).db_signals(include_sentinels=False)
        }

    result_fields = [on] if isinstance(on, str) else on
    source_fields = [right_on] if isinstance(right_on, str) else right_on
    signals: list[tuple[str, str]] = []
    for result_field, source_field in zip(result_fields, source_fields, strict=True):
        left = columns(result.signals_schema, result_field)
        right = columns(source.signals_schema, source_field)
        if left.keys() != right.keys():
            raise SchemaDriftError(
                f"Delta key fields do not match: {result_field!r} and {source_field!r}"
            )
        signals.extend((column, right[name]) for name, column in left.items())

    return result._evolve(query=result._query.subtract(source._query, signals))


def _join_replays_key(step: SQLJoin, key: Sequence[str]) -> bool:
    if step.full or not step.query1.delta_sources():
        return False
    predicates = step.predicates
    if (
        isinstance(predicates, BooleanClauseList)
        and predicates.operator is operators.and_
    ):
        predicates = tuple(predicates.clauses)
    elif not isinstance(predicates, tuple):
        predicates = (predicates,)
    names = []
    for predicate in predicates:
        if isinstance(predicate, (str, ColumnClause)):
            names.append(predicate if isinstance(predicate, str) else predicate.name)
        elif (
            isinstance(predicate, BinaryExpression)
            and predicate.operator is operators.eq
            and isinstance(predicate.left, ColumnClause)
            and isinstance(predicate.right, ColumnClause)
            and predicate.left.name == predicate.right.name
            and predicate.left.table is not predicate.right.table
        ):
            names.append(predicate.left.name)
        else:
            return False
    return tuple(names) == tuple(field.replace(".", "__") for field in key)


def _can_replay_changed_keys(
    query: "DatasetQuery", result_key: str | Sequence[str]
) -> bool:
    key = (result_key,) if isinstance(result_key, str) else tuple(result_key)
    for step in query.steps:
        if (
            isinstance(step, (Subtract, SQLDistinct, SQLGroupBy, SQLLimit, SQLOffset))
            or (isinstance(step, UDFStep) and step.is_aggregator)
            or (isinstance(step, SQLJoin) and not _join_replays_key(step, key))
        ):
            return False
        if isinstance(step, (SQLSelect, SQLMutate)) and any(
            isinstance(element, Over)
            for expression in step.parse_cols(step.args)
            if isinstance(expression, ColumnElement)
            for element in visitors.iterate(expression)
        ):
            return False
        if any(
            not _can_replay_changed_keys(child, result_key)
            for child in step.sub_queries
        ):
            return False
    return True


def _changed_keys_are_aligned(
    changed_keys: Sequence[tuple["DeltaSpec", "DataChain | None"]],
) -> bool:
    if len(changed_keys) < 2:
        return True
    first_spec, first_keys = changed_keys[0]
    if first_keys is None:
        return False
    for spec, keys in changed_keys[1:]:
        if (
            keys is None
            or not _subtract_source_keys(first_keys, keys, first_spec.on, spec.on).empty
            or not _subtract_source_keys(keys, first_keys, spec.on, first_spec.on).empty
        ):
            return False
    return True


def _static_input_changed(
    query: "DatasetQuery", dependencies: list[DatasetDependency | None]
) -> bool:
    if query.delta_spec is None and query.starting_step is not None:
        source = query.starting_step
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


def _build_source_replay_input(
    target_source_query: "DatasetQuery",
    target_source_spec: "DeltaSpec",
    result_dataset: "DataChain",
    result_dependencies: list[DatasetDependency | None],
) -> tuple["DataChain | None", "DataChain | None", str | None, bool]:
    """Return replay inputs, invalidated source rows (None when the source version
    is unchanged), the latest source version, and missing status."""
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
        return None, None, None, True

    assert source_ds_project is not None
    assert source_ds_version is not None
    assert source_ds_latest_version is not None

    current_only = target_source_query.listing_fn is not None
    diff_chain = _build_source_diff_chain(
        source_ds_name,
        source_ds_project,
        source_ds_version,
        source_ds_latest_version,
        target_source_spec.on,
        target_source_spec.compare,
        current_only=current_only,
    )

    invalidated_chain = None  # same version: nothing was modified or deleted
    if source_ds_version != source_ds_latest_version:
        invalidated_chain = _build_source_diff_chain(
            source_ds_name,
            source_ds_project,
            source_ds_latest_version,
            source_ds_version,
            target_source_spec.on,
            target_source_spec.compare,
            current_only=current_only,
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

    if retry_chain is not None:
        source_replay_input = _safe_union(
            diff_chain,
            retry_chain,
            context="combining retry records with delta changes",
        )
    else:
        source_replay_input = diff_chain

    return source_replay_input, invalidated_chain, source_ds_latest_version, False


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
    retained_chain = latest_dataset
    has_source_changes = False
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
            invalidated_chain,
            source_ds_latest_version,
            dependency_missing,
        ) = _build_source_replay_input(
            target_source_query=source,
            target_source_spec=target_source_spec,
            result_dataset=latest_dataset,
            result_dependencies=dependencies,
        )
        if dependency_missing:
            return None, None, True

        assert source_ds_latest_version is not None
        updated_versions[_dataset_identity(target_source_dataset)] = (
            source_ds_latest_version
        )

        assert source_replay_input is not None
        source_keys = None
        source_versions_changed |= invalidated_chain is not None
        if invalidated_chain is not None or target_source_spec.delta_retry:
            changed = [source_replay_input]
            if invalidated_chain is not None:
                changed.append(invalidated_chain.persist())
            # Invalidate by source keys before filters or generators can drop rows.
            for invalidated in (chain for chain in changed if not chain.empty):
                has_source_changes = True
                if len(delta_sources) > 1:
                    fields = target_source_spec.on
                    fields = [fields] if isinstance(fields, str) else fields
                    keys = invalidated._evolve(
                        signal_schema=invalidated.signals_schema.clone_without_sys_signals()
                    ).select(*fields)
                    source_keys = (
                        keys if source_keys is None else source_keys.union(keys)
                    )
                retained_chain = _subtract_source_keys(
                    retained_chain,
                    invalidated,
                    on=result_key,
                    right_on=target_source_spec.on,
                )
        changed_keys.append((target_source_spec, source_keys))
        source_replacements.append((source, source_replay_input))

    if not has_source_changes:
        return None, None, False

    # Partial replay is safe only when every participating source changes the same keys.
    if not _can_replay_changed_keys(
        dc._query, result_key
    ) or not _changed_keys_are_aligned(changed_keys):
        return None, None, True

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

    # Replay can include unchanged rows from non-delta branches.
    retained_chain = retained_chain.subtract(replayed_delta_chain, on=result_key)
    result_chain = _safe_union(
        retained_chain,
        replayed_delta_chain,
        context="merging the delta output with the existing dataset version",
    )
    return result_chain, normalized_dependencies, True
