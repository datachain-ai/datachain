from collections.abc import Sequence
from copy import copy
from functools import wraps
from typing import TYPE_CHECKING, TypeVar

import datachain
from datachain.dataset import DatasetDependency, DatasetRecord
from datachain.error import DatasetNotFoundError, SchemaDriftError
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


def delta_disabled(
    method: "Callable[Concatenate[T, P], T]",
) -> "Callable[Concatenate[T, P], T]":
    """
    Decorator for disabling DataChain methods (e.g `.agg()` or `.union()`) to
    work with delta updates. It throws `NotImplementedError` if chain on which
    method is called is marked as delta.
    """

    @wraps(method)
    def _inner(self: T, *args: "P.args", **kwargs: "P.kwargs") -> T:
        if self.delta and not self.delta_unsafe:
            raise NotImplementedError(
                f"Cannot use {method.__name__} with delta datasets - may cause"
                " inconsistency. Use delta_unsafe flag to allow this operation."
            )
        return method(self, *args, **kwargs)

    return _inner


def _replay_source(
    source_chain: "DataChain",
    original_chain: "DataChain",
    source_query: "DatasetQuery",
) -> "DataChain":
    replay = source_chain.clone()
    replay._query = original_chain._query.replace_source(
        source_query, source_chain._query
    )
    replay.signals_schema = original_chain.signals_schema
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


def _get_delta_chain(
    source_ds_name: str,
    source_ds_project: Project,
    source_ds_version: str,
    source_ds_latest_version: str,
    on: str | Sequence[str],
    compare: str | Sequence[str] | None = None,
) -> "DataChain":
    """Get delta chain for processing changes between versions."""
    source_dc = datachain.read_dataset(
        source_ds_name,
        namespace=source_ds_project.namespace.name,
        project=source_ds_project.name,
        version=source_ds_version,
    )
    source_dc_latest = datachain.read_dataset(
        source_ds_name,
        namespace=source_ds_project.namespace.name,
        project=source_ds_project.name,
        version=source_ds_latest_version,
    )

    # Calculate diff between source versions
    return source_dc_latest.diff(source_dc, on=on, compare=compare, deleted=False)


def _get_retry_chain(
    name: str,
    namespace_name: str,
    project_name: str,
    latest_version: str,
    source_ds_name: str,
    source_ds_project: Project,
    source_ds_version: str,
    on: str | Sequence[str],
    right_on: str | Sequence[str] | None,
    delta_retry: bool | str | None,
    diff_chain: "DataChain",
) -> "DataChain | None":
    """Get retry chain for processing error records and missing records."""
    # Import here to avoid circular import
    from datachain.lib.dc import C

    retry_chain = None

    # Read the latest version of the result dataset for retry logic
    result_dataset = datachain.read_dataset(
        name,
        namespace=namespace_name,
        project=project_name,
        version=latest_version,
    )
    source_dc = datachain.read_dataset(
        source_ds_name,
        namespace=source_ds_project.namespace.name,
        project=source_ds_project.name,
        version=source_ds_version,
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
        retry_chain.diff(
            diff_chain, on=on, added=True, same=True, modified=False, deleted=False
        ).distinct(*on)
        if retry_chain
        else None
    )


def _get_source_info(
    source_ds: DatasetRecord,
    name: str,
    namespace_name: str,
    project_name: str,
    latest_version: str,
    catalog,
) -> tuple[
    str | None,
    Project | None,
    str | None,
    str | None,
]:
    """Get source dataset information and dependencies.

    Returns:
        Tuple of (source_name, source_project, source_version, source_latest_version).
        Returns (None, None, None, None) if source dataset was removed.
    """
    dependencies = catalog.get_dataset_dependencies(
        name,
        latest_version,
        namespace_name=namespace_name,
        project_name=project_name,
        indirect=False,
    )

    source_ds_dep = next(
        (
            d
            for d in dependencies
            if d
            and d.name == source_ds.name
            and d.project == source_ds.project.name
            and d.namespace == source_ds.project.namespace.name
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


def _get_processing_chain(
    name: str,
    namespace_name: str,
    project_name: str,
    latest_version: str,
    source_query: "DatasetQuery",
    spec: "DeltaSpec",
) -> tuple["DataChain", str] | None:
    if source_query.starting_step is None:
        raise RuntimeError("Delta source query must be resolved before replay")

    source_ds_name, source_ds_project, source_ds_version, source_ds_latest_version = (
        _get_source_info(
            source_query.starting_step.dataset,
            name,
            namespace_name,
            project_name,
            latest_version,
            source_query.catalog,
        )
    )

    if source_ds_name is None:
        return None

    assert source_ds_project is not None
    assert source_ds_version is not None
    assert source_ds_latest_version is not None

    diff_chain = _get_delta_chain(
        source_ds_name,
        source_ds_project,
        source_ds_version,
        source_ds_latest_version,
        spec.on,
        spec.compare,
    )

    retry_chain = None
    if spec.delta_retry:
        retry_chain = _get_retry_chain(
            name,
            namespace_name,
            project_name,
            latest_version,
            source_ds_name,
            source_ds_project,
            source_ds_version,
            spec.on,
            spec.right_on,
            spec.delta_retry,
            diff_chain,
        )

    if retry_chain is not None:
        processing_chain = _safe_union(
            diff_chain,
            retry_chain,
            context="combining retry records with delta changes",
        )
    else:
        processing_chain = diff_chain

    if processing_chain.empty:
        return None

    return processing_chain, source_ds_latest_version


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
    dc._query.apply_listing_pre_step()

    # Check if dataset exists
    try:
        dataset = catalog.get_dataset(
            name,
            namespace_name=namespace_name,
            project_name=project_name,
            include_incomplete=False,
        )
        latest_version = dataset.latest_version
    except DatasetNotFoundError:
        # First creation of result dataset
        return None, None, True

    delta_sources = dc._query.delta_sources()
    if not delta_sources:
        return None, None, True

    dependencies = catalog.get_dataset_dependencies(
        name,
        latest_version,
        namespace_name=namespace_name,
        project_name=project_name,
        indirect=False,
    )
    updated_versions: dict[tuple[str, str, str], str] = {}
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

    processing_chain = None
    for source in delta_sources:
        spec = source.delta_spec
        assert spec is not None
        source_step = source.starting_step
        assert source_step is not None
        source_dataset = source_step.dataset

        source_info = _get_processing_chain(
            name,
            namespace_name,
            project_name,
            latest_version,
            source,
            spec,
        )
        if source_info is None:
            source_ds_name, source_ds_project, _, source_ds_latest_version = (
                _get_source_info(
                    source_dataset,
                    name,
                    namespace_name,
                    project_name,
                    latest_version,
                    catalog,
                )
            )
            if source_ds_name is None:
                return None, None, True
            assert source_ds_project is not None
            assert source_ds_latest_version is not None
            updated_versions[
                (
                    source_ds_project.namespace.name,
                    source_ds_project.name,
                    source_ds_name,
                )
            ] = source_ds_latest_version
            continue

        source_processing_chain, source_ds_latest_version = source_info
        updated_versions[
            (
                source_dataset.project.namespace.name,
                source_dataset.project.name,
                source_dataset.name,
            )
        ] = source_ds_latest_version

        replay_chain = _replay_source(source_processing_chain, dc, source).persist()
        if replay_chain.empty:
            continue

        if processing_chain is None:
            processing_chain = replay_chain
        else:
            processing_chain = _safe_union(
                processing_chain,
                replay_chain,
                context="combining replay results across delta sources",
            )

    if processing_chain is None:
        return None, None, False

    normalized_dependencies = _normalize_dependencies(dependencies, updated_versions)

    latest_dataset = datachain.read_dataset(
        name,
        namespace=namespace_name,
        project=project_name,
        version=latest_version,
    )
    result_key = next(iter(result_keys))
    diff_on: str | Sequence[str] = result_key[0] if len(result_key) == 1 else result_key
    compared_chain = latest_dataset.diff(
        processing_chain,
        on=diff_on,
        added=True,
        modified=False,
        deleted=False,
    )
    result_chain = _safe_union(
        compared_chain,
        processing_chain,
        context="merging the delta output with the existing dataset version",
    )
    return result_chain, normalized_dependencies, True
