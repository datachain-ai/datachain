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
        if self.delta and not self._delta_unsafe:
            raise NotImplementedError(
                f"Cannot use {method.__name__} with delta datasets - may cause"
                " inconsistency. Use delta_unsafe flag to allow this operation."
            )
        return method(self, *args, **kwargs)

    return _inner


def _append_steps(dc: "DataChain", other: "DataChain"):
    """Returns cloned chain with appended steps from other chain.
    Steps are all those modification methods applied like filters, mappers etc.
    """
    dc = dc.clone()
    dc._query.steps += other._query.steps.copy()
    dc.signals_schema = other.signals_schema
    return dc


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
    list[DatasetDependency] | None,
]:
    """Get source dataset information and dependencies.

    Returns:
        Tuple of (source_name, source_version, source_latest_version, dependencies)
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
        (d for d in dependencies if d and d.name == source_ds.name), None
    )
    if not source_ds_dep:
        # Starting dataset was removed, back off to normal dataset creation
        return None, None, None, None, None

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
        dependencies,
    )


def delta_retry_update(
    dc: "DataChain",
    namespace_name: str,
    project_name: str,
    name: str,
    on: str | Sequence[str],
    right_on: str | Sequence[str] | None = None,
    compare: str | Sequence[str] | None = None,
    delta_retry: bool | str | None = None,
) -> tuple["DataChain | None", list[DatasetDependency] | None, bool]:
    """
    Creates new chain that consists of the last version of current delta dataset
    plus diff from the source with all needed modifications.
    This way we don't need to re-calculate the whole chain from the source again
    (apply all the DataChain methods like filters, mappers, generators etc.)
    but just the diff part which is very important for performance.

    Note that currently delta update works only if there is only one direct
    dependency.

    Additionally supports retry functionality to filter records that either:
    1. Have a non-None value in the field specified by delta_retry (when it's a string)
    2. Exist in the source dataset but are missing in the result dataset
       (when delta_retry=True)

    Parameters:
        dc: The DataChain to filter for records that need reprocessing
        name: Name of the destination dataset
        on: Field(s) in source dataset that uniquely identify records
        right_on: Corresponding field(s) in result dataset if they differ from
                  source
        compare: Field(s) used to check if the same row has been modified
        delta_retry: If string, field in result dataset that indicates an error
                    when not None. If True, include records missing from result dataset.
                    If False/None, no retry functionality.

    Returns:
        A tuple containing (filtered chain for delta/retry processing,
                          dependencies, found records flag)
    """

    catalog = dc.session.catalog
    # project = catalog.metastore.get_project(project_name, namespace_name)
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

    # Initialize variables
    diff_chain = None
    dependencies = None
    retry_chain = None
    processing_chain = None

    (
        source_ds_name,
        source_ds_project,
        source_ds_version,
        source_ds_latest_version,
        dependencies,
    ) = _get_source_info(
        dc._query.starting_step.dataset,  # type: ignore[union-attr]
        name,
        namespace_name,
        project_name,
        latest_version,
        catalog,
    )

    # If source_ds_name is None, starting dataset was removed
    if source_ds_name is None:
        return None, None, True

    assert source_ds_project
    assert source_ds_version
    assert source_ds_latest_version

    diff_chain = _get_delta_chain(
        source_ds_name,
        source_ds_project,
        source_ds_version,
        source_ds_latest_version,
        on,
        compare,
    )

    # Filter out removed dep
    if dependencies:
        dependencies = copy(dependencies)
        dependencies = [d for d in dependencies if d is not None]
        source_ds_dep = next(d for d in dependencies if d.name == source_ds_name)
        # Update to latest version
        source_ds_dep.version = source_ds_latest_version  # type: ignore[union-attr]

    # Handle retry functionality if enabled
    if delta_retry:
        retry_chain = _get_retry_chain(
            name,
            namespace_name,
            project_name,
            latest_version,
            source_ds_name,
            source_ds_project,
            source_ds_version,
            on,
            right_on,
            delta_retry,
            diff_chain,
        )

    # Combine delta and retry chains
    if retry_chain is not None:
        processing_chain = _safe_union(
            diff_chain,
            retry_chain,
            context="combining retry records with delta changes",
        )
    else:
        processing_chain = diff_chain

    # Apply all the steps from the original chain to processing_chain
    processing_chain = _append_steps(processing_chain, dc).persist()

    # Check if chain becomes empty after applying steps
    if processing_chain is None or (processing_chain and processing_chain.empty):
        return None, None, False

    latest_dataset = datachain.read_dataset(
        name,
        namespace=namespace_name,
        project=project_name,
        version=latest_version,
    )
    compared_chain = latest_dataset.diff(
        processing_chain,
        on=right_on or on,
        added=True,
        modified=False,
        deleted=False,
    )
    result_chain = _safe_union(
        compared_chain,
        processing_chain,
        context="merging the delta output with the existing dataset version",
    )
    return result_chain, dependencies, True
