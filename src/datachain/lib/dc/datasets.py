from collections.abc import Sequence
from typing import TYPE_CHECKING, get_origin, get_type_hints

from datachain.error import (
    DatasetNotFoundError,
    DatasetVersionNotFoundError,
    ProjectNotFoundError,
)
from datachain.lib.dataset_info import DatasetInfo
from datachain.lib.projects import get as get_project
from datachain.lib.settings import Settings
from datachain.lib.signal_schema import SignalSchema
from datachain.query import Session
from datachain.query.dataset import DatasetQuery

from .utils import Sys, is_studio
from .values import read_values

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    from .datachain import DataChain

    P = ParamSpec("P")


def read_dataset(
    name: str,
    namespace: str | None = None,
    project: str | None = None,
    version: str | int | None = None,
    session: Session | None = None,
    settings: dict | None = None,
    delta: bool | None = False,
    delta_on: str | Sequence[str] | None = (
        "file.path",
        "file.etag",
        "file.version",
    ),
    delta_result_on: str | Sequence[str] | None = None,
    delta_compare: str | Sequence[str] | None = None,
    delta_retry: bool | str | None = None,
    delta_unsafe: bool = False,
    update: bool = False,
) -> "DataChain":
    """Get data from a saved Dataset. It returns the chain itself.
    If dataset or version is not found locally, it will try to pull it from Studio.

    Parameters:
        name: The dataset name, which can be a fully qualified name including the
            namespace and project. Alternatively, it can be a regular name, in which
            case the explicitly defined namespace and project will be used if they are
            set; otherwise, default values will be applied.
        namespace: optional name of namespace in which dataset to read is created
        project: optional name of project in which dataset to read is created
        version: dataset version. Supports:
            - Exact version strings: "1.2.3"
            - Legacy integer versions: 1, 2, 3 (finds latest major version)
            - Version specifiers (PEP 440): ">=1.0.0,<2.0.0", "~=1.4.2", "==1.2.*", etc.
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        delta: If True, only process new or changed files instead of reprocessing
            everything. This saves time by skipping files that were already processed in
            previous versions. The optimization is working when a new version of the
            dataset is created.
            Default is False.
        delta_on: Field(s) that uniquely identify each record in the source data.
            Used to detect which records are new or changed.
            Default is ("file.path", "file.etag", "file.version").
        delta_result_on: Field(s) in the result dataset that match `delta_on` fields.
            Only needed if you rename the identifying fields during processing.
            Default is None.
        delta_compare: Field(s) used to detect if a record has changed.
            If not specified, all fields except `delta_on` fields are used.
            Default is None.
        delta_retry: Controls retry behavior for failed records:
            - String (field name): Reprocess records where this field is not empty
              (error mode)
            - True: Reprocess records missing from the result dataset (missing mode)
            - None: No retry processing (default)
        update: If True always checks for newer versions available on Studio, even if
            some version of the dataset exists locally already. If False (default), it
            will only fetch the dataset from Studio if it is not found locally.
        delta_unsafe: Allow restricted ops in delta: merge, agg, union, group_by,
            distinct.


    Example:
        ```py
        import datachain as dc
        chain = dc.read_dataset("my_cats")
        ```

        ```py
        import datachain as dc
        chain = dc.read_dataset("dev.animals.my_cats")
        ```

        ```py
        chain = dc.read_dataset("my_cats", version="1.0.0")
        ```

        ```py
        # Using version specifiers (PEP 440)
        chain = dc.read_dataset("my_cats", version=">=1.0.0,<2.0.0")
        ```

        ```py
        # Legacy integer version support (finds latest in major version)
        chain = dc.read_dataset("my_cats", version=1)  # Latest 1.x.x version
        ```

        ```py
        # Always check for newer versions matching a version specifier from Studio
        chain = dc.read_dataset("my_cats", version=">=1.0.0", update=True)
        ```

        ```py
        session = Session.get(client_config={"aws_endpoint_url": "<minio-url>"})
        settings = {
            "cache": True,
            "parallel": 4,
            "workers": 4,
            "min_task_size": 1000,
            "prefetch": 10,
        }
        chain = dc.read_dataset(
            name="my_cats",
            version="1.0.0",
            session=session,
            settings=settings,
        )
        ```
    """
    from datachain.telemetry import telemetry

    from .datachain import DataChain

    telemetry.send_event_once("class", "datachain_init", name=name, version=version)

    session = Session.get(session)
    catalog = session.catalog

    namespace_name, project_name, name = catalog.get_full_dataset_name(
        name,
        project_name=project,
        namespace_name=namespace,
    )

    if version is not None:
        dataset = session.catalog.get_dataset_with_remote_fallback(
            name,
            namespace_name,
            project_name,
            update=update,
            include_incomplete=False,  # Never include incomplete datasets
        )

        # Convert legacy integer versions to version specifiers
        # For backward compatibility we still allow users to put version as integer
        # in which case we convert it to a version specifier that finds the latest
        # version where major part is equal to that input version.
        # For example if user sets version=2, we convert it to ">=2.0.0,<3.0.0"
        # which will find something like 2.4.3 (assuming 2.4.3 is the biggest among
        # all 2.* dataset versions)
        if isinstance(version, int):
            version_spec = f">={version}.0.0,<{version + 1}.0.0"
        else:
            version_spec = str(version)

        from packaging.specifiers import InvalidSpecifier, SpecifierSet

        try:
            # Try to parse as version specifier
            SpecifierSet(version_spec)
            # If it's a valid specifier set, find the latest compatible version
            latest_compatible = dataset.latest_compatible_version(version_spec)
            if not latest_compatible:
                raise DatasetVersionNotFoundError(
                    f"No dataset {name} version matching specifier {version_spec}"
                )
            version = latest_compatible
        except InvalidSpecifier:
            # If not a valid specifier, treat as exact version string
            # This handles cases like "1.2.3" which are exact versions, not specifiers
            pass

    if settings:
        _settings = Settings(**settings)
    else:
        _settings = Settings()

    query = DatasetQuery(
        name=name,
        project_name=project_name,
        namespace_name=namespace_name,
        version=version,  #  type: ignore[arg-type]
        session=session,
        update=update,
    )

    signals_schema = SignalSchema({"sys": Sys})
    if query.feature_schema:
        signals_schema |= SignalSchema.deserialize(query.feature_schema)
    else:
        signals_schema |= SignalSchema.from_column_types(query.column_types or {})

    if delta:
        signals_schema = signals_schema.clone_without_sys_signals()

    chain = DataChain(query, _settings, signals_schema)

    if delta:
        chain = chain._as_delta(
            on=delta_on,
            right_on=delta_result_on,
            compare=delta_compare,
            delta_retry=delta_retry,
            delta_unsafe=delta_unsafe,
        )

    return chain


def datasets(
    session: Session | None = None,
    settings: dict | None = None,
    in_memory: bool = False,
    column: str | None = None,
    include_listing: bool = False,
    studio: bool = False,
    attrs: list[str] | None = None,
) -> "DataChain":
    """Generate chain with list of registered datasets.

    Args:
        session: Optional session instance. If not provided, uses default session.
        settings: Optional dictionary of settings to configure the chain.
        in_memory: If True, creates an in-memory session. Defaults to False.
        column: Name of the output column in the chain. Defaults to None which
            means no top level column will be created.
        include_listing: If True, includes listing datasets. Defaults to False.
        studio: If True, returns datasets from Studio only,
            otherwise returns all local datasets. Defaults to False.
        attrs: Optional list of attributes to filter datasets on. It can be just
            attribute without value e.g "NLP", or attribute with value
            e.g "location=US". Attribute with value can also accept "*" to target
            all that have specific name e.g "location=*"

    Returns:
        DataChain: A new DataChain instance containing dataset information.

    Example:
        ```py
        import datachain as dc

        chain = dc.datasets(column="dataset")
        for ds in chain.to_iter("dataset"):
            print(f"{ds.name}@v{ds.version}")
        ```
    """

    session = Session.get(session, in_memory=in_memory)
    catalog = session.catalog

    datasets_values = [
        DatasetInfo.from_models(d, v, j)
        for d, v, j in catalog.list_datasets_versions(
            include_listing=include_listing, studio=studio
        )
    ]
    datasets_values = [d for d in datasets_values if not d.is_temp]

    if attrs:
        for attr in attrs:
            datasets_values = [d for d in datasets_values if d.has_attr(attr)]

    if not column:
        # flattening dataset fields
        schema = {
            k: get_origin(v) if get_origin(v) is dict else v
            for k, v in get_type_hints(DatasetInfo).items()
            if k in DatasetInfo.model_fields
        }
        data = {k: [] for k in DatasetInfo.model_fields}  # type: ignore[var-annotated]
        for d in [d.model_dump() for d in datasets_values]:
            for field, value in d.items():
                data[field].append(value)

        return read_values(
            session=session,
            settings=settings,
            in_memory=in_memory,
            output=schema,
            **data,  # type: ignore[arg-type]
        )

    return read_values(
        session=session,
        settings=settings,
        in_memory=in_memory,
        output={column: DatasetInfo},
        **{column: datasets_values},  # type: ignore[arg-type]
    )


def delete_dataset(
    name: str,
    namespace: str | None = None,
    project: str | None = None,
    version: str | None = None,
    force: bool | None = False,
    studio: bool | None = False,
    session: Session | None = None,
    in_memory: bool = False,
) -> None:
    """Removes specific dataset version or all dataset versions, depending on
    a force flag.

    Args:
        name: The dataset name, which can be a fully qualified name including the
            namespace and project. Alternatively, it can be a regular name, in which
            case the explicitly defined namespace and project will be used if they are
            set; otherwise, default values will be applied.
        namespace: optional name of namespace in which dataset to delete is created
        project: optional name of project in which dataset to delete is created
        version: Optional dataset version
        force: If true, all datasets versions will be removed. Defaults to False.
        studio: If True, removes dataset from Studio only, otherwise removes local
            dataset. Defaults to False.
        session: Optional session instance. If not provided, uses default session.
        in_memory: If True, creates an in-memory session. Defaults to False.

    Returns: None

    Example:
        ```py
        import datachain as dc
        dc.delete_dataset("cats")
        ```

        ```py
        import datachain as dc
        dc.delete_dataset("cats", version="1.0.0")
        ```
    """
    from datachain.studio import remove_studio_dataset

    session = Session.get(session, in_memory=in_memory)
    catalog = session.catalog

    namespace_name, project_name, name = catalog.get_full_dataset_name(
        name,
        project_name=project,
        namespace_name=namespace,
    )

    if not is_studio() and studio:
        return remove_studio_dataset(
            None, name, namespace_name, project_name, version=version, force=force
        )

    try:
        ds_project = get_project(project_name, namespace_name, session=session)
    except ProjectNotFoundError:
        raise DatasetNotFoundError(
            f"Dataset {name} not found in namespace {namespace_name} and project",
            f" {project_name}",
        ) from None

    if not force:
        version = (
            version
            or catalog.get_dataset(
                name,
                namespace_name=ds_project.namespace.name,
                project_name=ds_project.name,
                include_incomplete=False,
            ).latest_version
        )
    else:
        version = None
    catalog.remove_dataset(name, ds_project, version=version, force=force)


def move_dataset(
    src: str,
    dest: str,
    session: Session | None = None,
    in_memory: bool = False,
) -> None:
    """Moves an entire dataset between namespaces and projects.

    Args:
        src: The source dataset name. This can be a fully qualified name that includes
            the namespace and project, or a regular name. If a regular name is used,
            default values will be applied. The source dataset will no longer exist
            after the move.
        dest: The destination dataset name. This can also be a fully qualified
            name with a namespace and project, or just a regular name (default values
            will be used in that case). The original dataset will be moved here.
        session: An optional session instance. If not provided, the default session
            will be used.
        in_memory: If True, creates an in-memory session. Defaults to False.

    Returns:
        None

    Examples:
        ```python
        import datachain as dc
        dc.move_dataset("cats", "new_cats")
        ```

        ```python
        import datachain as dc
        dc.move_dataset("dev.animals.cats", "prod.animals.cats")
        ```
    """
    session = Session.get(session, in_memory=in_memory)
    catalog = session.catalog

    namespace, project, name = catalog.get_full_dataset_name(src)
    dest_namespace, dest_project, dest_name = catalog.get_full_dataset_name(dest)

    dataset = catalog.get_dataset(
        name, namespace_name=namespace, project_name=project, include_incomplete=False
    )

    catalog.update_dataset(
        dataset,
        name=dest_name,
        project_id=catalog.metastore.get_project(
            dest_project,
            dest_namespace,
            create=is_studio(),
        ).id,
    )
