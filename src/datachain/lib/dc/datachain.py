import copy
import hashlib
import logging
import os
import os.path
import sys
import warnings
from collections.abc import Callable, Iterator, Sequence
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    BinaryIO,
    ClassVar,
    Literal,
    TypeVar,
    cast,
    overload,
)

import sqlalchemy
from pydantic import BaseModel
from sqlalchemy.sql.elements import ColumnElement
from tqdm import tqdm

from datachain import json, semver
from datachain.dataset import DatasetRecord
from datachain.delta import delta_disabled
from datachain.error import (
    JobAncestryDepthExceededError,
    ProjectCreateNotAllowedError,
    ProjectNotFoundError,
)
from datachain.func import literal
from datachain.func.base import Function
from datachain.func.func import Func
from datachain.job import Job
from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.lib.data_model import (
    DataModel,
    DataType,
    DataValue,
    StandardType,
    dict_to_data_model,
)
from datachain.lib.file import EXPORT_FILES_MAX_THREADS, ArrowRow, File, FileExporter
from datachain.lib.file import ExportPlacement as FileExportPlacement
from datachain.lib.model_store import ModelStore
from datachain.lib.settings import Settings
from datachain.lib.signal_schema import (
    SignalResolvingError,
    SignalSchema,
)
from datachain.lib.udf import Aggregator, BatchMapper, Generator, Mapper, UDFBase
from datachain.lib.udf_signature import UdfSignature
from datachain.lib.utils import DataChainColumnError, DataChainParamsError
from datachain.project import Project
from datachain.query import Session
from datachain.query.dataset import (
    DatasetQuery,
    PartitionByType,
    RegenerateSystemColumns,
    UnionSchemaMismatchError,
)
from datachain.query.schema import DEFAULT_DELIMITER, Column
from datachain.sql.functions import path as pathfunc
from datachain.utils import batched_it, env2bool, inside_notebook, row_to_nested_dict

from .database import DEFAULT_DATABASE_BATCH_SIZE
from .utils import (
    DatasetMergeError,
    DatasetPrepareError,
    MergeColType,
    OutputType,
    Sys,
    _get_merge_error_str,
    _validate_merge_on,
    is_studio,
    resolve_columns,
)

logger = logging.getLogger("datachain")

C = Column

_T = TypeVar("_T")
UDFObjT = TypeVar("UDFObjT", bound=UDFBase)

DEFAULT_PARQUET_CHUNK_SIZE = 100_000

if TYPE_CHECKING:
    import sqlite3

    import pandas as pd
    from sqlalchemy.orm import Session as OrmSession
    from typing_extensions import ParamSpec, Self

    P = ParamSpec("P")

    ConnectionType = (
        str
        | sqlalchemy.engine.URL
        | sqlalchemy.engine.interfaces.Connectable
        | sqlalchemy.engine.Engine
        | sqlalchemy.engine.Connection
        | OrmSession
        | sqlite3.Connection
    )


T = TypeVar("T", bound="DataChain")


class DataChain:
    """DataChain - a data structure for batch data processing and evaluation.

    It represents a sequence of data manipulation steps such as reading data from
    storages, running AI or LLM models or calling external services API to validate or
    enrich data.

    Data in DataChain is presented as Python classes with arbitrary set of fields,
    including nested classes. The data classes have to inherit from `DataModel` class.
    The supported set of field types include: majority of the type supported by the
    underlyind library `Pydantic`.

    See Also:
        `read_storage("s3://my-bucket/my-dir/")` - reading unstructured
            data files from storages such as S3, gs or Azure ADLS.

        `DataChain.save("name")` - saving to a dataset.

        `read_dataset("name")` - reading from a dataset.

        `read_values(fib=[1, 2, 3, 5, 8])` - generating from values.

        `read_pandas(pd.DataFrame(...))` - generating from pandas.

        `read_json("file.json")` - generating from json.

        `read_csv("file.csv")` - generating from csv.

        `read_parquet("file.parquet")` - generating from parquet.

    Example:
        ```py
        import os

        from mistralai.client import MistralClient
        from mistralai.models.chat_completion import ChatMessage
        import datachain as dc

        PROMPT = (
            "Was this bot dialog successful? "
            "Describe the 'result' as 'Yes' or 'No' in a short JSON"
        )

        model = "mistral-large-latest"
        api_key = os.environ["MISTRAL_API_KEY"]

        chain = (
            dc.read_storage("gs://datachain-demo/chatbot-KiT/")
            .limit(5)
            .settings(cache=True, parallel=5)
            .map(
                mistral_response=lambda file: MistralClient(api_key=api_key)
                .chat(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        ChatMessage(role="user", content=f"{PROMPT}: {file.read()}")
                    ],
                )
                .choices[0]
                .message.content,
            )
            .persist()
        )

        try:
            print(chain.select("mistral_response").results())
        except Exception as e:
            print(f"do you have the right Mistral API key? {e}")
        ```
    """

    DEFAULT_FILE_RECORD: ClassVar[dict] = {
        "source": "",
        "path": "",
        "size": 0,
    }

    def __init__(
        self,
        query: DatasetQuery,
        settings: Settings,
        signal_schema: SignalSchema,
        setup: dict | None = None,
        _sys: bool = False,
    ) -> None:
        """Don't instantiate this directly, use one of the from_XXX constructors."""
        self._query = query
        self._settings = settings
        self.signals_schema = signal_schema
        self._setup: dict = setup or {}
        self._sys = _sys
        self._delta = False
        self._delta_unsafe = False
        self._delta_on: str | Sequence[str] | None = None
        self._delta_result_on: str | Sequence[str] | None = None
        self._delta_compare: str | Sequence[str] | None = None
        self._delta_retry: bool | str | None = None

    def __repr__(self) -> str:
        """Return a string representation of the chain."""
        classname = self.__class__.__name__
        if not self._effective_signals_schema.values:
            return f"Empty {classname}"

        import io

        file = io.StringIO()
        self.print_schema(file=file)
        return file.getvalue()

    def hash(self) -> str:
        """
        Calculates SHA hash of this chain. Hash calculation is fast and consistent.
        It takes into account all the steps added to the chain and their inputs.
        Order of the steps is important.
        """
        return self._query.hash()

    def _as_delta(
        self,
        on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        compare: str | Sequence[str] | None = None,
        delta_retry: bool | str | None = None,
        delta_unsafe: bool = False,
    ) -> "Self":
        """Marks this chain as delta, which means special delta process will be
        called on saving dataset for optimization"""
        if on is None:
            raise ValueError("'delta on' fields must be defined")
        self._delta = True
        self._delta_on = on
        self._delta_result_on = right_on
        self._delta_compare = compare
        self._delta_retry = delta_retry
        self._delta_unsafe = delta_unsafe
        return self

    @property
    def empty(self) -> bool:
        """Returns True if chain has zero number of rows"""
        return not bool(self.count())

    @property
    def delta(self) -> bool:
        """Returns True if this chain is ran in "delta" update mode"""
        return self._delta

    @property
    def delta_unsafe(self) -> bool:
        return self._delta_unsafe

    @property
    def schema(self) -> dict[str, DataType]:
        """Get schema of the chain."""
        return self._effective_signals_schema.values

    def column(self, name: str) -> Column:
        """Returns Column instance with a type if name is found in current schema,
        otherwise raises an exception.
        """
        if "." in name:
            name_path = name.split(".")
        elif DEFAULT_DELIMITER in name:
            name_path = name.split(DEFAULT_DELIMITER)
        else:
            name_path = [name]
        for path, type_, _, _ in self.signals_schema.get_flat_tree():
            if path == name_path:
                return Column(name, python_to_sql(type_))

        raise ValueError(f"Column with name {name} not found in the schema")

    def c(self, column: str | Column) -> Column:
        """Returns Column instance attached to the current chain."""
        c = self.column(column) if isinstance(column, str) else self.column(column.name)
        c.table = self._query.table
        return c

    @property
    def session(self) -> Session:
        """Session of the chain."""
        return self._query.session

    @property
    def name(self) -> str | None:
        """Name of the underlying dataset, if there is one."""
        return self._query.name

    @property
    def version(self) -> str | None:
        """Version of the underlying dataset, if there is one."""
        return self._query.version

    @property
    def dataset(self) -> DatasetRecord | None:
        """Underlying dataset, if there is one."""
        if not self.name:
            return None
        return self.session.catalog.get_dataset(
            self.name,
            namespace_name=self._query.project.namespace.name,
            project_name=self._query.project.name,
            include_incomplete=False,
        )

    def __or__(self, other: "Self") -> "Self":
        """Return `self.union(other)`."""
        return self.union(other)

    def print_schema(self, file: IO | None = None) -> None:
        """Print schema of the chain."""
        self._effective_signals_schema.print_tree(file=file)

    def clone(self) -> "Self":
        """Make a copy of the chain in a new table."""
        return self._evolve(query=self._query.clone(new_table=True))

    def _evolve(
        self,
        *,
        query: DatasetQuery | None = None,
        settings: Settings | None = None,
        signal_schema=None,
        _sys=None,
    ) -> "Self":
        if query is None:
            query = self._query.clone(new_table=False)
        if settings is None:
            settings = self._settings
        if signal_schema is None:
            signal_schema = copy.deepcopy(self.signals_schema)
        if _sys is None:
            _sys = self._sys
        chain = type(self)(
            query, settings, signal_schema=signal_schema, setup=self._setup, _sys=_sys
        )
        if self.delta:
            chain = chain._as_delta(
                on=self._delta_on,
                right_on=self._delta_result_on,
                compare=self._delta_compare,
                delta_retry=self._delta_retry,
                delta_unsafe=self._delta_unsafe,
            )

        return chain

    def settings(
        self,
        cache: bool | None = None,
        prefetch: bool | int | None = None,
        parallel: bool | int | None = None,
        workers: int | None = None,
        namespace: str | None = None,
        project: str | None = None,
        min_task_size: int | None = None,
        batch_size: int | None = None,
        sys: bool | None = None,
    ) -> "Self":
        """
        Set chain execution parameters. Returns the chain itself, allowing method
        chaining for subsequent operations. To restore all settings to their default
        values, use `reset_settings()`.

        Parameters:
            cache: Enable files caching to speed up subsequent accesses to the same
                files from the same or different chains. Defaults to False.
            prefetch: Enable prefetching of files. This will download files in
                advance in parallel. If an integer is provided, it specifies the number
                of files to prefetch concurrently for each process on each worker.
                Defaults to 2. Set to 0 or False to disable prefetching.
            parallel: Number of processes to use for processing user-defined functions
                (UDFs) in parallel. If an integer is provided, it specifies the number
                of CPUs to use. If True, all available CPUs are used. Defaults to 1.
            namespace: Namespace to use for the chain by default.
            project: Project to use for the chain by default.
            min_task_size: Minimum number of rows per worker/process for parallel
                processing by UDFs. Defaults to 1.
            batch_size: Number of rows per insert by UDF to fine tune and balance speed
                and memory usage. This might be useful when processing large rows
                or when running into memory issues. Defaults to 2000.

        Example:
            ```py
            chain = (
                chain
                .settings(cache=True, parallel=8, batch_size=300)
                .map(laion=process_webdataset(spec=WDSLaion), params="file")
            )
            ```
        """
        if sys is None:
            sys = self._sys
        settings = copy.copy(self._settings)
        settings.add(
            Settings(
                cache=cache,
                prefetch=prefetch,
                parallel=parallel,
                workers=workers,
                namespace=namespace,
                project=project,
                min_task_size=min_task_size,
                batch_size=batch_size,
            )
        )
        return self._evolve(settings=settings, _sys=sys)

    def reset_settings(self, settings: Settings | None = None) -> "Self":
        """Reset all chain settings to default values."""
        self._settings = settings if settings else Settings()
        return self

    @classmethod
    def from_storage(
        cls,
        *args,
        **kwargs,
    ) -> "DataChain":
        from .storage import read_storage

        warnings.warn(
            "Class method `from_storage` is deprecated. "
            "Use `read_storage` function instead from top_module.",
            DeprecationWarning,
            stacklevel=2,
        )
        return read_storage(*args, **kwargs)

    @classmethod
    def from_dataset(cls, *args, **kwargs) -> "DataChain":
        from .datasets import read_dataset

        warnings.warn(
            "Class method `from_dataset` is deprecated. "
            "Use `read_dataset` function instead from top_module.",
            DeprecationWarning,
            stacklevel=2,
        )
        return read_dataset(*args, **kwargs)

    @classmethod
    def from_json(
        cls,
        *args,
        **kwargs,
    ) -> "DataChain":
        from .json import read_json

        warnings.warn(
            "Class method `from_json` is deprecated. "
            "Use `read_json` function instead from top_module.",
            DeprecationWarning,
            stacklevel=2,
        )
        return read_json(*args, **kwargs)

    def explode(
        self,
        col: str,
        model_name: str | None = None,
        column: str | None = None,
        schema_sample_size: int = 1,
    ) -> "DataChain":
        """Explodes a column containing JSON objects (dict or str DataChain type) into
           individual columns based on the schema of the JSON. Schema is inferred from
           the first row of the column.

        Args:
            col: the name of the column containing JSON to be exploded.
            model_name: optional generated model name.  By default generates the name
                automatically.
            column: optional generated column name. By default generates the
                name automatically.
            schema_sample_size: the number of rows to use for inferring the schema of
                the JSON (in case some fields are optional and it's not enough to
                analyze a single row).

        Returns:
            DataChain: A new DataChain instance with the new set of columns.
        """
        import pyarrow as pa

        from datachain.lib.arrow import schema_to_output

        json_values = self.limit(schema_sample_size).to_list(col)
        json_dicts = [
            json.loads(json_value) if isinstance(json_value, str) else json_value
            for (json_value,) in json_values
        ]

        if any(not isinstance(json_dict, dict) for json_dict in json_dicts):
            raise TypeError(f"Column {col} should be a string or dict type with JSON")

        schema = pa.Table.from_pylist(json_dicts).schema
        output, original_names = schema_to_output(schema, None)

        if not model_name:
            model_name = f"{col.title()}ExplodedModel"

        model = dict_to_data_model(model_name, output, original_names)

        def json_to_model(json_value: str | dict):
            json_dict = (
                json.loads(json_value) if isinstance(json_value, str) else json_value
            )
            return model.model_validate(json_dict)

        if not column:
            column = f"{col}_expl"

        return self.map(json_to_model, params=col, output={column: model})

    @classmethod
    def datasets(
        cls,
        *args,
        **kwargs,
    ) -> "DataChain":
        from .datasets import datasets

        warnings.warn(
            "Class method `datasets` is deprecated. "
            "Use `datasets` function instead from top_module.",
            DeprecationWarning,
            stacklevel=2,
        )
        return datasets(*args, **kwargs)

    @classmethod
    def listings(
        cls,
        *args,
        **kwargs,
    ) -> "DataChain":
        from .listings import listings

        warnings.warn(
            "Class method `listings` is deprecated. "
            "Use `listings` function instead from top_module.",
            DeprecationWarning,
            stacklevel=2,
        )
        return listings(*args, **kwargs)

    @property
    def namespace_name(self) -> str:
        """Current namespace name in which the chain is running"""
        return (
            self._settings.namespace
            or self.session.catalog.metastore.default_namespace_name
        )

    @property
    def project_name(self) -> str:
        """Current project name in which the chain is running"""
        return (
            self._settings.project
            or self.session.catalog.metastore.default_project_name
        )

    def persist(self) -> "Self":
        """Saves temporary chain that will be removed after the process ends.
        Temporary datasets are useful for optimization, for example when we have
        multiple chains starting with identical sub-chain. We can then persist that
        common chain and use it to calculate other chains, to avoid re-calculation
        every time.
        It returns the chain itself.
        """
        schema = self.signals_schema.clone_without_sys_signals().serialize()
        project = self.session.catalog.metastore.get_project(
            self.project_name,
            self.namespace_name,
            create=True,
        )
        return self._evolve(
            query=self._query.save(project=project, feature_schema=schema),
            signal_schema=self.signals_schema | SignalSchema({"sys": Sys}),
        )

    def _calculate_job_hash(self, job_id: str) -> str:
        """
        Calculates hash of the job at the place of this chain's save method.
        Hash is calculated using previous job checkpoint hash (if exists) and
        adding hash of this chain to produce new hash.
        """
        last_checkpoint = self.session.catalog.metastore.get_last_checkpoint(job_id)

        return hashlib.sha256(
            (bytes.fromhex(last_checkpoint.hash) if last_checkpoint else b"")
            + bytes.fromhex(self.hash())
        ).hexdigest()

    def save(  # type: ignore[override]
        self,
        name: str,
        version: str | None = None,
        description: str | None = None,
        attrs: list[str] | None = None,
        update_version: str | None = "patch",
        **kwargs,
    ) -> "DataChain":
        """Save to a Dataset. It returns the chain itself.

        Parameters:
            name: dataset name. This can be either a fully qualified name, including
                the namespace and project, or just a regular dataset name. In the latter
                case, the namespace and project will be taken from the settings
                (if specified) or from the default values otherwise.
            version: version of a dataset. If version is not specified and dataset
                already exists, version patch increment will happen e.g 1.2.1 -> 1.2.2.
            description: description of a dataset.
            attrs: attributes of a dataset. They can be without value, e.g "NLP",
                or with a value, e.g "location=US".
            update_version: which part of the dataset version to automatically increase.
                Available values: `major`, `minor` or `patch`. Default is `patch`.
        """

        catalog = self.session.catalog

        result = None  # result chain that will be returned at the end

        # Version validation
        self._validate_version(version)
        self._validate_update_version(update_version)

        # get existing job if running in SaaS, or creating new one if running locally
        job = self.session.get_or_create_job()

        namespace_name, project_name, name = catalog.get_full_dataset_name(
            name,
            namespace_name=self._settings.namespace,
            project_name=self._settings.project,
        )
        project = self._get_or_create_project(namespace_name, project_name)

        # Checkpoint handling
        _hash, result = self._resolve_checkpoint(name, project, job, kwargs)
        if bool(result):
            # Checkpoint was found and reused
            print(f"Checkpoint found for dataset '{name}', skipping creation")

        # Schema preparation
        schema = self.signals_schema.clone_without_sys_signals().serialize()

        # Handle retry and delta functionality
        if not result:
            result = self._handle_delta(name, version, project, schema, kwargs)

        if not result:
            # calculate chain if we already don't have result from checkpoint or delta
            result = self._evolve(
                query=self._query.save(
                    name=name,
                    version=version,
                    project=project,
                    description=description,
                    attrs=attrs,
                    feature_schema=schema,
                    update_version=update_version,
                    **kwargs,
                )
            )

        catalog.metastore.create_checkpoint(job.id, _hash)  # type: ignore[arg-type]
        return result

    def _validate_version(self, version: str | None) -> None:
        """Validate dataset version if provided."""
        if version is not None:
            semver.validate(version)

    def _validate_update_version(self, update_version: str | None) -> None:
        """Ensure update_version is one of: major, minor, patch."""
        allowed = ["major", "minor", "patch"]
        if update_version not in allowed:
            raise ValueError(f"update_version must be one of {allowed}")

    def _get_or_create_project(self, namespace: str, project_name: str) -> Project:
        """Get project or raise if creation not allowed."""
        try:
            return self.session.catalog.metastore.get_project(
                project_name,
                namespace,
                create=is_studio(),
            )
        except ProjectNotFoundError as e:
            raise ProjectCreateNotAllowedError("Creating project is not allowed") from e

    def _resolve_checkpoint(
        self,
        name: str,
        project: Project,
        job: Job,
        kwargs: dict,
    ) -> tuple[str, "DataChain | None"]:
        """Check if checkpoint exists and return cached dataset if possible."""
        from .datasets import read_dataset

        metastore = self.session.catalog.metastore
        checkpoints_reset = env2bool("DATACHAIN_CHECKPOINTS_RESET", undefined=True)

        _hash = self._calculate_job_hash(job.id)

        if (
            job.parent_job_id
            and not checkpoints_reset
            and metastore.find_checkpoint(job.parent_job_id, _hash)
        ):
            # checkpoint found → find which dataset version to reuse

            # Find dataset version that was created by any ancestor job
            try:
                dataset_version = metastore.get_dataset_version_for_job_ancestry(
                    name,
                    project.namespace.name,
                    project.name,
                    job.id,
                )
            except JobAncestryDepthExceededError:
                raise JobAncestryDepthExceededError(
                    "Job continuation chain is too deep. "
                    "Please run the job from scratch without continuing from a "
                    "parent job."
                ) from None

            if not dataset_version:
                logger.debug(
                    "Checkpoint found but no dataset version for '%s' "
                    "in job ancestry (job_id=%s). Creating new version.",
                    name,
                    job.id,
                )
                # Dataset version not found (e.g deleted by user) - skip
                # checkpoint and recreate
                return _hash, None

            logger.debug(
                "Reusing dataset version '%s' v%s from job ancestry "
                "(job_id=%s, dataset_version_id=%s)",
                name,
                dataset_version.version,
                job.id,
                dataset_version.id,
            )

            # Read the specific version from ancestry
            chain = read_dataset(
                name,
                namespace=project.namespace.name,
                project=project.name,
                version=dataset_version.version,
                **kwargs,
            )

            # Link current job to this dataset version (not creator).
            # This also updates dataset_version.job_id.
            metastore.link_dataset_version_to_job(
                dataset_version.id,
                job.id,
                is_creator=False,
            )

            return _hash, chain

        return _hash, None

    def _handle_delta(
        self,
        name: str,
        version: str | None,
        project: Project,
        schema: dict,
        kwargs: dict,
    ) -> "DataChain | None":
        """Try to save as a delta dataset.
        Returns:
            A DataChain if delta logic could handle it, otherwise None to fall back
            to the regular save path (e.g., on first dataset creation).
        """
        from datachain.delta import delta_retry_update

        from .datasets import read_dataset

        if not self.delta or not name:
            return None

        assert self._delta_on is not None, "Delta chain must have delta_on defined"

        result_ds, dependencies, has_changes = delta_retry_update(
            self,
            project.namespace.name,
            project.name,
            name,
            on=self._delta_on,
            right_on=self._delta_result_on,
            compare=self._delta_compare,
            delta_retry=self._delta_retry,
        )

        # Case 1: delta produced a new dataset
        if result_ds:
            return self._evolve(
                query=result_ds._query.save(
                    name=name,
                    version=version,
                    project=project,
                    feature_schema=schema,
                    dependencies=dependencies,
                    **kwargs,
                )
            )

        # Case 2: no changes → reuse last version
        if not has_changes:
            # sources have not been changed so new version of resulting dataset
            # would be the same as previous one. To avoid duplicating exact
            # datasets, we won't create new version of it and we will return
            # current latest version instead.
            return read_dataset(
                name,
                namespace=project.namespace.name,
                project=project.name,
                **kwargs,
            )

        # Case 3: first creation of dataset
        return None

    def apply(self, func, *args, **kwargs):
        """Apply any function to the chain.

        Useful for reusing in a chain of operations.

        Example:
            ```py
            import datachain as dc
            def parse_stem(chain):
                return chain.map(
                    lambda file: file.get_file_stem()
                    output={"stem": str}
                )

            chain = (
                dc.read_storage("s3://my-bucket")
                .apply(parse_stem)
                .filter(C("stem").glob("*cat*"))
            )
            ```
        """
        return func(self, *args, **kwargs)

    def map(
        self,
        func: Callable | None = None,
        params: str | Sequence[str] | None = None,
        output: OutputType = None,
        **signal_map: Any,
    ) -> "Self":
        """Apply a function to each row to create new signals. The function should
        return a new object for each row. It returns a chain itself with new signals.

        Input-output relationship: 1:1

        Parameters:
            func: Function applied to each row.
            params: List of column names used as input for the function. Default
                    is taken from function signature.
            output: Dictionary defining new signals and their corresponding types.
                    Default type is taken from function signature. Default can be also
                    taken from kwargs - **signal_map (see below).
                    If signal name is defined using signal_map (see below) only a single
                    type value can be used.
            **signal_map: kwargs can be used to define `func` together with its return
                    signal name in format of `map(my_sign=my_func)`. This helps define
                    signal names and functions in a nicer way.

        Example:
            Using signal_map and single type in output:
            ```py
            chain = chain.map(value=lambda name: name[:-4] + ".json", output=str)
            chain.save("new_dataset")
            ```

            Using func and output as a map:
            ```py
            chain = chain.map(
                lambda name: name.split("."), output={"stem": str, "ext": str}
            )
            chain.save("new_dataset")
            ```
        """
        udf_obj = self._udf_to_obj(Mapper, func, params, output, signal_map)
        if (prefetch := self._settings.prefetch) is not None:
            udf_obj.prefetch = prefetch

        sys_schema = SignalSchema({"sys": Sys})
        return self._evolve(
            query=self._query.add_signals(
                udf_obj.to_udf_wrapper(self._settings.batch_size),
                **self._settings.to_dict(),
            ),
            signal_schema=sys_schema | self.signals_schema | udf_obj.output,
        )

    def gen(
        self,
        func: Callable | Generator | None = None,
        params: str | Sequence[str] | None = None,
        output: OutputType = None,
        **signal_map,
    ) -> "Self":
        r"""Apply a function to each row to create new rows (with potentially new
        signals). The function needs to return a new objects for each of the new rows.
        It returns a chain itself with new signals.

        Input-output relationship: 1:N

        This method is similar to `map()`, uses the same list of parameters, but with
        one key differences: It produces a sequence of rows for each input row (like
        extracting multiple file records from a single tar file or bounding boxes from a
        single image file).

        Example:
            ```py
            chain = chain.gen(
                line=lambda file: [l for l in file.read().split("\n")],
                output=str,
            )
            chain.save("new_dataset")
            ```
        """
        udf_obj = self._udf_to_obj(Generator, func, params, output, signal_map)
        if (prefetch := self._settings.prefetch) is not None:
            udf_obj.prefetch = prefetch
        return self._evolve(
            query=self._query.generate(
                udf_obj.to_udf_wrapper(self._settings.batch_size),
                **self._settings.to_dict(),
            ),
            signal_schema=SignalSchema({"sys": Sys}) | udf_obj.output,
        )

    @delta_disabled
    def agg(
        self,
        /,
        func: Callable | None = None,
        partition_by: PartitionByType | None = None,
        params: str | Sequence[str] | None = None,
        output: OutputType = None,
        **signal_map: Callable,
    ) -> "Self":
        """Aggregate rows using `partition_by` statement and apply a function to the
        groups of aggregated rows. The function needs to return new objects for each
        group of the new rows. It returns a chain itself with new signals.

        Input-output relationship: N:M

        This method bears similarity to `gen()` and `map()`, employing a comparable set
        of parameters, yet differs in two crucial aspects:

        1. The `partition_by` parameter: This specifies the column name or a list of
           column names that determine the grouping criteria for aggregation.
        2. Group-based UDF function input: Instead of individual rows, the function
           receives a list of all rows within each group defined by `partition_by`.

        If `partition_by` is not set or is an empty list, all rows will be placed
        into a single group.

        Parameters:
            func: Function applied to each group of rows.
            partition_by: Column name(s) to group by. If None, all rows go
                into one group.
            params: List of column names used as input for the function. Default is
                taken from function signature.
            output: Dictionary defining new signals and their corresponding types.
                Default type is taken from function signature.
            **signal_map: kwargs can be used to define `func` together with its return
                signal name in format of `agg(result_column=my_func)`.

        Examples:
            Basic aggregation with lambda function:
            ```py
            chain = chain.agg(
                total=lambda category, amount: [sum(amount)],
                output=float,
                partition_by="category",
            )
            chain.save("new_dataset")
            ```

            An alternative syntax, when you need to specify a more complex function:
            ```py
            # It automatically resolves which columns to pass to the function
            # by looking at the function signature.
            def agg_sum(
                file: list[File], amount: list[float]
            ) -> Iterator[tuple[File, float]]:
                yield file[0], sum(amount)

            chain = chain.agg(
                agg_sum,
                output={"file": File, "total": float},
                # Alternative syntax is to use `C` (short for Column) to specify
                # a column name or a nested column, e.g. C("file.path").
                partition_by=C("category"),
            )
            chain.save("new_dataset")
            ```

            Using complex signals for partitioning (`File` or any Pydantic `BaseModel`):
            ```py
            def my_agg(files: list[File]) -> Iterator[tuple[File, int]]:
                yield files[0], sum(f.size for f in files)

            chain = chain.agg(
                my_agg,
                params=("file",),
                output={"file": File, "total": int},
                partition_by="file",  # Column referring to all sub-columns of File
            )
            chain.save("new_dataset")
            ```

            Aggregating all rows into a single group (when `partition_by` is not set):
            ```py
            chain = chain.agg(
                total_size=lambda file, size: [sum(size)],
                output=int,
                # No partition_by specified - all rows go into one group
            )
            chain.save("new_dataset")
            ```

            Multiple partition columns:
            ```py
            chain = chain.agg(
                total=lambda category, subcategory, amount: [sum(amount)],
                output=float,
                partition_by=["category", "subcategory"],
            )
            chain.save("new_dataset")
            ```
        """
        if partition_by is not None:
            # Convert string partition_by parameters to Column objects
            if isinstance(partition_by, (str, Function, ColumnElement)):
                list_partition_by = [partition_by]
            else:
                list_partition_by = list(partition_by)

            processed_partition_columns: list[ColumnElement] = []
            for col in list_partition_by:
                if isinstance(col, str):
                    columns = self.signals_schema.db_signals(name=col, as_columns=True)
                    if not columns:
                        raise SignalResolvingError([col], "is not found")
                    processed_partition_columns.extend(cast("list[Column]", columns))
                elif isinstance(col, Function):
                    column = col.get_column(self.signals_schema)
                    processed_partition_columns.append(column)
                else:
                    # Assume it's already a ColumnElement
                    processed_partition_columns.append(col)

            processed_partition_by = processed_partition_columns
        else:
            processed_partition_by = []

        udf_obj = self._udf_to_obj(Aggregator, func, params, output, signal_map)
        return self._evolve(
            query=self._query.generate(
                udf_obj.to_udf_wrapper(self._settings.batch_size),
                partition_by=processed_partition_by,
                **self._settings.to_dict(),
            ),
            signal_schema=SignalSchema({"sys": Sys}) | udf_obj.output,
        )

    def batch_map(
        self,
        func: Callable | None = None,
        params: str | Sequence[str] | None = None,
        output: OutputType = None,
        batch: int = 1000,
        **signal_map,
    ) -> "Self":
        """This is a batch version of `map()`.

        Input-output relationship: N:N

        It accepts the same parameters plus an
        additional parameter:

            batch: Size of each batch passed to `func`. Defaults to 1000.

        Example:
            ```py
            chain = chain.batch_map(
                sqrt=lambda size: np.sqrt(size),
                output=float
            )
            chain.save("new_dataset")
            ```

        .. deprecated:: 0.29.0
            This method is deprecated and will be removed in a future version.
            Use `agg()` instead, which provides the similar functionality.
        """
        import warnings

        warnings.warn(
            "batch_map() is deprecated and will be removed in a future version. "
            "Use agg() instead, which provides the similar functionality.",
            DeprecationWarning,
            stacklevel=2,
        )
        udf_obj = self._udf_to_obj(BatchMapper, func, params, output, signal_map)

        return self._evolve(
            query=self._query.add_signals(
                udf_obj.to_udf_wrapper(self._settings.batch_size, batch=batch),
                **self._settings.to_dict(),
            ),
            signal_schema=self.signals_schema | udf_obj.output,
        )

    def _udf_to_obj(
        self,
        target_class: type[UDFObjT],
        func: Callable | UDFObjT | None,
        params: str | Sequence[str] | None,
        output: OutputType,
        signal_map: dict[str, Callable],
    ) -> UDFObjT:
        is_batch = target_class.is_input_batched
        is_generator = target_class.is_output_batched
        name = self.name or ""

        sign = UdfSignature.parse(name, signal_map, func, params, output, is_generator)
        DataModel.register(list(sign.output_schema.values.values()))

        params_schema = self.signals_schema.slice(
            sign.params, self._setup, is_batch=is_batch
        )

        return target_class._create(sign, params_schema)

    def _extend_to_data_model(self, method_name, *args, **kwargs):
        query_func = getattr(self._query, method_name)

        new_schema = self.signals_schema.resolve(*args)
        columns = new_schema.db_signals(as_columns=True)
        return query_func(*columns, **kwargs)

    @resolve_columns
    def order_by(self, *args, descending: bool = False) -> "Self":
        """Orders by specified set of columns.

        Parameters:
            descending (bool): Whether to sort in descending order or not.

        Example:
            ```py
            dc.order_by("similarity_score", descending=True).limit(10)
            ```

        Note:
            Order is not guaranteed when steps are added after an `order_by` statement.
            I.e. when using `read_dataset` an `order_by` statement should be used if
            the order of the records in the chain is important.
            Using `order_by` directly before `limit`, `to_list` and similar methods
            will give expected results.
            See https://github.com/datachain-ai/datachain/issues/477
            for further details.
        """
        if descending:
            args = tuple(sqlalchemy.desc(a) for a in args)

        return self._evolve(query=self._query.order_by(*args))

    @delta_disabled
    def distinct(self, arg: str, *args: str) -> "Self":  # type: ignore[override]
        """Removes duplicate rows based on uniqueness of some input column(s)
        i.e if rows are found with the same value of input column(s), only one
        row is left in the result set.

        Example:
            ```py
            dc.distinct("file.path")
            ```
        """
        return self._evolve(
            query=self._query.distinct(
                *self.signals_schema.resolve(arg, *args).db_signals()
            )
        )

    def select(self, *args: str) -> "Self":
        """Select only a specified set of signals.

        Nested selections (e.g. ``"file.path"``) preserve the parent object by
        generating partial models rather than flattening into standalone fields.
        """
        if not args:
            return self
        new_schema = self.signals_schema.to_partial(*args)

        if "sys" in self.signals_schema.values and "sys" not in new_schema.values:
            new_schema = SignalSchema({"sys": self.signals_schema.values["sys"]}) | (
                new_schema
            )

        columns = new_schema.db_signals()
        return self._evolve(
            query=self._query.select(*columns),
            signal_schema=new_schema,
        )

    def select_except(self, *args: str) -> "Self":
        """Select all signals except the specified ones.

        Supports excluding nested fields (e.g. ``"file.path"``), in which case a
        partial model is generated for the parent signal.
        """
        if not args:
            return self

        new_schema = self.signals_schema.select_except_signals(*args)

        columns = new_schema.db_signals()
        return self._evolve(
            query=self._query.select(*columns),
            signal_schema=new_schema,
        )

    @delta_disabled  # type: ignore[arg-type]
    def group_by(  # noqa: C901, PLR0912
        self,
        *,
        partition_by: str | Func | Sequence[str | Func] | None = None,
        **kwargs: Func,
    ) -> "Self":
        """Group rows by specified set of signals and return new signals
        with aggregated values.

        The supported functions:
           count(), sum(), avg(), min(), max(), any_value(), collect(), concat()

        Example:
            ```py
            chain = chain.group_by(
                cnt=func.count(),
                partition_by=("file_source", "file_ext"),
            )
            ```

            Using complex signals:
            ```py
            chain = chain.group_by(
                total_size=func.sum("file.size"),
                count=func.count(),
                partition_by="file",  # Uses column name, expands to File's unique keys
            )
            ```
        """
        if partition_by is None:
            partition_by = []
        elif isinstance(partition_by, (str, Func)):
            partition_by = [partition_by]

        partition_by_columns: list[Column] = []
        signal_columns: list[Column] = []
        schema_fields: dict[str, DataType] = {}
        keep_columns: list[str] = []
        partial_fields: list[str] = []  # Track specific fields for partial creation
        schema_partition_by: list[str] = []

        for col in partition_by:
            if isinstance(col, str):
                columns = self.signals_schema.db_signals(name=col, as_columns=True)
                if not columns:
                    raise SignalResolvingError([col], "is not found")
                partition_by_columns.extend(cast("list[Column]", columns))

                # For nested field references (e.g., "nested.level1.name"),
                # we need to distinguish between:
                # 1. References to fields within a complex signal (create partials)
                # 2. Deep nested references that should be flattened
                if "." in col:
                    # Split the column reference to analyze it
                    parts = col.split(".")
                    parent_signal = parts[0]
                    parent_type = self.signals_schema.values.get(parent_signal)

                    if ModelStore.is_partial(parent_type):
                        if parent_signal not in keep_columns:
                            keep_columns.append(parent_signal)
                        partial_fields.append(col)
                        schema_partition_by.append(col)
                    else:
                        # BaseModel or other - add flattened columns directly
                        for column in cast("list[Column]", columns):
                            col_type = self.signals_schema.get_column_type(column.name)
                            schema_fields[column.name] = col_type
                        schema_partition_by.append(col)
                else:
                    # simple signal - but we need to check if it's a complex signal
                    # complex signal - only include the columns used for partitioning
                    col_type = self.signals_schema.get_column_type(
                        col, with_subtree=True
                    )
                    if isinstance(col_type, type) and issubclass(col_type, BaseModel):
                        # Complex signal - add only the partitioning columns
                        for column in cast("list[Column]", columns):
                            col_type = self.signals_schema.get_column_type(column.name)
                            schema_fields[column.name] = col_type
                        schema_partition_by.append(col)
                    # Simple signal - keep the entire signal
                    else:
                        if col not in keep_columns:
                            keep_columns.append(col)
                        schema_partition_by.append(col)
            elif isinstance(col, Function):
                column = col.get_column(self.signals_schema)
                col_db_name = column.name
                col_type = column.type.python_type
                schema_fields[col_db_name] = col_type
                partition_by_columns.append(column)
                signal_columns.append(column)
            else:
                raise DataChainColumnError(
                    col,
                    (
                        f"partition_by column {col} has type {type(col)}"
                        " but expected str or Function"
                    ),
                )

        if not kwargs:
            raise ValueError("At least one column should be provided for group_by")
        for col_name, func in kwargs.items():
            if not isinstance(func, Func):
                raise DataChainColumnError(
                    col_name,
                    f"Column {col_name} has type {type(func)} but expected Func object",
                )
            column = func.get_column(self.signals_schema, label=col_name)
            signal_columns.append(column)
            schema_fields[col_name] = func.get_result_type(self.signals_schema)

        signal_schema = self.signals_schema.group_by(
            schema_partition_by, signal_columns
        )

        return self._evolve(
            query=self._query.group_by(signal_columns, partition_by_columns),
            signal_schema=signal_schema,
        )

    def mutate(self, **kwargs) -> "Self":
        """Create or modify signals based on existing signals.

        This method is vectorized and more efficient compared to map(), and it does not
        extract or download any data from the internal database. However, it can only
        utilize predefined built-in functions and their combinations.

        Supported functions:
           Numerical:   +, -, *, /, rand(), avg(), count(), func(),
                        greatest(), least(), max(), min(), sum()
           String:      length(), split(), replace(), regexp_replace()
           Filename:    name(), parent(), file_stem(), file_ext()
           Array:       length(), sip_hash_64(), euclidean_distance(),
                        cosine_distance()
           Window:      row_number(), rank(), dense_rank(), first()

        Example:
        ```py
         dc.mutate(
            area=Column("image.height") * Column("image.width"),
            extension=file_ext(Column("file.path")),
            dist=cosine_distance(embedding_text, embedding_image)
        )
        ```

        Window function example:
        ```py
        window = func.window(partition_by="file.parent", order_by="file.size")
        dc.mutate(
            row_number=func.row_number().over(window),
        )
        ```

        This method can be also used to rename signals. If the Column("name") provided
        as value for the new signal - the old signal will be dropped. Otherwise a new
        signal is created. Exception, if the old signal is nested one (e.g.
        `C("file.path")`), it will be kept to keep the object intact.

        Example:
        ```py
         dc.mutate(
            newkey=Column("oldkey") # drops oldkey
        )
        ```

        ```py
         dc.mutate(
            size=Column("file.size") # keeps `file.size`
        )
        ```
        """
        from sqlalchemy.sql.sqltypes import NullType

        primitives = (bool, str, int, float)

        for col_name, expr in kwargs.items():
            if not isinstance(expr, (*primitives, Column, Func)) and isinstance(
                expr.type, NullType
            ):
                raise DataChainColumnError(
                    col_name, f"Cannot infer type with expression {expr}"
                )

        mutated = {}
        schema = self.signals_schema
        for name, value in kwargs.items():
            if isinstance(value, Column):
                # renaming existing column
                for signal in schema.db_signals(name=value.name, as_columns=True):
                    mutated[signal.name.replace(value.name, name, 1)] = signal  # type: ignore[union-attr]
            elif isinstance(value, Func):
                # adding new signal
                mutated[name] = value.get_column(schema)
            elif isinstance(value, primitives):
                # adding simple python constant primitives like str, int, float, bool
                val = literal(value)
                val.type = python_to_sql(type(value))()
                mutated[name] = val  # type: ignore[assignment]
            else:
                # adding new signal
                mutated[name] = value

        new_schema = schema.mutate(kwargs)
        return self._evolve(
            query=self._query.mutate(new_schema=new_schema, **mutated),
            signal_schema=new_schema,
        )

    @property
    def _effective_signals_schema(self) -> "SignalSchema":
        """Effective schema used for user-facing API like to_list, to_pandas, etc."""
        signals_schema = self.signals_schema
        if not self._sys:
            return signals_schema.clone_without_sys_signals()
        return signals_schema

    @overload
    def _leaf_values(self) -> Iterator[tuple[Any, ...]]: ...

    @overload
    def _leaf_values(self, *, include_hidden: bool) -> Iterator[tuple[Any, ...]]: ...

    @overload
    def _leaf_values(
        self, *, row_factory: Callable[[list[str], tuple[Any, ...]], _T]
    ) -> Iterator[_T]: ...

    @overload
    def _leaf_values(
        self,
        *,
        row_factory: Callable[[list[str], tuple[Any, ...]], _T],
        include_hidden: bool,
    ) -> Iterator[_T]: ...

    def _leaf_values(self, *, row_factory=None, include_hidden: bool = True):
        """Yields flattened rows of values as a tuple.

        Args:
            row_factory: A callable to convert row to a custom format.
                It should accept two arguments: a list of column names and
                a tuple of row values.
            include_hidden: Whether to include hidden signals from the schema.
        """
        db_signals = self._effective_signals_schema.db_signals(
            include_hidden=include_hidden
        )

        with self._query.ordered_select(*db_signals).as_iterable() as rows:
            if row_factory:
                rows = (row_factory(db_signals, r) for r in rows)  # type: ignore[assignment]
            yield from rows

    def to_columnar_data_with_names(
        self, chunk_size: int = DEFAULT_PARQUET_CHUNK_SIZE
    ) -> tuple[list[str], Iterator[list[list[Any]]]]:
        """Returns column names and the results as an iterator that provides chunks,
        with each chunk containing a list of columns, where each column contains a
        list of the row values for that column in that chunk. Useful for columnar data
        formats, such as parquet or other OLAP databases.
        """
        headers, _ = self._effective_signals_schema.get_headers_with_length()
        column_names = [".".join(filter(None, header)) for header in headers]

        results_iter = self._leaf_values()

        def column_chunks() -> Iterator[list[list[Any]]]:
            for chunk_iter in batched_it(results_iter, chunk_size):
                columns: list[list[Any]] = [[] for _ in column_names]
                for row in chunk_iter:
                    for i, col in enumerate(columns):
                        col.append(row[i])
                yield columns

        return column_names, column_chunks()

    @overload
    def results(self) -> list[tuple[Any, ...]]: ...

    @overload
    def results(
        self, *, row_factory: Callable[[list[str], tuple[Any, ...]], _T]
    ) -> list[_T]: ...

    @overload
    def results(
        self,
        *,
        row_factory: Callable[[list[str], tuple[Any, ...]], _T],
        include_hidden: bool,
    ) -> list[_T]: ...

    @overload
    def results(self, *, include_hidden: bool) -> list[tuple[Any, ...]]: ...

    def results(self, *, row_factory=None, include_hidden=True):
        if row_factory is None:
            return list(self._leaf_values(include_hidden=include_hidden))
        return list(
            self._leaf_values(row_factory=row_factory, include_hidden=include_hidden)
        )

    def to_records(self) -> list[dict[str, Any]]:
        """Convert every row to a dictionary."""

        def to_dict(cols: list[str], row: tuple[Any, ...]) -> dict[str, Any]:
            return dict(zip(cols, row, strict=False))

        return self.results(row_factory=to_dict)

    def to_iter(self, *cols: str) -> Iterator[tuple[DataValue, ...]]:
        """Yields rows of values, optionally limited to the specified columns.

        Args:
            *cols: Limit to the specified columns. By default, all columns are selected.

        Yields:
            (tuple[DataType, ...]): Yields a tuple of items for each row.

        Example:
            Iterating over all rows:
            ```py
            for row in ds.to_iter():
                print(row)
            ```

            DataChain is iterable and can be used in a for loop directly which is
            equivalent to `ds.to_iter()`:
            ```py
            for row in ds:
                print(row)
            ```

            Iterating over all rows with selected columns:
            ```py
            for name, size in ds.to_iter("file.path", "file.size"):
                print(name, size)
            ```

            Iterating over a single column:
            ```py
            for (file,) in ds.to_iter("file.path"):
                print(file)
            ```
        """
        signals_schema = (
            self.signals_schema.resolve(*cols)
            if cols
            else self._effective_signals_schema
        )
        db_signals = signals_schema.db_signals()
        with self._query.ordered_select(*db_signals).as_iterable() as rows:
            for row in rows:
                ret = signals_schema.row_to_features(
                    row, catalog=self.session.catalog, cache=self._settings.cache
                )
                yield tuple(ret)

    @overload
    def collect(self) -> Iterator[tuple[DataValue, ...]]: ...

    @overload
    def collect(self, col: str) -> Iterator[DataValue]: ...

    @overload
    def collect(self, *cols: str) -> Iterator[tuple[DataValue, ...]]: ...

    def collect(self, *cols: str) -> Iterator[DataValue | tuple[DataValue, ...]]:  # type: ignore[overload-overlap,misc]
        """
        Deprecated. Use `to_iter` method instead.
        """
        warnings.warn(
            "Method `collect` is deprecated. Use `to_iter` method instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if len(cols) == 1:
            yield from [item[0] for item in self.to_iter(*cols)]
        else:
            yield from self.to_iter(*cols)

    def to_pytorch(
        self,
        transform=None,
        tokenizer=None,
        tokenizer_kwargs=None,
        num_samples=0,
        remove_prefetched: bool = False,
    ):
        """Convert to pytorch dataset format.

        Args:
            transform (Transform): Torchvision transforms to apply to the dataset.
            tokenizer (Callable): Tokenizer to use to tokenize text values.
            tokenizer_kwargs (dict): Additional kwargs to pass when calling tokenizer.
            num_samples (int): Number of random samples to draw for each epoch.
                This argument is ignored if `num_samples=0` (the default).
            remove_prefetched (bool): Whether to remove prefetched files after reading.

        Example:
            ```py
            from torch.utils.data import DataLoader
            loader = DataLoader(
                chain.select("file", "label").to_pytorch(),
                batch_size=16
            )
            ```
        """
        from datachain.torch import PytorchDataset

        if self._query.attached:
            chain = self
        else:
            chain = self.persist()
        assert chain.name is not None  # for mypy
        return PytorchDataset(
            chain.name,
            chain.version,
            catalog=self.session.catalog,
            transform=transform,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            num_samples=num_samples,
            dc_settings=chain._settings,
            remove_prefetched=remove_prefetched,
        )

    @delta_disabled
    def merge(
        self,
        right_ds: "DataChain",
        on: MergeColType | Sequence[MergeColType],
        right_on: MergeColType | Sequence[MergeColType] | None = None,
        inner=False,
        full=False,
        rname="right_",
    ) -> "Self":
        """Merge two chains based on the specified criteria.

        Parameters:
            right_ds: Chain to join with.
            on: Predicate ("column.name", C("column.name"), or Func) or list of
                Predicates to join on. If both chains have the same predicates then
                this predicate is enough for the join. Otherwise, `right_on` parameter
                has to specify the predicates for the other chain.
            right_on: Optional predicate or list of Predicates for the `right_ds`
                to join.
            inner (bool): Whether to run inner join or outer join.
            full (bool): Whether to run full outer join.
            rname (str): Name prefix for conflicting signal names.

        Examples:
            ```py
            meta = meta_emd.merge(meta_pq, on=(C.name, C.emd__index),
                                  right_on=(C.name, C.pq__index))
            ```

            ```py
            imgs.merge(captions,
                       on=func.path.file_stem(imgs.c("file.path")),
                       right_on=func.path.file_stem(captions.c("file.path"))
            ```
        )
        """
        if on is None:
            raise DatasetMergeError(["None"], None, "'on' must be specified")

        on = _validate_merge_on(on, self)
        if not on:
            raise DatasetMergeError(
                on,
                right_on,
                (
                    "'on' must be 'str', 'Func' or 'Sequence' object "
                    f"but got type '{type(on)}'"
                ),
            )

        if right_on is not None:
            right_on = _validate_merge_on(right_on, right_ds)
            if not right_on:
                raise DatasetMergeError(
                    on,
                    right_on,
                    "'right_on' must be 'str', 'Func' or 'Sequence' object"
                    f" but got type '{type(right_on)}'",
                )

            if len(right_on) != len(on):
                raise DatasetMergeError(
                    on, right_on, "'on' and 'right_on' must have the same length'"
                )

        if self == right_ds:
            right_ds = right_ds.clone()

        errors = []

        def _resolve(
            ds: DataChain,
            col: str | Function | sqlalchemy.ColumnElement,
            side: str | None,
        ):
            try:
                if isinstance(col, Function):
                    return ds.c(col.get_column())
                return ds.c(col) if isinstance(col, (str, C)) else col
            except ValueError:
                if side:
                    errors.append(f"{_get_merge_error_str(col)} in {side}")

        ops = [
            _resolve(self, left, "left")
            == _resolve(right_ds, right, "right" if right_on else None)
            for left, right in zip(on, right_on or on, strict=False)
        ]

        if errors:
            raise DatasetMergeError(
                on, right_on, f"Could not resolve {', '.join(errors)}"
            )

        query = self._query.join(
            right_ds._query, sqlalchemy.and_(*ops), inner, full, rname
        )
        query.feature_schema = None
        ds = self._evolve(query=query)

        # Note: merge drops sys signals from both sides, make sure to not include it
        # in the resulting schema
        signals_schema = self.signals_schema.clone_without_sys_signals()
        right_signals_schema = right_ds.signals_schema.clone_without_sys_signals()

        ds.signals_schema = signals_schema.merge(right_signals_schema, rname)

        return ds

    @delta_disabled
    def union(self, other: "Self") -> "Self":
        """Return the set union of the two datasets.

        Parameters:
            other: chain whose rows will be added to `self`.
        """
        self_schema = self.signals_schema
        other_schema = other.signals_schema
        missing_left, missing_right = self_schema.compare_signals(other_schema)
        if missing_left or missing_right:
            raise UnionSchemaMismatchError.from_column_sets(
                missing_left,
                missing_right,
            )

        self.signals_schema = self_schema.clone_without_sys_signals()
        return self._evolve(query=self._query.union(other._query))

    def subtract(  # type: ignore[override]
        self,
        other: "DataChain",
        on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
    ) -> "Self":
        """Remove rows that appear in another chain.

        Parameters:
            other: chain whose rows will be removed from `self`
            on: columns to consider for determining row equality in `self`.
                If unspecified, defaults to all common columns
                between `self` and `other`.
            right_on: columns to consider for determining row equality in `other`.
                If unspecified, defaults to the same values as `on`.
        """
        if isinstance(on, str):
            if not on:
                raise DataChainParamsError("'on' cannot be an empty string")
            on = [on]
        elif isinstance(on, Sequence):
            if not on or any(not col for col in on):
                raise DataChainParamsError("'on' cannot contain empty strings")

        if isinstance(right_on, str):
            if not right_on:
                raise DataChainParamsError("'right_on' cannot be an empty string")
            right_on = [right_on]
        elif isinstance(right_on, Sequence):
            if not right_on or any(not col for col in right_on):
                raise DataChainParamsError("'right_on' cannot contain empty strings")

        if on is None and right_on is None:
            other_columns = set(other._effective_signals_schema.db_signals())
            signals = [
                c
                for c in self._effective_signals_schema.db_signals()
                if c in other_columns
            ]
            if not signals:
                raise DataChainParamsError("subtract(): no common columns")
        elif on is not None and right_on is None:
            right_on = on
            signals = list(self.signals_schema.resolve(*on).db_signals())
        elif on is None and right_on is not None:
            raise DataChainParamsError(
                "'on' must be specified when 'right_on' is provided"
            )
        else:
            if not isinstance(on, Sequence) or not isinstance(right_on, Sequence):
                raise TypeError(
                    "'on' and 'right_on' must be 'str' or 'Sequence' object"
                )
            if len(on) != len(right_on):
                raise DataChainParamsError(
                    "'on' and 'right_on' must have the same length"
                )
            signals = list(
                zip(
                    self.signals_schema.resolve(*on).db_signals(),
                    other.signals_schema.resolve(*right_on).db_signals(),
                    strict=False,
                )  # type: ignore[arg-type]
            )
        return self._evolve(query=self._query.subtract(other._query, signals))  # type: ignore[arg-type]

    def diff(
        self,
        other: "DataChain",
        on: str | Sequence[str],
        right_on: str | Sequence[str] | None = None,
        compare: str | Sequence[str] | None = None,
        right_compare: str | Sequence[str] | None = None,
        added: bool = True,
        deleted: bool = True,
        modified: bool = True,
        same: bool = False,
        status_col: str | None = None,
    ) -> "DataChain":
        """Calculate differences between two chains.

        This method identifies records that are added, deleted, modified, or unchanged
        between two chains. It adds a status column with values: A=added, D=deleted,
        M=modified, S=same.

        Parameters:
            other: Chain to compare against.
            on: Column(s) to match records between chains.
            right_on: Column(s) in the other chain to match against. Defaults to `on`.
            compare: Column(s) to check for changes.
                     If not specified,all columns are used.
            right_compare: Column(s) in the other chain to compare against.
                     Defaults to values of `compare`.
            added (bool): Include records that exist in this chain but not in the other.
            deleted (bool): Include records that exist only in the other chain.
            modified (bool): Include records that exist in both
                     but have different values.
            same (bool): Include records that are identical in both chains.
            status_col (str): Name for the status column showing differences.

        Default behavior: By default, shows added, deleted, and modified records,
        but excludes unchanged records (same=False). Status column is not created.

        Example:
            ```py
            res = persons.diff(
                new_persons,
                on=["id"],
                right_on=["other_id"],
                compare=["name"],
                added=True,
                deleted=True,
                modified=True,
                same=True,
                status_col="diff"
            )
            ```
        """
        from datachain.diff import _compare

        return _compare(
            self,
            other,
            on,
            right_on=right_on,
            compare=compare,
            right_compare=right_compare,
            added=added,
            deleted=deleted,
            modified=modified,
            same=same,
            status_col=status_col,
        )

    def file_diff(
        self,
        other: "DataChain",
        on: str = "file",
        right_on: str | None = None,
        added: bool = True,
        modified: bool = True,
        deleted: bool = False,
        same: bool = False,
        status_col: str | None = None,
    ) -> "DataChain":
        """Calculate differences between two chains containing files.

        This method is specifically designed for file chains. It uses file `source`
        and `path` to match files, and file `version` and `etag` to detect changes.

        Parameters:
            other: Chain to compare against.
            on: File column name in this chain. Default is "file".
            right_on: File column name in the other chain. Defaults to `on`.
            added (bool): Include files that exist in this chain but not in the other.
            deleted (bool): Include files that exist only in the other chain.
            modified (bool): Include files that exist in both but have different
                             versions/etags.
            same (bool): Include files that are identical in both chains.
            status_col (str): Name for the status column showing differences
                              (A=added, D=deleted, M=modified, S=same).

        Default behavior: By default, includes only new files (added=True and
        modified=True). This is useful for incremental processing.

        Example:
            ```py
            diff = images.file_diff(
                new_images,
                on="file",
                right_on="other_file",
                added=True,
                deleted=True,
                modified=True,
                same=True,
                status_col="diff"
            )
            ```
        """
        on_file_signals = ["source", "path"]
        compare_file_signals = ["version", "etag"]

        def get_file_signals(file: str, signals):
            return [f"{file}.{c}" for c in signals]

        right_on = right_on or on

        on_cols = get_file_signals(on, on_file_signals)
        right_on_cols = get_file_signals(right_on, on_file_signals)
        compare_cols = get_file_signals(on, compare_file_signals)
        right_compare_cols = get_file_signals(right_on, compare_file_signals)

        return self.diff(
            other,
            on_cols,
            right_on=right_on_cols,
            compare=compare_cols,
            right_compare=right_compare_cols,
            added=added,
            deleted=deleted,
            modified=modified,
            same=same,
            status_col=status_col,
        )

    @classmethod
    def from_values(
        cls,
        *args,
        **kwargs,
    ) -> "DataChain":
        from .values import read_values

        warnings.warn(
            "Class method `from_values` is deprecated. "
            "Use `read_values` function instead from top_module.",
            DeprecationWarning,
            stacklevel=2,
        )
        return read_values(*args, **kwargs)

    @classmethod
    def from_pandas(
        cls,
        *args,
        **kwargs,
    ) -> "DataChain":
        from .pandas import read_pandas

        warnings.warn(
            "Class method `from_pandas` is deprecated. "
            "Use `read_pandas` function instead from top_module.",
            DeprecationWarning,
            stacklevel=2,
        )
        return read_pandas(*args, **kwargs)

    def to_pandas(
        self,
        flatten: bool = False,
        include_hidden: bool = True,
        as_object: bool = False,
    ) -> "pd.DataFrame":
        """Return a pandas DataFrame from the chain.

        Parameters:
            flatten: Whether to use a multiindex or flatten column names.
            include_hidden: Whether to include hidden columns.
            as_object: Whether to emit a dataframe backed by Python objects
                rather than pandas-inferred dtypes.

        Returns:
            pd.DataFrame: A pandas DataFrame representation of the chain.
        """
        import pandas as pd

        headers, max_length = self._effective_signals_schema.get_headers_with_length(
            include_hidden=include_hidden
        )

        columns: list[str] | pd.MultiIndex
        if flatten or max_length < 2:
            columns = [".".join(filter(None, header)) for header in headers]
        else:
            columns = pd.MultiIndex.from_tuples(map(tuple, headers))

        results = self.results(include_hidden=include_hidden)
        if as_object:
            df = pd.DataFrame(results, columns=columns, dtype=object)
            df.where(pd.notna(df), None, inplace=True)
            return df
        return pd.DataFrame.from_records(results, columns=columns)

    def show(
        self,
        limit: int = 20,
        flatten: bool = False,
        transpose: bool = False,
        truncate: bool = True,
        include_hidden: bool = False,
    ) -> None:
        """Show a preview of the chain results.

        Parameters:
            limit: How many rows to show.
            flatten: Whether to use a multiindex or flatten column names.
            transpose: Whether to transpose rows and columns.
            truncate: Whether or not to truncate the contents of columns.
            include_hidden: Whether to include hidden columns.
        """
        import pandas as pd

        dc = self.limit(limit) if limit > 0 else self  # type: ignore[misc]
        df = dc.to_pandas(
            flatten,
            include_hidden=include_hidden,
            as_object=True,
        )

        if df.empty:
            print("Empty result")
            print(f"Columns: {list(df.columns)}")
            return

        if transpose:
            df = df.T

        options: list = [
            "display.max_columns",
            None,
            "display.multi_sparse",
            False,
        ]

        try:
            if columns := os.get_terminal_size().columns:
                options.extend(["display.width", columns])
        except OSError:
            pass

        if not truncate:
            options.extend(["display.max_colwidth", None])

        with pd.option_context(*options):
            if inside_notebook():
                from IPython.display import display

                display(df)
            else:
                print(df)

        if len(df) == limit:
            print(f"\n[Limited by {len(df)} rows]")

    @classmethod
    def from_hf(
        cls,
        *args,
        **kwargs,
    ) -> "DataChain":
        from .hf import read_hf

        warnings.warn(
            "Class method `from_hf` is deprecated. "
            "Use `read_hf` function instead from top_module.",
            DeprecationWarning,
            stacklevel=2,
        )
        return read_hf(*args, **kwargs)

    def parse_tabular(
        self,
        output: OutputType = None,
        column: str = "",
        model_name: str = "",
        source: bool = True,
        nrows: int | None = None,
        **kwargs: Any,
    ) -> "Self":
        """Generate chain from list of tabular files.

        Parameters:
            output: Dictionary or feature class defining column names and their
                corresponding types. List of column names is also accepted, in which
                case types will be inferred.
            column: Generated column name.
            model_name: Generated model name.
            source: Whether to include info about the source file.
            nrows: Optional row limit.
            kwargs: Parameters to pass to pyarrow.dataset.dataset.

        Example:
            Reading a json lines file:
            ```py
            import datachain as dc
            chain = dc.read_storage("s3://mybucket/file.jsonl")
            chain = chain.parse_tabular(format="json")
            ```

            Reading a filtered list of files as a dataset:
            ```py
            import datachain as dc

            chain = dc.read_storage("s3://mybucket")
            chain = chain.filter(dc.C("file.path").glob("*.jsonl"))
            chain = chain.parse_tabular(format="json")
            ```
        """
        from pyarrow.dataset import CsvFileFormat, JsonFileFormat

        from datachain.lib.arrow import (
            ArrowGenerator,
            fix_pyarrow_format,
            infer_schema,
            schema_to_output,
        )

        parse_options = kwargs.pop("parse_options", None)
        if format := kwargs.get("format"):
            kwargs["format"] = fix_pyarrow_format(format, parse_options)

        if (
            nrows
            and format not in ["csv", "json"]
            and not isinstance(format, (CsvFileFormat, JsonFileFormat))
        ):
            raise DatasetPrepareError(
                self.name,
                "error in `parse_tabular` - "
                "`nrows` only supported for csv and json formats.",
            )

        if "file" not in self.schema or not self.count():
            raise DatasetPrepareError(self.name, "no files to parse.")

        schema = None
        col_names = output if isinstance(output, Sequence) else None
        if col_names or not output:
            try:
                schema = infer_schema(self, **kwargs, parse_options=parse_options)
                output, _ = schema_to_output(schema, col_names)
            except ValueError as e:
                raise DatasetPrepareError(self.name, e) from e

        if isinstance(output, dict):
            model_name = model_name or column or ""
            model = dict_to_data_model(model_name, output)
            output = model
        else:
            model = output  # type: ignore[assignment]

        if column:
            output = {column: model}  # type: ignore[dict-item]
        elif isinstance(output, type(BaseModel)):
            output = {
                name: info.annotation  # type: ignore[misc]
                for name, info in output.model_fields.items()
            }

        if source:
            output = {"source": ArrowRow} | output  # type: ignore[assignment,operator]

        # disable prefetch if nrows is set
        settings = {"prefetch": 0} if nrows else {}
        return self.settings(**settings).gen(  # type: ignore[arg-type]
            ArrowGenerator(
                schema,
                model,
                source,
                nrows,
                parse_options=parse_options,
                **kwargs,
            ),
            output=output,
        )

    @classmethod
    def from_csv(
        cls,
        *args,
        **kwargs,
    ) -> "DataChain":
        from .csv import read_csv

        warnings.warn(
            "Class method `from_csv` is deprecated. "
            "Use `read_csv` function instead from top_module.",
            DeprecationWarning,
            stacklevel=2,
        )
        return read_csv(*args, **kwargs)

    @classmethod
    def from_parquet(
        cls,
        *args,
        **kwargs,
    ) -> "DataChain":
        from .parquet import read_parquet

        warnings.warn(
            "Class method `from_parquet` is deprecated. "
            "Use `read_parquet` function instead from top_module.",
            DeprecationWarning,
            stacklevel=2,
        )
        return read_parquet(*args, **kwargs)

    def to_parquet(
        self,
        path: str | os.PathLike[str] | BinaryIO,
        partition_cols: Sequence[str] | None = None,
        chunk_size: int = DEFAULT_PARQUET_CHUNK_SIZE,
        fs_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Save chain to parquet file with SignalSchema metadata.

        Parameters:
            path: Path or a file-like binary object to save the file. This supports
                local paths as well as remote paths, such as s3:// or hf:// with fsspec.
            partition_cols: Column names by which to partition the dataset.
            chunk_size: The chunk size of results to read and convert to columnar
                data, to avoid running out of memory.
            fs_kwargs: Optional kwargs forwarded to the underlying fsspec filesystem
                when writing (e.g., s3://, gs://, hf://), fsspec-specific options
                are supported.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        from datachain.lib.arrow import DATACHAIN_SIGNAL_SCHEMA_PARQUET_KEY

        fsspec_fs = None

        if isinstance(path, str) and "://" in path:
            from datachain.client.fsspec import Client

            fs_kwargs = {
                **self._query.catalog.client_config,
                **(fs_kwargs or {}),
            }

            client = Client.get_implementation(path)

            if path.startswith("file://"):
                # pyarrow does not handle file:// uris, and needs a direct path instead.
                from urllib.parse import urlparse

                path = urlparse(path).path
                if sys.platform == "win32":
                    path = os.path.normpath(path.lstrip("/"))

            fsspec_fs = client.create_fs(**fs_kwargs)

        _partition_cols = list(partition_cols) if partition_cols else None
        signal_schema_metadata = json.dumps(
            self._effective_signals_schema.serialize(), ensure_ascii=False
        ).encode("utf-8")

        column_names, column_chunks = self.to_columnar_data_with_names(chunk_size)

        parquet_schema = None
        parquet_writer = None
        first_chunk = True

        for chunk in column_chunks:
            # pyarrow infers the best parquet schema from the python types of
            # the input data.
            table = pa.Table.from_pydict(
                dict(zip(column_names, chunk, strict=False)),
                schema=parquet_schema,
            )

            # Preserve any existing metadata, and add the DataChain SignalSchema.
            existing_metadata = table.schema.metadata or {}
            merged_metadata = {
                **existing_metadata,
                DATACHAIN_SIGNAL_SCHEMA_PARQUET_KEY: signal_schema_metadata,
            }
            table = table.replace_schema_metadata(merged_metadata)
            parquet_schema = table.schema

            if _partition_cols:
                # Write to a partitioned parquet dataset.
                pq.write_to_dataset(
                    table,
                    root_path=path,
                    partition_cols=_partition_cols,
                    filesystem=fsspec_fs,
                    **kwargs,
                )
            else:
                if first_chunk:
                    # Write to a single parquet file.
                    parquet_writer = pq.ParquetWriter(
                        path, parquet_schema, filesystem=fsspec_fs, **kwargs
                    )
                    first_chunk = False

                assert parquet_writer
                parquet_writer.write_table(table)

        if parquet_writer:
            parquet_writer.close()

    def to_csv(
        self,
        path: str | os.PathLike[str],
        delimiter: str = ",",
        fs_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> File:
        """Save chain to a csv (comma-separated values) file and return the stored
        `File`.

        Parameters:
            path: Path to save the file. This supports local paths as well as
                remote paths, such as s3:// or hf:// with fsspec.
            delimiter: Delimiter to use for the resulting file.
            fs_kwargs: Optional kwargs forwarded to the underlying fsspec filesystem
                when writing (e.g., s3://, gs://, hf://), fsspec-specific options
                are supported.
        Returns:
            File: The stored file with refreshed metadata (version, etag, size).
        """
        import csv

        target = File.at(path, session=self.session)

        headers, _ = self._effective_signals_schema.get_headers_with_length()
        column_names = [".".join(filter(None, header)) for header in headers]

        with target.open("w", newline="", client_config=fs_kwargs) as f:
            writer = csv.writer(f, delimiter=delimiter, **kwargs)
            writer.writerow(column_names)
            for row in self._leaf_values():
                writer.writerow(row)

        return target

    def to_json(
        self,
        path: str | os.PathLike[str],
        fs_kwargs: dict[str, Any] | None = None,
        include_outer_list: bool = True,
    ) -> File:
        """Save chain to a JSON file and return the stored `File`.

        Parameters:
            path: Path to save the file. This supports local paths as well as
                remote paths, such as s3:// or hf:// with fsspec.
            fs_kwargs: Optional kwargs forwarded to the underlying fsspec filesystem
                when writing (e.g., s3://, gs://, hf://), fsspec-specific options
                are supported.
            include_outer_list: Sets whether to include an outer list for all rows.
                Setting this to True makes the file valid JSON, while False instead
                writes in the JSON lines format.
        Returns:
            File: The stored file with refreshed metadata (version, etag, size).
        """
        target = File.at(path, session=self.session)
        headers, _ = self._effective_signals_schema.get_headers_with_length()
        headers = [list(filter(None, h)) for h in headers]
        with target.open("wb", client_config=fs_kwargs) as f:
            self._write_json_stream(f, headers, include_outer_list)
        return target

    def _write_json_stream(
        self,
        f: IO[bytes],
        headers: list[list[str]],
        include_outer_list: bool,
    ) -> None:
        is_first = True
        if include_outer_list:
            f.write(b"[\n")
        for row in self._leaf_values():
            if not is_first:
                f.write(b",\n" if include_outer_list else b"\n")
            else:
                is_first = False
            f.write(
                json.dumps(
                    row_to_nested_dict(headers, row),
                    ensure_ascii=False,
                ).encode("utf-8")
            )
        if include_outer_list:
            f.write(b"\n]\n")

    def to_jsonl(
        self,
        path: str | os.PathLike[str],
        fs_kwargs: dict[str, Any] | None = None,
    ) -> File:
        """Save chain to a JSON lines file.

        Parameters:
            path: Path to save the file. This supports local paths as well as
                remote paths, such as s3:// or hf:// with fsspec.
            fs_kwargs: Optional kwargs forwarded to the underlying fsspec filesystem
                when writing (e.g., s3://, gs://, hf://), fsspec-specific options
                are supported.
        Returns:
            File: The stored file with refreshed metadata (version, etag, size).
        """
        return self.to_json(path, fs_kwargs, include_outer_list=False)

    def to_database(
        self,
        table_name: str,
        connection: "ConnectionType",
        *,
        batch_size: int = DEFAULT_DATABASE_BATCH_SIZE,
        on_conflict: str | None = None,
        conflict_columns: list[str] | None = None,
        column_mapping: dict[str, str | None] | None = None,
    ) -> int:
        """Save chain to a database table using a given database connection.

        This method exports all DataChain records to a database table, creating the
        table if it doesn't exist and appending data if it does. The table schema
        is automatically inferred from the DataChain's signal schema.

        For PostgreSQL, tables are created in the schema specified by the connection's
        search_path (defaults to 'public'). Use URL parameters to target specific
        schemas.

        Parameters:
            table_name: Name of the database table to create/write to.
            connection: SQLAlchemy connectable, str, or a sqlite3 connection
                Using SQLAlchemy makes it possible to use any DB supported by that
                library. If a DBAPI2 object, only sqlite3 is supported. The user is
                responsible for engine disposal and connection closure for the
                SQLAlchemy connectable; str connections are closed automatically.
            batch_size: Number of rows to insert per batch for optimal performance.
                Larger batches are faster but use more memory. Default: 10,000.
            on_conflict: Strategy for handling duplicate rows (requires table
                constraints):
                - None: Raise error (`sqlalchemy.exc.IntegrityError`) on conflict
                  (default)
                - "ignore": Skip duplicate rows silently
                - "update": Update existing rows with new values
            conflict_columns: List of column names that form a unique constraint
                for conflict resolution. Required when on_conflict='update' and
                using PostgreSQL.
            column_mapping: Optional mapping to rename or skip columns:
                - Dict mapping DataChain column names to database column names
                - Set values to None to skip columns entirely, or use `defaultdict` to
                  skip all columns except those specified.

        Returns:
            int: Number of rows affected (inserted/updated). -1 if DB driver doesn't
                 support telemetry.

        Examples:
            Basic usage with PostgreSQL:
            ```py
            import datachain as dc

            rows_affected = (dc
              .read_storage("s3://my-bucket/")
              .to_database("files_table", "postgresql://user:pass@localhost/mydb")
            )
            print(f"Inserted/updated {rows_affected} rows")
            ```

            Using SQLite with connection string:
            ```py
            rows_affected = chain.to_database("my_table", "sqlite:///data.db")
            print(f"Affected {rows_affected} rows")
            ```

            Column mapping and renaming:
            ```py
            mapping = {
                "user.id": "id",
                "user.name": "name",
                "user.password": None  # Skip this column
            }
            chain.to_database("users", engine, column_mapping=mapping)
            ```

            Handling conflicts (requires PRIMARY KEY or UNIQUE constraints):
            ```py
            # Skip duplicates
            chain.to_database("my_table", engine, on_conflict="ignore")

            # Update existing records
            chain.to_database(
               "my_table", engine, on_conflict="update", conflict_columns=["id"]
            )
            ```

            Working with different databases:
            ```py
            # MySQL
            mysql_engine = sa.create_engine("mysql+pymysql://user:pass@host/db")
            chain.to_database("mysql_table", mysql_engine)

            # SQLite in-memory
            chain.to_database("temp_table", "sqlite:///:memory:")
            ```

            PostgreSQL with schema support:
            ```py
            pg_url = "postgresql://user:pass@host/db?options=-c search_path=analytics"
            chain.to_database("processed_data", pg_url)
            ```
        """
        from .database import to_database

        return to_database(
            self,
            table_name,
            connection,
            batch_size=batch_size,
            on_conflict=on_conflict,
            conflict_columns=conflict_columns,
            column_mapping=column_mapping,
        )

    @classmethod
    def from_records(
        cls,
        *args,
        **kwargs,
    ) -> "DataChain":
        from .records import read_records

        warnings.warn(
            "Class method `from_records` is deprecated. "
            "Use `read_records` function instead from top_module.",
            DeprecationWarning,
            stacklevel=2,
        )
        return read_records(*args, **kwargs)

    def sum(self, col: str) -> StandardType:  # type: ignore[override]
        """Compute the sum of a column.

        Parameters:
            col: The column to compute the sum for.

        Returns:
            The sum of the column values.

        Example:
            ```py
            total_size = chain.sum("file.size")
            print(f"Total size: {total_size}")
            ```
        """
        return self._extend_to_data_model("sum", col)

    def avg(self, col: str) -> StandardType:  # type: ignore[override]
        """Compute the average of a column.

        Parameters:
            col: The column to compute the average for.

        Returns:
            The average of the column values.

        Example:
            ```py
            average_size = chain.avg("file.size")
            print(f"Average size: {average_size}")
            ```
        """
        return self._extend_to_data_model("avg", col)

    def min(self, col: str) -> StandardType:  # type: ignore[override]
        """Compute the minimum of a column.

        Parameters:
            col: The column to compute the minimum for.

        Returns:
            The minimum value in the column.

        Example:
            ```py
            min_size = chain.min("file.size")
            print(f"Minimum size: {min_size}")
            ```
        """
        return self._extend_to_data_model("min", col)

    def max(self, col: str) -> StandardType:  # type: ignore[override]
        """Compute the maximum of a column.

        Parameters:
            col: The column to compute the maximum for.

        Returns:
            The maximum value in the column.

        Example:
            ```py
            max_size = chain.max("file.size")
            print(f"Maximum size: {max_size}")
            ```
        """
        return self._extend_to_data_model("max", col)

    def setup(self, **kwargs) -> "Self":
        """Setup variables to pass to UDF functions.

        Use before running map/gen/agg to save an object and pass it as an
        argument to the UDF.

        The value must be a callable (a `lambda: <value>` syntax can be used to quickly
        create one) that returns the object to be passed to the UDF. It is evaluated
        lazily when UDF is running, in case of multiple machines the callable is run on
        a worker machine.

        Example:
            ```py
            import anthropic
            from anthropic.types import Message
            import datachain as dc

            (
                dc.read_storage(DATA, type="text")
                .settings(parallel=4, cache=True)

                # Setup Anthropic client and pass it to the UDF below automatically
                # The value is callable (see the note above)
                .setup(client=lambda: anthropic.Anthropic(api_key=API_KEY))

                .map(
                    claude=lambda client, file: client.messages.create(
                        model=MODEL,
                        system=PROMPT,
                        messages=[{"role": "user", "content": file.get_value()}],
                    ),
                    output=Message,
                )
            )
            ```
        """
        intersection = set(self._setup.keys()) & set(kwargs.keys())
        if intersection:
            keys = ", ".join(intersection)
            raise DatasetPrepareError(self.name, f"this value(s) already setup: {keys}")

        self._setup = self._setup | kwargs
        return self

    def to_storage(
        self,
        output: str | os.PathLike[str],
        signal: str = "file",
        placement: FileExportPlacement = "fullpath",
        link_type: Literal["copy", "symlink"] = "copy",
        num_threads: int | None = EXPORT_FILES_MAX_THREADS,
        anon: bool | None = None,
        client_config: dict | None = None,
    ) -> None:
        """Export files from a specified signal to a directory. Files can be
        exported to a local or cloud directory.

        Args:
            output: Path to the target directory for exporting files.
            signal: Name of the signal to export files from.
            placement: The method to use for naming exported files.
                The possible values are: "filename", "etag", "fullpath",
                "filepath", and "checksum".
                Example path translations for an object located at
                ``s3://bucket/data/img.jpg`` and exported to ``./out``:

                - "filename" -> ``./out/img.jpg`` (no directories)
                - "filepath" -> ``./out/data/img.jpg`` (relative path kept)
                - "fullpath" -> ``./out/bucket/data/img.jpg`` (remote host kept)
                - "etag" -> ``./out/<etag>.jpg`` (unique name via object digest)

                Local sources behave like "filepath" for "fullpath" placement.
                Relative destinations such as "." or ".." and absolute paths
                are supported for every strategy.
            link_type: Method to use for exporting files.
                Falls back to `'copy'` if symlinking fails.
            num_threads: number of threads to use for exporting files.
                By default, it uses 5 threads.
            anon: If True, we will treat cloud bucket as a public one. Default behavior
                depends on the previous session configuration (e.g. happens in the
                initial `read_storage`) and particular cloud storage client
                implementation (e.g. S3 fallbacks to anonymous access if no credentials
                were found).
            client_config: Optional configuration for the destination storage client

        Example:
            Cross cloud transfer
            ```py
            import datachain as dc

            ds = dc.read_storage("s3://mybucket")
            ds.to_storage("gs://mybucket", placement="filename")
            ```
        """
        chain = self.persist()
        count = chain.count()

        if placement == "filename" and (
            chain._query.distinct(pathfunc.name(C(f"{signal}__path"))).count() != count
        ):
            raise ValueError("Files with the same name found")

        if anon is not None:
            client_config = (client_config or {}) | {"anon": anon}

        progress_bar = tqdm(
            desc=f"Exporting files to {output}: ",
            unit=" files",
            unit_scale=True,
            unit_divisor=10,
            total=count,
            leave=False,
        )
        file_exporter = FileExporter(
            output,
            placement,
            self._settings.cache if self._settings else False,
            link_type,
            max_threads=num_threads or 1,
            client_config=client_config,
        )
        file_exporter.run(
            (rows[0] for rows in chain.to_iter(signal)),
            progress_bar,
        )

    def shuffle(self) -> "Self":
        """Shuffle rows with a best-effort deterministic ordering.

        This produces repeatable shuffles. Merge and union operations can
        lead to non-deterministic results. Use order by or save a dataset
        afterward to guarantee the same result.
        """
        query = self._query.clone(new_table=False)
        query.steps.append(RegenerateSystemColumns(self._query.catalog))

        chain = self._evolve(
            query=query,
            signal_schema=SignalSchema({"sys": Sys}) | self.signals_schema,
        )
        return chain.order_by("sys.rand")

    def sample(self, n: int) -> "Self":
        """Return a random sample from the chain.

        Parameters:
            n: Number of samples to draw.

        Note:
            Samples are not deterministic, and streamed/paginated queries or
            multiple workers will draw samples with replacement.
        """
        return self._evolve(query=self._query.sample(n))

    def filter(self, *args: Any) -> "Self":
        """Filter the chain according to conditions.

        Example:
            Basic usage with built-in operators
            ```py
            dc.filter(C("width") < 200)
            ```

            Using glob to match patterns
            ```py
            dc.filter(C("file.path").glob("*.jpg"))
            ```

            Using in to match lists
            ```py
            ids = [1,2,3]
            dc.filter(C("experiment_id").in_(ids))
            ```

            Using `datachain.func`
            ```py
            from datachain.func import string
            dc.filter(string.length(C("file.path")) > 5)
            ```

            Combining filters with "or"
            ```py
            dc.filter(
                C("file.path").glob("cat*") |
                C("file.path").glob("dog*")
            )
            ```

            ```py
            dc.filter(dc.func.or_(
                C("file.path").glob("cat*"),
                C("file.path").glob("dog*")
            ))
            ```

            Combining filters with "and"
            ```py
            dc.filter(
                C("file.path").glob("*.jpg"),
                string.length(C("file.path")) > 5
            )
            ```

            ```py
            dc.filter(
                C("file.path").glob("*.jpg") &
                (string.length(C("file.path")) > 5)
            )
            ```

            ```py
            dc.filter(dc.func.and_(
                C("file.path").glob("*.jpg"),
                string.length(C("file.path")) > 5
            ))
            ```

            Combining filters with "not"
            ```py
            dc.filter(~(C("file.path").glob("*.jpg")))
            ```
        """
        return self._evolve(query=self._query.filter(*args))

    def limit(self, n: int) -> "Self":
        """Return the first `n` rows of the chain.

        If the chain is unordered, which rows are returned is undefined.
        If the chain has less than `n` rows, the whole chain is returned.

        Parameters:
            n (int): Number of rows to return.
        """
        return self._evolve(query=self._query.limit(n))

    def offset(self, offset: int) -> "Self":
        """Return the results starting with the offset row.

        If the chain is unordered, which rows are skipped in undefined.
        If the chain has less than `offset` rows, the result is an empty chain.

        Parameters:
            offset (int): Number of rows to skip.
        """
        return self._evolve(query=self._query.offset(offset))

    def count(self) -> int:
        """Return the number of rows in the chain."""
        return self._query.count()

    def exec(self) -> "Self":
        """Execute the chain."""
        return self._evolve(query=self._query.exec())

    def chunk(self, index: int, total: int) -> "Self":
        """Split a chain into smaller chunks for e.g. parallelization.

        Parameters:
            index: The index of the chunk (0-indexed).
            total: The total number of chunks.

        Example:
            ```py
            import datachain as dc

            chain = dc.read_storage(...)
            chunk_1 = query._chunk(0, 2)
            chunk_2 = query._chunk(1, 2)
            ```

        Note:
            Bear in mind that `index` is 0-indexed but `total` isn't.
            Use 0/3, 1/3 and 2/3, not 1/3, 2/3 and 3/3.
        """
        return self._evolve(query=self._query.chunk(index, total))

    def to_list(self, *cols: str) -> list[tuple[DataValue, ...]]:
        """Returns a list of rows of values, optionally limited to the specified
        columns.

        Parameters:
            *cols: Limit to the specified columns. By default, all columns are selected.

        Returns:
            list[tuple[DataType, ...]]: Returns a list of tuples of items for each row.

        Example:
            Getting all rows as a list:
            ```py
            rows = dc.to_list()
            print(rows)
            ```

            Getting all rows with selected columns as a list:
            ```py
            name_size_pairs = dc.to_list("file.path", "file.size")
            print(name_size_pairs)
            ```

            Getting a single column as a list:
            ```py
            files = dc.to_list("file.path")
            print(files)  # Returns list of 1-tuples
            ```
        """
        return list(self.to_iter(*cols))

    def to_values(self, col: str) -> list[DataValue]:
        """Returns a flat list of values from a single column.

        Parameters:
            col: The name of the column to extract values from.

        Returns:
            list[DataValue]: Returns a flat list of values from the specified column.

        Example:
            Getting all values from a single column:
            ```py
            file_paths = dc.to_values("file.path")
            print(file_paths)  # Returns list of strings
            ```

            Getting all file sizes:
            ```py
            sizes = dc.to_values("file.size")
            print(sizes)  # Returns list of integers
            ```
        """
        return [row[0] for row in self.to_list(col)]

    def __iter__(self) -> Iterator[tuple[DataValue, ...]]:
        """Make DataChain objects iterable.

        Yields:
            (tuple[DataValue, ...]): Yields tuples of all column values for each row.

        Example:
            ```py
            for row in chain:
                print(row)
            ```
        """
        return self.to_iter()
