import copy
import logging
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta, timezone
from functools import cached_property, reduce
from itertools import groupby
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Table,
    Text,
    UniqueConstraint,
    and_,
    cast,
    desc,
    literal,
    or_,
    select,
)
from sqlalchemy.sql import func as f
from sqlalchemy.sql.selectable import Join

from datachain import json
from datachain.catalog.dependency import DatasetDependencyNode
from datachain.checkpoint import Checkpoint, CheckpointStatus
from datachain.checkpoint_event import (
    CheckpointEvent,
    CheckpointEventType,
    CheckpointStepType,
)
from datachain.data_storage import JobQueryType, JobStatus
from datachain.data_storage.serializer import Serializable
from datachain.dataset import (
    DatasetDependency,
    DatasetListRecord,
    DatasetListVersion,
    DatasetRecord,
    DatasetStatus,
    DatasetVersion,
    parse_schema,
)
from datachain.error import (
    CheckpointNotFoundError,
    DataChainError,
    DatasetNotFoundError,
    NamespaceDeleteNotAllowedError,
    NamespaceNotFoundError,
    ProjectDeleteNotAllowedError,
    ProjectNotFoundError,
    TableMissingError,
)
from datachain.job import Job
from datachain.namespace import Namespace
from datachain.project import Project

# Versions with no job_id (e.g. from pull_dataset) are only eligible
# for gc cleanup if they are older than this threshold, to avoid
# cleaning up in-flight operations.
STALE_CREATED_THRESHOLD_HOURS = 1

if TYPE_CHECKING:
    from sqlalchemy import CTE, Delete, Insert, Select, Subquery, Update
    from sqlalchemy.schema import SchemaItem
    from sqlalchemy.sql.elements import ColumnElement

    from datachain.data_storage import schema
    from datachain.data_storage.db_engine import DatabaseEngine

logger = logging.getLogger("datachain")
DEPTH_LIMIT_DEFAULT = 100
JOB_ANCESTRY_MAX_DEPTH = 100


class AbstractMetastore(ABC, Serializable):
    """
    Abstract Metastore class.
    This manages the storing, searching, and retrieval of indexed metadata.
    """

    schema: "schema.Schema"
    namespace_class: type[Namespace] = Namespace
    project_class: type[Project] = Project
    dataset_class: type[DatasetRecord] = DatasetRecord
    dataset_version_class: type[DatasetVersion] = DatasetVersion
    dataset_list_class: type[DatasetListRecord] = DatasetListRecord
    dataset_list_version_class: type[DatasetListVersion] = DatasetListVersion
    dependency_class: type[DatasetDependency] = DatasetDependency
    dependency_node_class: type[DatasetDependencyNode] = DatasetDependencyNode
    job_class: type[Job] = Job
    checkpoint_class: type[Checkpoint] = Checkpoint
    checkpoint_event_class: type[CheckpointEvent] = CheckpointEvent

    def __enter__(self) -> Self:
        """Returns self upon entering context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Default behavior is to do nothing, as connections may be shared."""

    @abstractmethod
    def clone(self, use_new_connection: bool = False) -> Self:
        """Clones AbstractMetastore implementation.
        Setting use_new_connection will always use a new database connection.
        New connections should only be used if needed due to errors with
        closed connections."""

    def close(self) -> None:
        """Closes any active database or HTTP connections."""

    def close_on_exit(self) -> None:
        """Closes any active database or HTTP connections, called on Session exit or
        for test cleanup only, as some Metastore implementations may handle this
        differently."""
        self.close()

    @contextmanager
    def _init_guard(self):
        """Ensure resources acquired during __init__ are released on failure."""
        try:
            yield
        except Exception:
            with suppress(Exception):
                self.close_on_exit()
            raise

    def cleanup_tables(self, temp_table_names: list[str]) -> None:
        """Cleanup temp tables."""

    def cleanup_for_tests(self) -> None:
        """Cleanup for tests."""

    #
    # Namespaces
    #

    @property
    @abstractmethod
    def default_namespace_name(self):
        """Gets default namespace name"""

    @property
    def system_namespace_name(self):
        return Namespace.system()

    @abstractmethod
    def create_namespace(
        self,
        name: str,
        description: str | None = None,
        uuid: str | None = None,
        ignore_if_exists: bool = True,
        validate: bool = True,
        **kwargs,
    ) -> Namespace:
        """Creates new namespace"""

    @abstractmethod
    def get_namespace(self, name: str) -> Namespace:
        """Gets a single namespace by name"""

    @abstractmethod
    def remove_namespace(self, namespace_id: int) -> None:
        """Removes a single namespace by id"""

    @abstractmethod
    def list_namespaces(self) -> list[Namespace]:
        """Gets a list of all namespaces"""

    #
    # Projects
    #

    @property
    @abstractmethod
    def default_project_name(self):
        """Gets default project name"""

    @property
    def listing_project_name(self):
        return Project.listing()

    @cached_property
    def default_project(self) -> Project:
        return self.get_project(
            self.default_project_name, self.default_namespace_name, create=True
        )

    @cached_property
    def listing_project(self) -> Project:
        return self.get_project(self.listing_project_name, self.system_namespace_name)

    @abstractmethod
    def create_project(
        self,
        namespace_name: str,
        name: str,
        description: str | None = None,
        uuid: str | None = None,
        ignore_if_exists: bool = True,
        validate: bool = True,
        **kwargs,
    ) -> Project:
        """Creates new project in specific namespace"""

    @abstractmethod
    def get_project(
        self, name: str, namespace_name: str, create: bool = False
    ) -> Project:
        """
        Gets a single project inside some namespace by name.
        It also creates project if not found and create flag is set to True.
        """

    def is_default_project(self, project_name: str, namespace_name: str) -> bool:
        return (
            project_name == self.default_project_name
            and namespace_name == self.default_namespace_name
        )

    def is_listing_project(self, project_name: str, namespace_name: str) -> bool:
        return (
            project_name == self.listing_project_name
            and namespace_name == self.system_namespace_name
        )

    @abstractmethod
    def get_project_by_id(self, project_id: int) -> Project:
        """Gets a single project by id"""

    @abstractmethod
    def count_projects(self, namespace_id: int | None = None) -> int:
        """Counts projects in some namespace or in general."""

    @abstractmethod
    def remove_project(self, project_id: int) -> None:
        """Removes a single project by id"""

    @abstractmethod
    def list_projects(self, namespace_id: int | None) -> list[Project]:
        """Gets list of projects in some namespace or in general (in all namespaces)"""

    #
    # Datasets
    #

    @abstractmethod
    def create_dataset(
        self,
        name: str,
        project_id: int | None = None,
        status: int = DatasetStatus.CREATED,
        sources: list[str] | None = None,
        feature_schema: dict | None = None,
        query_script: str = "",
        schema: dict[str, Any] | None = None,
        ignore_if_exists: bool = False,
        description: str | None = None,
        attrs: list[str] | None = None,
    ) -> DatasetRecord:
        """Creates new dataset."""

    @abstractmethod
    def create_dataset_version(  # noqa: PLR0913
        self,
        dataset: DatasetRecord,
        version: str,
        status: int,
        sources: str = "",
        feature_schema: dict | None = None,
        query_script: str = "",
        error_message: str = "",
        error_stack: str = "",
        script_output: str = "",
        created_at: datetime | None = None,
        finished_at: datetime | None = None,
        schema: dict[str, Any] | None = None,
        ignore_if_exists: bool = False,
        num_objects: int | None = None,
        size: int | None = None,
        preview: list[dict] | None = None,
        job_id: str | None = None,
        uuid: str | None = None,
    ) -> tuple[DatasetRecord, bool]:
        """Creates new dataset version.

        Returns:
            A tuple of (dataset_record, version_created) where version_created
            is True if this call actually created the version, False if the
            version already existed (only possible when ignore_if_exists=True).
        """

    @abstractmethod
    def remove_dataset(self, dataset: DatasetRecord) -> None:
        """Removes dataset."""

    @abstractmethod
    def update_dataset(self, dataset: DatasetRecord, **kwargs) -> DatasetRecord:
        """Updates dataset fields."""

    @abstractmethod
    def update_dataset_version(
        self, dataset: DatasetRecord, version: str, **kwargs
    ) -> DatasetVersion:
        """Updates dataset version fields."""

    @abstractmethod
    def remove_dataset_version(
        self, dataset: DatasetRecord, version: str
    ) -> DatasetRecord:
        """
        Deletes one single dataset version.
        If it was last version, it removes dataset completely.
        """

    @abstractmethod
    def get_dataset_versions_to_clean(
        self, job_id: str | None = None
    ) -> list[tuple[DatasetRecord, str]]:
        """
        Get incomplete, stale, or removed dataset versions to clean up.

        When job_id is provided, returns versions belonging to that specific
        job (used during job failure cleanup).

        When job_id is None, returns all versions that are safe to delete:
        - Status CREATED, FAILED, STALE where either:
          - the associated job has finished, or
          - there is no associated job (job_id is NULL) and the version is
            older than STALE_CREATED_THRESHOLD_HOURS
        - Status REMOVING: marked for deletion

        Returns:
            List of (DatasetRecord, version_string) tuples. Each DatasetRecord
            contains only one version (the incomplete version to clean).
        """

    @abstractmethod
    def get_dataset_versions(
        self, job_id: str | None = None, version_ids: list[int] | None = None
    ) -> list[tuple[DatasetRecord, str]]:
        """
        Get dataset versions by job ID and/or version IDs.

        Returns dataset versions filtered by job_id and/or version_ids:
        - If job_id is provided, returns versions for that job.
        - If version_ids are provided, returns versions matching those IDs.
        - If both are provided, returns versions matching both filters.
        - If neither is provided, returns all dataset versions.

        Returns:
            List of (DatasetRecord, version_string) tuples.
            Each DatasetRecord contains only one version.
        """

    @abstractmethod
    def list_datasets(
        self, project_id: int | None = None
    ) -> Iterator[DatasetListRecord]:
        """Lists all datasets in some project or in all projects."""

    @abstractmethod
    def count_datasets(self, project_id: int | None = None) -> int:
        """Counts datasets in some project or in all projects."""

    @abstractmethod
    def list_datasets_by_prefix(
        self,
        prefix: str,
        project_id: int | None = None,
        include_incomplete: bool = False,
    ) -> Iterator[DatasetListRecord]:
        """
        Lists all datasets which names start with prefix in some project or in all
        projects.
        """

    def get_dataset_by_version_uuid(
        self,
        uuid: str,
        include_incomplete: bool = False,
        *,
        include_preview: bool = False,
    ) -> DatasetRecord:
        """Gets a dataset that contains a version with the given UUID."""
        raise NotImplementedError

    @abstractmethod
    def get_dataset(
        self,
        name: str,  # normal, not full dataset name
        namespace_name: str | None = None,
        project_name: str | None = None,
        *,
        versions: Sequence[str] | None = (),
        include_incomplete: bool = True,
        include_preview: bool = False,
    ) -> DatasetRecord:
        """Gets a single dataset by name.

        Args:
            versions: Controls which dataset versions are loaded.
                - () (default): no versions are fetched (empty versions list).
                - None: all versions are fetched.
                - list[str]: only the specified versions are fetched.
        """

    @abstractmethod
    def update_dataset_status(
        self,
        dataset: DatasetRecord,
        status: int,
        version: str | None = None,
        error_message="",
        error_stack="",
        script_output="",
    ) -> DatasetRecord:
        """Updates dataset status and appropriate fields related to status."""

    @abstractmethod
    def mark_job_dataset_versions_as_failed(self, job_id: str) -> None:
        """
        Mark all non-COMPLETE dataset versions created by a job as FAILED.

        This is called when a job fails to ensure that any dataset versions
        it was creating are marked as failed rather than left in CREATED state.

        Args:
            job_id: ID of the failed job whose dataset versions should be marked
        """

    #
    # Dataset dependencies
    #

    @abstractmethod
    def add_dataset_dependency(
        self,
        source_dataset: DatasetRecord,
        source_dataset_version: str,
        dep_dataset: DatasetRecord,
        dep_dataset_version: str,
    ) -> None:
        """Adds dataset dependency to dataset."""

    @abstractmethod
    def update_dataset_dependency_source(
        self,
        source_dataset: DatasetRecord,
        source_dataset_version: str,
        new_source_dataset: DatasetRecord,
        new_source_dataset_version: str,
    ) -> None:
        """Updates dataset dependency source."""

    @abstractmethod
    def get_direct_dataset_dependencies(
        self, dataset: DatasetRecord, version: str
    ) -> list[DatasetDependency | None]:
        """Gets direct dataset dependencies."""

    @abstractmethod
    def get_dataset_dependency_nodes(
        self, dataset_id: int, version_id: int
    ) -> list[DatasetDependencyNode | None]:
        """Gets dataset dependency node from database."""

    @abstractmethod
    def remove_dataset_dependencies(
        self, dataset: DatasetRecord, version: str | None = None
    ) -> None:
        """
        When we remove dataset, we need to clean up it's dependencies as well.
        """

    @abstractmethod
    def remove_dataset_dependants(
        self, dataset: DatasetRecord, version: str | None = None
    ) -> None:
        """
        When we remove dataset, we need to clear its references in other dataset
        dependencies.
        """

    #
    # Jobs
    #

    @abstractmethod
    def list_jobs_by_ids(self, ids: list[str]) -> Iterator[Job]:
        """Lists jobs by ids."""

    @abstractmethod
    def create_job(
        self,
        name: str,
        query: str,
        query_type: JobQueryType = JobQueryType.PYTHON,
        status: JobStatus = JobStatus.CREATED,
        workers: int = 1,
        python_version: str | None = None,
        params: dict[str, str] | None = None,
        parent_job_id: str | None = None,
        rerun_from_job_id: str | None = None,
        run_group_id: str | None = None,
        is_remote_execution: bool = False,
        job_id: str | None = None,
    ) -> str:
        """
        Creates a new job.
        Returns the job id.
        """

    @abstractmethod
    def get_job(self, job_id: str) -> Job | None:
        """Returns the job with the given ID."""

    @abstractmethod
    def get_ancestor_job_ids(self, job_id: str) -> list[str]:
        """Returns list of ancestor job IDs in order from parent to root."""

    @abstractmethod
    def update_job(
        self,
        job_id: str,
        status: JobStatus | None = None,
        error_message: str | None = None,
        error_stack: str | None = None,
        finished_at: datetime | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> Job | None:
        """Updates job fields."""

    @abstractmethod
    def set_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: str | None = None,
        error_stack: str | None = None,
    ) -> None:
        """Set the status of the given job."""

    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus | None:
        """Returns the status of the given job."""

    @abstractmethod
    def get_last_job_by_name(
        self, name: str, is_remote_execution: bool = False
    ) -> Job | None:
        """Returns the last job with the given name, ordered by created_at."""

    #
    # Checkpoints
    #

    @abstractmethod
    def list_checkpoints(
        self,
        job_ids: list[str] | None = None,
        status: "CheckpointStatus | None" = None,
    ) -> Iterator[Checkpoint]:
        """List checkpoints, optionally filtered by job IDs and/or status."""

    @abstractmethod
    def get_last_checkpoint(self, job_id: str) -> Checkpoint | None:
        """Get last created checkpoint for some job."""

    @abstractmethod
    def get_checkpoint_by_id(self, checkpoint_id: str) -> Checkpoint:
        """Gets single checkpoint by id"""

    def find_checkpoint(
        self, job_id: str, _hash: str, partial: bool = False
    ) -> Checkpoint | None:
        """
        Tries to find checkpoint for a job with specific hash and optionally partial
        """

    @abstractmethod
    def get_or_create_checkpoint(
        self,
        job_id: str,
        _hash: str,
        partial: bool = False,
    ) -> Checkpoint:
        """Get or create checkpoint. Must be atomic and idempotent."""

    @abstractmethod
    def expire_checkpoints(
        self, ttl_threshold: datetime
    ) -> tuple[list[Checkpoint], list[str]]:
        """Atomically mark checkpoints as EXPIRED for finished jobs where ALL
        active checkpoints are older than ttl_threshold.

        Returns all EXPIRED checkpoints (including leftovers from prior
        crashed cleanup runs) and run group IDs where no ACTIVE checkpoints
        remain."""

    @abstractmethod
    def update_checkpoints(
        self,
        checkpoint_ids: list[str],
        status: "CheckpointStatus | None" = None,
    ) -> None:
        """Update checkpoint fields by IDs."""

    #
    # Checkpoint Events
    #

    @abstractmethod
    def log_checkpoint_event(  # noqa: PLR0913
        self,
        job_id: str,
        event_type: CheckpointEventType,
        step_type: CheckpointStepType,
        run_group_id: str | None = None,
        udf_name: str | None = None,
        dataset_name: str | None = None,
        checkpoint_hash: str | None = None,
        hash_partial: str | None = None,
        hash_input: str | None = None,
        hash_output: str | None = None,
        rows_input: int | None = None,
        rows_processed: int | None = None,
        rows_output: int | None = None,
        rows_input_reused: int | None = None,
        rows_output_reused: int | None = None,
        rerun_from_job_id: str | None = None,
        details: dict | None = None,
    ) -> CheckpointEvent:
        """Log a checkpoint event."""

    @abstractmethod
    def get_checkpoint_events(
        self,
        job_id: str | None = None,
        run_group_id: str | None = None,
    ) -> Iterator[CheckpointEvent]:
        """Get checkpoint events, optionally filtered by job_id or run_group_id."""

    #
    # UDF Registry (SaaS only, no-op for local metastores)
    #

    def add_udf(
        self,
        udf_id: str,
        name: str,
        status: str,
        rows_total: int,
        job_id: str,
        tasks_created: int,
        skipped: bool = False,
        continued: bool = False,
        rows_reused: int = 0,
        output_rows_reused: int = 0,
    ) -> None:
        """
        Register a UDF in the registry.
        No-op for local metastores, implemented in SaaS APIMetastore.
        """

    #
    # Dataset Version Jobs (many-to-many)
    #

    @abstractmethod
    def link_dataset_version_to_job(
        self,
        dataset_version_id: int,
        job_id: str,
        is_creator: bool = False,
    ) -> None:
        """Link dataset version to job. Must be atomic."""

    @abstractmethod
    def get_dataset_version_for_job_ancestry(
        self,
        dataset_name: str,
        namespace_name: str,
        project_name: str,
        job_id: str,
    ) -> DatasetVersion | None:
        """
        Find the dataset version that was created by any job in the ancestry.
        Returns the most recently linked version from these jobs.
        """


class AbstractDBMetastore(AbstractMetastore):
    """
    Abstract Database Metastore class, to be implemented
    by any Database Adapters for a specific database system.
    This manages the storing, searching, and retrieval of indexed metadata,
    and has shared logic for all database systems currently in use.
    """

    NAMESPACE_TABLE = "namespaces"
    PROJECT_TABLE = "projects"
    DATASET_TABLE = "datasets"
    DATASET_VERSION_TABLE = "datasets_versions"
    DATASET_DEPENDENCY_TABLE = "datasets_dependencies"
    DATASET_VERSION_JOBS_TABLE = "dataset_version_jobs"
    JOBS_TABLE = "jobs"
    CHECKPOINTS_TABLE = "checkpoints"
    CHECKPOINT_EVENTS_TABLE = "checkpoint_events"

    db: "DatabaseEngine"

    def close(self) -> None:
        """Closes any active database connections."""
        self.db.close()

    def cleanup_tables(self, temp_table_names: list[str]) -> None:
        """Cleanup temp tables."""

    @classmethod
    def _namespaces_columns(cls) -> list["SchemaItem"]:
        """Namespace table columns."""
        return [
            Column("id", Integer, primary_key=True),
            Column("uuid", Text, nullable=False, default=uuid4()),
            Column("name", Text, nullable=False),
            Column("description", Text),
            Column("created_at", DateTime(timezone=True)),
        ]

    @cached_property
    def _namespaces_fields(self) -> list[str]:
        return [
            c.name  # type: ignore [attr-defined]
            for c in self._namespaces_columns()
            if c.name  # type: ignore [attr-defined]
        ]

    @classmethod
    def _projects_columns(cls) -> list["SchemaItem"]:
        """Project table columns."""
        return [
            Column("id", Integer, primary_key=True),
            Column("uuid", Text, nullable=False, default=uuid4()),
            Column("name", Text, nullable=False),
            Column("description", Text),
            Column("created_at", DateTime(timezone=True)),
            Column(
                "namespace_id",
                Integer,
                ForeignKey(f"{cls.NAMESPACE_TABLE}.id", ondelete="CASCADE"),
                nullable=False,
            ),
            UniqueConstraint("namespace_id", "name"),
        ]

    @cached_property
    def _projects_fields(self) -> list[str]:
        return [
            c.name  # type: ignore [attr-defined]
            for c in self._projects_columns()
            if c.name  # type: ignore [attr-defined]
        ]

    @classmethod
    def _datasets_columns(cls) -> list["SchemaItem"]:
        """Datasets table columns."""
        return [
            Column("id", Integer, primary_key=True),
            Column(
                "project_id",
                Integer,
                ForeignKey(f"{cls.PROJECT_TABLE}.id", ondelete="CASCADE"),
                nullable=False,
            ),
            Column("name", Text, nullable=False),
            Column("description", Text),
            Column("attrs", JSON, nullable=True),
            Column("status", Integer, nullable=False),
            Column("feature_schema", JSON, nullable=True),
            Column("created_at", DateTime(timezone=True)),
            Column("finished_at", DateTime(timezone=True)),
            Column("error_message", Text, nullable=False, default=""),
            Column("error_stack", Text, nullable=False, default=""),
            Column("script_output", Text, nullable=False, default=""),
            Column("sources", Text, nullable=False, default=""),
            Column("query_script", Text, nullable=False, default=""),
            Column("schema", JSON, nullable=True),
        ]

    @cached_property
    def _dataset_fields(self) -> list[str]:
        return [
            c.name  # type: ignore [attr-defined]
            for c in self._datasets_columns()
            if c.name  # type: ignore [attr-defined]
        ]

    @cached_property
    def _dataset_list_fields(self) -> list[str]:
        return [
            c.name  # type: ignore [attr-defined]
            for c in self._datasets_columns()
            if c.name in self.dataset_list_class.__dataclass_fields__  # type: ignore [attr-defined]
        ]

    @classmethod
    def _datasets_versions_columns(cls) -> list["SchemaItem"]:
        """Datasets versions table columns."""
        return [
            Column("id", Integer, primary_key=True),
            Column("uuid", Text, nullable=False, default=uuid4()),
            Column(
                "dataset_id",
                Integer,
                ForeignKey(f"{cls.DATASET_TABLE}.id", ondelete="CASCADE"),
                nullable=False,
            ),
            Column("version", Text, nullable=False, default="1.0.0"),
            Column(
                "status",
                Integer,
                nullable=False,
            ),
            Column("feature_schema", JSON, nullable=True),
            Column("created_at", DateTime(timezone=True)),
            Column("finished_at", DateTime(timezone=True)),
            Column("error_message", Text, nullable=False, default=""),
            Column("error_stack", Text, nullable=False, default=""),
            Column("script_output", Text, nullable=False, default=""),
            Column("num_objects", BigInteger, nullable=True),
            Column("size", BigInteger, nullable=True),
            Column("preview", JSON, nullable=True),
            Column("sources", Text, nullable=False, default=""),
            Column("query_script", Text, nullable=False, default=""),
            Column("schema", JSON, nullable=True),
            Column("job_id", Text, nullable=True),
            UniqueConstraint("dataset_id", "version"),
        ]

    @cached_property
    def _dataset_version_fields(self) -> list[str]:
        return [
            c.name  # type: ignore [attr-defined]
            for c in self._datasets_versions_columns()
            if c.name  # type: ignore [attr-defined]
        ]

    @cached_property
    def _dataset_list_version_fields(self) -> list[str]:
        return [
            c.name  # type: ignore [attr-defined]
            for c in self._datasets_versions_columns()
            if c.name  # type: ignore [attr-defined]
            in self.dataset_list_version_class.__dataclass_fields__
        ]

    @classmethod
    def _datasets_dependencies_columns(cls) -> list["SchemaItem"]:
        """Datasets dependencies table columns."""
        return [
            Column("id", Integer, primary_key=True),
            # TODO remove when https://github.com/iterative/dvcx/issues/959 is done
            Column(
                "source_dataset_id",
                Integer,
                ForeignKey(f"{cls.DATASET_TABLE}.id"),
                nullable=False,
            ),
            Column(
                "source_dataset_version_id",
                Integer,
                ForeignKey(f"{cls.DATASET_VERSION_TABLE}.id"),
                nullable=True,
            ),
            # TODO remove when https://github.com/iterative/dvcx/issues/959 is done
            Column(
                "dataset_id",
                Integer,
                ForeignKey(f"{cls.DATASET_TABLE}.id"),
                nullable=True,
            ),
            Column(
                "dataset_version_id",
                Integer,
                ForeignKey(f"{cls.DATASET_VERSION_TABLE}.id"),
                nullable=True,
            ),
        ]

    #
    # Query Tables
    #

    @cached_property
    def _namespaces(self) -> Table:
        return Table(
            self.NAMESPACE_TABLE, self.db.metadata, *self._namespaces_columns()
        )

    @cached_property
    def _projects(self) -> Table:
        return Table(self.PROJECT_TABLE, self.db.metadata, *self._projects_columns())

    @cached_property
    def _datasets(self) -> Table:
        return Table(self.DATASET_TABLE, self.db.metadata, *self._datasets_columns())

    @cached_property
    def _datasets_versions(self) -> Table:
        return Table(
            self.DATASET_VERSION_TABLE,
            self.db.metadata,
            *self._datasets_versions_columns(),
        )

    @cached_property
    def _datasets_dependencies(self) -> Table:
        return Table(
            self.DATASET_DEPENDENCY_TABLE,
            self.db.metadata,
            *self._datasets_dependencies_columns(),
        )

    #
    # Query Starters (These can be overridden by subclasses)
    #

    @abstractmethod
    def _namespaces_insert(self) -> "Insert": ...

    def _namespaces_select(self, *columns) -> "Select":
        if not columns:
            return self._namespaces.select()
        return select(*columns)

    def _namespaces_update(self) -> "Update":
        return self._namespaces.update()

    def _namespaces_delete(self) -> "Delete":
        return self._namespaces.delete()

    @abstractmethod
    def _projects_insert(self) -> "Insert": ...

    def _projects_select(self, *columns) -> "Select":
        if not columns:
            return self._projects.select()
        return select(*columns)

    def _projects_delete(self) -> "Delete":
        return self._projects.delete()

    @abstractmethod
    def _datasets_insert(self) -> "Insert": ...

    def _datasets_select(self, *columns) -> "Select":
        if not columns:
            return self._datasets.select()
        return select(*columns)

    def _datasets_update(self) -> "Update":
        return self._datasets.update()

    def _datasets_delete(self) -> "Delete":
        return self._datasets.delete()

    @abstractmethod
    def _datasets_versions_insert(self) -> "Insert": ...

    def _datasets_versions_select(self, *columns) -> "Select":
        if not columns:
            return self._datasets_versions.select()
        return select(*columns)

    def _datasets_versions_update(self) -> "Update":
        return self._datasets_versions.update()

    def _datasets_versions_delete(self) -> "Delete":
        return self._datasets_versions.delete()

    @abstractmethod
    def _datasets_dependencies_insert(self) -> "Insert": ...

    def _datasets_dependencies_select(self, *columns) -> "Select":
        if not columns:
            return self._datasets_dependencies.select()
        return select(*columns)

    def _datasets_dependencies_update(self) -> "Update":
        return self._datasets_dependencies.update()

    def _datasets_dependencies_delete(self) -> "Delete":
        return self._datasets_dependencies.delete()

    #
    # Namespaces
    #

    def create_namespace(
        self,
        name: str,
        description: str | None = None,
        uuid: str | None = None,
        ignore_if_exists: bool = True,
        validate: bool = True,
        **kwargs,
    ) -> Namespace:
        if validate:
            Namespace.validate_name(name)
        query = self._namespaces_insert().values(
            name=name,
            uuid=uuid or str(uuid4()),
            created_at=datetime.now(timezone.utc),
            description=description,
        )
        if ignore_if_exists and hasattr(query, "on_conflict_do_nothing"):
            # SQLite and PostgreSQL both support 'on_conflict_do_nothing',
            # but generic SQL does not
            query = query.on_conflict_do_nothing(index_elements=["name"])
        self.db.execute(query)

        return self.get_namespace(name)

    def remove_namespace(self, namespace_id: int) -> None:
        num_projects = self.count_projects(namespace_id)
        if num_projects > 0:
            raise NamespaceDeleteNotAllowedError(
                f"Namespace cannot be removed. It contains {num_projects} project(s). "
                "Please remove the project(s) first."
            )

        n = self._namespaces
        with self.db.transaction():
            self.db.execute(self._namespaces_delete().where(n.c.id == namespace_id))

    def get_namespace(self, name: str) -> Namespace:
        """Gets a single namespace by name"""
        n = self._namespaces

        query = self._namespaces_select(
            *(getattr(n.c, f) for f in self._namespaces_fields),
        ).where(n.c.name == name)
        rows = list(self.db.execute(query))
        if not rows:
            raise NamespaceNotFoundError(f"Namespace {name} not found.")
        return self.namespace_class.parse(*rows[0])

    def list_namespaces(self) -> list[Namespace]:
        """Gets a list of all namespaces"""
        n = self._namespaces

        query = self._namespaces_select(
            *(getattr(n.c, f) for f in self._namespaces_fields),
        )
        rows = list(self.db.execute(query))

        return [self.namespace_class.parse(*r) for r in rows]

    #
    # Projects
    #

    def create_project(
        self,
        namespace_name: str,
        name: str,
        description: str | None = None,
        uuid: str | None = None,
        ignore_if_exists: bool = True,
        validate: bool = True,
        **kwargs,
    ) -> Project:
        if validate:
            Project.validate_name(name)
        try:
            namespace = self.get_namespace(namespace_name)
        except NamespaceNotFoundError:
            namespace = self.create_namespace(namespace_name, validate=validate)

        query = self._projects_insert().values(
            namespace_id=namespace.id,
            uuid=uuid or str(uuid4()),
            name=name,
            created_at=datetime.now(timezone.utc),
            description=description,
        )
        if ignore_if_exists and hasattr(query, "on_conflict_do_nothing"):
            # SQLite and PostgreSQL both support 'on_conflict_do_nothing',
            # but generic SQL does not
            query = query.on_conflict_do_nothing(
                index_elements=["namespace_id", "name"]
            )
        self.db.execute(query)

        return self.get_project(name, namespace.name)

    def _projects_base_query(self) -> "Select":
        n = self._namespaces
        p = self._projects

        query = self._projects_select(
            *(getattr(n.c, f) for f in self._namespaces_fields),
            *(getattr(p.c, f) for f in self._projects_fields),
        )
        return query.select_from(n.join(p, n.c.id == p.c.namespace_id))

    def get_project(
        self, name: str, namespace_name: str, create: bool = False
    ) -> Project:
        """Gets a single project inside some namespace by name"""
        n = self._namespaces
        p = self._projects
        validate = True

        if self.is_listing_project(name, namespace_name) or self.is_default_project(
            name, namespace_name
        ):
            # we are always creating default and listing projects if they don't exist
            create = True
            validate = False

        query = self._projects_base_query().where(
            p.c.name == name, n.c.name == namespace_name
        )

        rows = list(self.db.execute(query))
        if not rows:
            if create:
                return self.create_project(namespace_name, name, validate=validate)
            raise ProjectNotFoundError(
                f"Project {name} in namespace {namespace_name} not found."
            )
        return self.project_class.parse(*rows[0])

    def get_project_by_id(self, project_id: int) -> Project:
        """Gets a single project by id"""
        p = self._projects

        query = self._projects_base_query().where(p.c.id == project_id)

        rows = list(self.db.execute(query))
        if not rows:
            raise ProjectNotFoundError(f"Project with id {project_id} not found.")
        return self.project_class.parse(*rows[0])

    def count_projects(self, namespace_id: int | None = None) -> int:
        p = self._projects

        query = self._projects_base_query()
        if namespace_id:
            query = query.where(p.c.namespace_id == namespace_id)

        query = select(f.count(1)).select_from(query.subquery())

        return next(self.db.execute(query))[0]

    def remove_project(self, project_id: int) -> None:
        num_datasets = self.count_datasets(project_id)
        if num_datasets > 0:
            raise ProjectDeleteNotAllowedError(
                f"Project cannot be removed. It contains {num_datasets} dataset(s). "
                "Please remove the dataset(s) first."
            )

        p = self._projects
        with self.db.transaction():
            self.db.execute(self._projects_delete().where(p.c.id == project_id))

    def list_projects(self, namespace_id: int | None = None) -> list[Project]:
        """
        Gets a list of projects inside some namespace, or in all namespaces
        """
        p = self._projects

        query = self._projects_base_query()

        if namespace_id:
            query = query.where(p.c.namespace_id == namespace_id)

        rows = list(self.db.execute(query))

        return [self.project_class.parse(*r) for r in rows]

    #
    # Datasets
    #

    def create_dataset(
        self,
        name: str,
        project_id: int | None = None,
        status: int = DatasetStatus.CREATED,
        sources: list[str] | None = None,
        feature_schema: dict | None = None,
        query_script: str = "",
        schema: dict[str, Any] | None = None,
        ignore_if_exists: bool = False,
        description: str | None = None,
        attrs: list[str] | None = None,
        **kwargs,  # TODO registered = True / False
    ) -> DatasetRecord:
        """Creates new dataset."""
        if not project_id:
            project = self.default_project
        else:
            project = self.get_project_by_id(project_id)

        query = self._datasets_insert().values(
            name=name,
            project_id=project.id,
            status=status,
            feature_schema=json.dumps(feature_schema or {}),
            created_at=datetime.now(timezone.utc),
            error_message="",
            error_stack="",
            script_output="",
            sources="\n".join(sources) if sources else "",
            query_script=query_script,
            schema=json.dumps(schema or {}),
            description=description,
            attrs=json.dumps(attrs or []),
        )
        if ignore_if_exists and hasattr(query, "on_conflict_do_nothing"):
            # SQLite and PostgreSQL both support 'on_conflict_do_nothing',
            # but generic SQL does not
            query = query.on_conflict_do_nothing(index_elements=["project_id", "name"])
        self.db.execute(query)

        return self.get_dataset(
            name,
            namespace_name=project.namespace.name,
            project_name=project.name,
            versions=None,
            include_incomplete=True,
        )

    def create_dataset_version(  # noqa: PLR0913
        self,
        dataset: DatasetRecord,
        version: str,
        status: int,
        sources: str = "",
        feature_schema: dict | None = None,
        query_script: str = "",
        error_message: str = "",
        error_stack: str = "",
        script_output: str = "",
        created_at: datetime | None = None,
        finished_at: datetime | None = None,
        schema: dict[str, Any] | None = None,
        ignore_if_exists: bool = False,
        num_objects: int | None = None,
        size: int | None = None,
        preview: list[dict] | None = None,
        job_id: str | None = None,
        uuid: str | None = None,
    ) -> tuple[DatasetRecord, bool]:
        """Creates new dataset version.

        Returns:
            A tuple of (dataset_record, version_created) where version_created
            is True if this call actually created the version, False if the
            version already existed (only possible when ignore_if_exists=True).
        """
        if status in [DatasetStatus.COMPLETE, DatasetStatus.FAILED]:
            finished_at = finished_at or datetime.now(timezone.utc)
        else:
            finished_at = None

        my_uuid = uuid or str(uuid4())

        query = self._datasets_versions_insert().values(
            dataset_id=dataset.id,
            uuid=my_uuid,
            version=version,
            status=status,
            feature_schema=json.dumps(feature_schema or {}),
            created_at=created_at or datetime.now(timezone.utc),
            finished_at=finished_at,
            error_message=error_message,
            error_stack=error_stack,
            script_output=script_output,
            sources=sources,
            query_script=query_script,
            schema=json.dumps(schema or {}),
            num_objects=num_objects,
            size=size,
            preview=json.dumps(preview or []),
            job_id=job_id or os.getenv("DATACHAIN_JOB_ID"),
        )
        if ignore_if_exists and hasattr(query, "on_conflict_do_nothing"):
            # SQLite and PostgreSQL both support 'on_conflict_do_nothing',
            # but generic SQL does not
            query = query.on_conflict_do_nothing(
                index_elements=["dataset_id", "version"]
            )
        self.db.execute(query)

        dataset = self.get_dataset(
            dataset.name,
            namespace_name=dataset.project.namespace.name,
            project_name=dataset.project.name,
            versions=None,
            include_incomplete=True,
        )

        # Detect whether this call actually created the version by comparing
        # the UUID we attempted to insert with the one stored in the DB.
        # If another writer won the ON CONFLICT race, the stored UUID will
        # differ from ours.
        dataset_version = dataset.get_version(version)
        version_created = dataset_version.uuid == my_uuid

        return dataset, version_created

    def remove_dataset(self, dataset: DatasetRecord) -> None:
        """Removes dataset."""
        d = self._datasets
        with self.db.transaction():
            self.remove_dataset_dependencies(dataset)
            self.remove_dataset_dependants(dataset)
            self.db.execute(self._datasets_delete().where(d.c.id == dataset.id))

    def update_dataset(self, dataset: DatasetRecord, **kwargs) -> DatasetRecord:
        """Updates dataset fields."""
        values: dict[str, Any] = {}
        dataset_values: dict[str, Any] = {}
        for field, value in kwargs.items():
            if field in ("id", "created_at") or field not in self._dataset_fields:
                continue  # these fields are read-only or not applicable

            if value is None and field in ("name", "status", "sources", "query_script"):
                raise ValueError(f"Field {field} cannot be None")
            if field == "name" and not value:
                raise ValueError("name cannot be empty")

            if field == "attrs":
                if value is None:
                    values[field] = None
                else:
                    values[field] = json.dumps(value)
                dataset_values[field] = value
            elif field == "schema":
                if value is None:
                    values[field] = None
                    dataset_values[field] = None
                else:
                    values[field] = json.dumps(value)
                    dataset_values[field] = parse_schema(value)
            elif field == "project_id":
                if not value:
                    raise ValueError("Cannot set empty project_id for dataset")
                dataset_values["project"] = self.get_project_by_id(value)
                values[field] = value
            else:
                values[field] = value
                dataset_values[field] = value

        if not values:
            return dataset  # nothing to update

        d = self._datasets
        self.db.execute(
            self._datasets_update()
            .where(d.c.name == dataset.name, d.c.project_id == dataset.project.id)
            .values(values),
        )  # type: ignore [attr-defined]

        result_ds = copy.deepcopy(dataset)
        result_ds.update(**dataset_values)
        return result_ds

    def update_dataset_version(
        self, dataset: DatasetRecord, version: str, **kwargs
    ) -> DatasetVersion:
        """Updates dataset fields."""
        logger.debug(
            "Metastore.update_dataset_version called for %s@%s: "
            "num_objects=%s, size=%s, preview_len=%s, all_fields=%s",
            dataset.name,
            version,
            kwargs.get("num_objects"),
            kwargs.get("size"),
            len(kwargs["preview"])
            if "preview" in kwargs and kwargs["preview"] is not None
            else None,
            list(kwargs.keys()),
        )
        values: dict[str, Any] = {}
        version_values: dict[str, Any] = {}
        for field, value in kwargs.items():
            if (
                field in ("id", "created_at")
                or field not in self._dataset_version_fields
            ):
                continue  # these fields are read-only or not applicable

            if value is None and field in (
                "status",
                "sources",
                "query_script",
                "error_message",
                "error_stack",
                "script_output",
                "uuid",
            ):
                raise ValueError(f"Field {field} cannot be None")

            if field == "schema":
                values[field] = json.dumps(value) if value else None
                version_values[field] = parse_schema(value) if value else None
            elif field == "feature_schema":
                if value is None:
                    values[field] = None
                else:
                    values[field] = json.dumps(value)
                version_values[field] = value
            elif field == "preview":
                if value is None:
                    values[field] = None
                elif not isinstance(value, list):
                    raise ValueError(
                        f"Field '{field}' must be a list, got {type(value).__name__}"
                    )
                else:
                    values[field] = json.dumps(value, serialize_bytes=True)
                version_values["_preview_data"] = value
                version_values["_preview_loaded"] = True
            else:
                values[field] = value
                version_values[field] = value

        dataset_version = dataset.get_version(version)

        if not values:
            return dataset_version

        logger.debug(
            "Writing to database for %s@%s: num_objects=%s, size=%s, "
            "preview_serialized=%s, fields_to_update=%s",
            dataset.name,
            version,
            values.get("num_objects"),
            values.get("size"),
            bool(values.get("preview")),
            list(values.keys()),
        )

        dv = self._datasets_versions
        self.db.execute(
            self._datasets_versions_update()
            .where(dv.c.dataset_id == dataset.id, dv.c.version == version)
            .values(values),
        )  # type: ignore [attr-defined]

        dataset_version.update(**version_values)
        logger.debug(
            "Dataset version updated successfully: %s@%s, "
            "final_num_objects=%s, final_size=%s, has_preview=%s",
            dataset.name,
            version,
            dataset_version.num_objects,
            dataset_version.size,
            bool(getattr(dataset_version, "_preview_data", None)),
        )
        return dataset_version

    def _parse_dataset(
        self,
        rows,
        *,
        versions_loaded: bool,
        preview_loaded: bool,
    ) -> DatasetRecord | None:
        parse_dataset: Any = self.dataset_class.parse
        versions = [
            parse_dataset(
                *r,
                versions_loaded=versions_loaded,
                preview_loaded=preview_loaded,
            )
            for r in rows
        ]
        if not versions:
            return None
        return reduce(lambda ds, version: ds.merge_versions(version), versions)

    def _parse_list_dataset(self, rows) -> DatasetListRecord | None:
        versions = [self.dataset_list_class.parse(*r) for r in rows]
        if not versions:
            return None
        return reduce(lambda ds, version: ds.merge_versions(version), versions)

    def _parse_dataset_list(self, rows) -> Iterator["DatasetListRecord"]:
        # grouping rows by dataset id
        for _, g in groupby(rows, lambda r: r[11]):
            dataset = self._parse_list_dataset(list(g))
            if dataset:
                yield dataset

    def _get_dataset_query(
        self,
        namespace_fields: list[str],
        project_fields: list[str],
        dataset_fields: list[str],
        dataset_version_fields: list[str],
        isouter: bool = True,
        include_incomplete: bool = True,
        include_preview: bool = False,
        versions: Sequence[str] | None = None,
    ) -> "Select":
        if not (
            self.db.has_table(self._datasets.name)
            and self.db.has_table(self._datasets_versions.name)
        ):
            raise TableMissingError

        n = self._namespaces
        p = self._projects
        d = self._datasets
        dv = self._datasets_versions

        if versions is not None and not versions:
            query = self._datasets_select(
                *(getattr(n.c, f) for f in namespace_fields),
                *(getattr(p.c, f) for f in project_fields),
                *(getattr(d.c, f) for f in dataset_fields),
            )
            j = n.join(p, n.c.id == p.c.namespace_id).join(d, p.c.id == d.c.project_id)
            query = query.select_from(j)
            if not include_incomplete:
                query = query.where(
                    select(literal(1))
                    .where(
                        and_(
                            dv.c.dataset_id == d.c.id,
                            dv.c.status == DatasetStatus.COMPLETE,
                        )
                    )
                    .exists()
                )
            return query

        version_columns = []
        for field in dataset_version_fields:
            if field == "preview" and not include_preview:
                version_columns.append(literal(None).label("preview"))
                continue
            version_columns.append(getattr(dv.c, field))

        query = self._datasets_select(
            *(getattr(n.c, f) for f in namespace_fields),
            *(getattr(p.c, f) for f in project_fields),
            *(getattr(d.c, f) for f in dataset_fields),
            *version_columns,
        )

        # Build join condition with status filter
        join_condition = d.c.id == dv.c.dataset_id
        if not include_incomplete:
            # Only include COMPLETE dataset versions (hide CREATED/FAILED)
            join_condition = and_(join_condition, dv.c.status == DatasetStatus.COMPLETE)

        j = (
            n.join(p, n.c.id == p.c.namespace_id)
            .join(d, p.c.id == d.c.project_id)
            .join(dv, join_condition, isouter=isouter)
        )
        query = query.select_from(j)

        if versions is not None:
            query = query.where(dv.c.version.in_(versions))

        return query

    def _base_dataset_query(
        self,
        include_incomplete: bool = True,
        *,
        include_preview: bool = False,
        versions: Sequence[str] | None = None,
    ) -> "Select":
        # When filtering by status, use inner join so datasets without COMPLETE
        # versions are excluded
        isouter = include_incomplete
        return self._get_dataset_query(
            self._namespaces_fields,
            self._projects_fields,
            self._dataset_fields,
            self._dataset_version_fields,
            isouter=isouter,
            include_incomplete=include_incomplete,
            include_preview=include_preview,
            versions=versions,
        )

    def _base_list_datasets_query(self, include_incomplete: bool = True) -> "Select":
        return self._get_dataset_query(
            self._namespaces_fields,
            self._projects_fields,
            self._dataset_list_fields,
            self._dataset_list_version_fields,
            isouter=False,
            include_incomplete=include_incomplete,
        )

    def list_datasets(
        self, project_id: int | None = None
    ) -> Iterator["DatasetListRecord"]:
        d = self._datasets
        query = self._base_list_datasets_query(include_incomplete=False).order_by(
            self._datasets.c.name, self._datasets_versions.c.version
        )
        if project_id:
            query = query.where(d.c.project_id == project_id)
        yield from self._parse_dataset_list(self.db.execute(query))

    def count_datasets(self, project_id: int | None = None) -> int:
        d = self._datasets
        query = self._datasets_select()
        if project_id:
            query = query.where(d.c.project_id == project_id)

        query = select(f.count(1)).select_from(query.subquery())

        return next(self.db.execute(query))[0]

    def list_datasets_by_prefix(
        self,
        prefix: str,
        project_id: int | None = None,
        include_incomplete: bool = False,
    ) -> Iterator["DatasetListRecord"]:
        d = self._datasets
        query = self._base_list_datasets_query(include_incomplete=include_incomplete)
        if project_id:
            query = query.where(d.c.project_id == project_id)
        query = query.where(self._datasets.c.name.startswith(prefix))
        yield from self._parse_dataset_list(self.db.execute(query))

    def get_dataset_by_version_uuid(
        self,
        uuid: str,
        include_incomplete: bool = False,
        *,
        include_preview: bool = False,
    ) -> DatasetRecord:
        """Gets a dataset that contains a version with the given UUID."""
        dv = self._datasets_versions
        query = self._base_dataset_query(
            include_incomplete=include_incomplete,
            include_preview=include_preview,
        )
        query = query.where(dv.c.uuid == uuid)
        ds = self._parse_dataset(
            self.db.execute(query),
            versions_loaded=True,
            preview_loaded=include_preview,
        )
        if not ds:
            raise DatasetNotFoundError(f"Dataset with version uuid {uuid} not found.")
        return ds

    def get_dataset(
        self,
        name: str,  # normal, not full dataset name
        namespace_name: str | None = None,
        project_name: str | None = None,
        *,
        versions: Sequence[str] | None = (),
        include_incomplete: bool = True,
        include_preview: bool = False,
    ) -> DatasetRecord:
        """
        Gets a single dataset in project by dataset name.
        """
        namespace_name = namespace_name or self.default_namespace_name
        project_name = project_name or self.default_project_name

        d = self._datasets
        n = self._namespaces
        p = self._projects

        query = self._base_dataset_query(
            include_incomplete=include_incomplete,
            include_preview=include_preview,
            versions=versions,
        )
        query = query.where(
            d.c.name == name,
            n.c.name == namespace_name,
            p.c.name == project_name,
        )  # type: ignore [attr-defined]

        ds = self._parse_dataset(
            self.db.execute(query),
            versions_loaded=versions is None or bool(versions),
            preview_loaded=include_preview,
        )
        if not ds:
            raise DatasetNotFoundError(
                f"Dataset {name} not found in namespace {namespace_name}"
                f" and project {project_name}"
            )

        return ds

    def remove_dataset_version(
        self, dataset: DatasetRecord, version: str
    ) -> DatasetRecord:
        """
        Deletes one single dataset version.
        If it was last version, it removes dataset completely
        """
        if not dataset.has_version(version):
            raise DatasetNotFoundError(
                f"Dataset {dataset.name} version {version} not found."
            )

        d = self._datasets
        dv = self._datasets_versions

        with self.db.transaction():
            self.remove_dataset_dependencies(dataset, version)
            self.remove_dataset_dependants(dataset, version)

            self.db.execute(
                self._datasets_versions_delete().where(
                    (dv.c.dataset_id == dataset.id) & (dv.c.version == version)
                )
            )

            if dataset.versions and len(dataset.versions) == 1:
                # had only one version, fully deleting dataset
                self.db.execute(self._datasets_delete().where(d.c.id == dataset.id))

        dataset.remove_version(version)
        return dataset

    def _dataset_version_query_base(
        self,
    ) -> tuple[tuple["ColumnElement[Any]", ...], Join]:
        n = self._namespaces
        p = self._projects
        d = self._datasets
        dv = self._datasets_versions

        select_cols = (
            *(getattr(n.c, f) for f in self._namespaces_fields),
            *(getattr(p.c, f) for f in self._projects_fields),
            *(getattr(d.c, f) for f in self._dataset_fields),
            *(
                literal(None).label("preview") if f == "preview" else getattr(dv.c, f)
                for f in self._dataset_version_fields
            ),
        )
        base_from = (
            n.join(p, n.c.id == p.c.namespace_id)
            .join(d, p.c.id == d.c.project_id)
            .join(dv, d.c.id == dv.c.dataset_id)
        )
        return select_cols, base_from

    def _fetch_version_pairs(self, query) -> list[tuple[DatasetRecord, str]]:
        results: list[tuple[DatasetRecord, str]] = []
        for row in self.db.execute(query):
            dataset = self.dataset_class.parse(*row, preview_loaded=False)
            if dataset.versions:
                results.append((dataset, dataset.versions[0].version))
        return results

    def get_dataset_versions_to_clean(
        self, job_id: str | None = None
    ) -> list[tuple[DatasetRecord, str]]:
        select_cols, base_from = self._dataset_version_query_base()
        d = self._datasets
        dv = self._datasets_versions
        j = self._jobs

        # LEFT JOIN on jobs so versions with job_id=NULL are included.
        # Only skip versions whose job is still running.
        query = (
            self._datasets_select(*select_cols)
            .select_from(
                base_from.join(
                    j,
                    cast(dv.c.job_id, j.c.id.type) == j.c.id,
                    isouter=True,
                )
            )
            .where(
                or_(
                    # Incomplete/failed/stale versions from finished jobs
                    and_(
                        dv.c.status.in_(
                            [
                                DatasetStatus.CREATED,
                                DatasetStatus.FAILED,
                                DatasetStatus.STALE,
                                DatasetStatus.REMOVING,
                            ]
                        ),
                        or_(
                            j.c.status.in_(
                                [
                                    JobStatus.COMPLETE,
                                    JobStatus.FAILED,
                                    JobStatus.CANCELED,
                                ]
                            ),
                            and_(
                                dv.c.job_id.is_(None),
                                dv.c.created_at
                                < datetime.now(timezone.utc)
                                - timedelta(hours=STALE_CREATED_THRESHOLD_HOURS),
                            ),
                        ),
                    ),
                    # Session datasets from finished jobs (orphaned intermediates)
                    and_(
                        d.c.name.startswith("session_"),
                        dv.c.status == DatasetStatus.COMPLETE,
                        dv.c.job_id.isnot(None),
                        j.c.status.in_(
                            [
                                JobStatus.COMPLETE,
                                JobStatus.FAILED,
                                JobStatus.CANCELED,
                            ]
                        ),
                    ),
                ),
            )
        )

        if job_id:
            query = query.where(dv.c.job_id == job_id)

        return self._fetch_version_pairs(query)

    def get_dataset_versions(
        self, job_id: str | None = None, version_ids: list[int] | None = None
    ) -> list[tuple[DatasetRecord, str]]:
        select_cols, base_from = self._dataset_version_query_base()
        dv = self._datasets_versions

        query = self._datasets_select(*select_cols).select_from(base_from)

        if job_id is not None:
            query = query.where(dv.c.job_id == job_id)

        if version_ids is not None:
            query = query.where(dv.c.id.in_(version_ids))

        return self._fetch_version_pairs(query)

    def update_dataset_status(
        self,
        dataset: DatasetRecord,
        status: int,
        version: str | None = None,
        error_message="",
        error_stack="",
        script_output="",
    ) -> DatasetRecord:
        """
        Updates dataset status and appropriate fields related to status
        It also updates version if specified.
        """
        update_data: dict[str, Any] = {"status": status}
        if status in [DatasetStatus.COMPLETE, DatasetStatus.FAILED]:
            # if in final state, updating finished_at datetime
            update_data["finished_at"] = datetime.now(timezone.utc)
            if script_output:
                update_data["script_output"] = script_output

        if status == DatasetStatus.FAILED:
            update_data["error_message"] = error_message
            update_data["error_stack"] = error_stack

        dataset = self.update_dataset(dataset, **update_data)

        if version:
            self.update_dataset_version(dataset, version, **update_data)

        return dataset

    def mark_job_dataset_versions_as_failed(self, job_id: str) -> None:
        """
        Mark all non-COMPLETE dataset versions created by a job as FAILED.

        This is called when a job fails to ensure that any dataset versions
        it was creating are marked as failed rather than left in CREATED state.

        Args:
            job_id: ID of the failed job whose dataset versions should be marked
        """
        dv = self._datasets_versions

        # Update status to FAILED for all non-COMPLETE versions with this job_id
        update_stmt = (
            dv.update()
            .where((dv.c.job_id == job_id) & (dv.c.status != DatasetStatus.COMPLETE))
            .values(
                status=DatasetStatus.FAILED,
                finished_at=datetime.now(timezone.utc),
            )
        )

        self.db.execute(update_stmt)

    #
    # Dataset dependencies
    #

    def add_dataset_dependency(
        self,
        source_dataset: "DatasetRecord",
        source_dataset_version: str,
        dep_dataset: "DatasetRecord",
        dep_dataset_version: str,
    ) -> None:
        """Adds dataset dependency to dataset."""
        self.db.execute(
            self._datasets_dependencies_insert().values(
                source_dataset_id=source_dataset.id,
                source_dataset_version_id=(
                    source_dataset.get_version(source_dataset_version).id
                ),
                dataset_id=dep_dataset.id,
                dataset_version_id=dep_dataset.get_version(dep_dataset_version).id,
            )
        )

    def update_dataset_dependency_source(
        self,
        source_dataset: DatasetRecord,
        source_dataset_version: str,
        new_source_dataset: DatasetRecord,
        new_source_dataset_version: str,
    ) -> None:
        dd = self._datasets_dependencies

        q = self._datasets_dependencies_update().where(
            dd.c.source_dataset_id == source_dataset.id
        )
        q = q.where(
            dd.c.source_dataset_version_id
            == source_dataset.get_version(source_dataset_version).id
        )

        data = {
            "source_dataset_id": new_source_dataset.id,
            "source_dataset_version_id": new_source_dataset.get_version(
                new_source_dataset_version
            ).id,
        }

        q = q.values(**data)
        self.db.execute(q)

    @abstractmethod
    def _dataset_dependencies_select_columns(self) -> list["SchemaItem"]:
        """
        Returns a list of columns to select in a query for fetching dataset dependencies
        """

    @abstractmethod
    def _dataset_dependency_nodes_select_columns(
        self,
        namespaces_subquery: "Subquery",
        dependency_tree_cte: "CTE",
        datasets_subquery: "Subquery",
    ) -> list["ColumnElement"]:
        """
        Returns a list of columns to select in a query for fetching
        dataset dependency nodes.
        """

    def get_direct_dataset_dependencies(
        self, dataset: DatasetRecord, version: str
    ) -> list[DatasetDependency | None]:
        n = self._namespaces
        p = self._projects
        d = self._datasets
        dd = self._datasets_dependencies
        dv = self._datasets_versions

        dataset_version = dataset.get_version(version)

        select_cols = self._dataset_dependencies_select_columns()

        query = (
            self._datasets_dependencies_select(*select_cols)
            .select_from(
                dd.join(d, dd.c.dataset_id == d.c.id, isouter=True)
                .join(dv, dd.c.dataset_version_id == dv.c.id, isouter=True)
                .join(p, d.c.project_id == p.c.id, isouter=True)
                .join(n, p.c.namespace_id == n.c.id, isouter=True)
            )
            .where(
                (dd.c.source_dataset_id == dataset.id)
                & (dd.c.source_dataset_version_id == dataset_version.id)
            )
        )

        return [self.dependency_class.parse(*r) for r in self.db.execute(query)]

    def get_dataset_dependency_nodes(
        self, dataset_id: int, version_id: int, depth_limit: int = DEPTH_LIMIT_DEFAULT
    ) -> list[DatasetDependencyNode | None]:
        n = self._namespaces_select().subquery()
        p = self._projects
        d = self._datasets_select().subquery()
        dd = self._datasets_dependencies
        dv = self._datasets_versions

        # Common dependency fields for CTE
        dep_fields = [
            dd.c.id,
            dd.c.source_dataset_id,
            dd.c.source_dataset_version_id,
            dd.c.dataset_id,
            dd.c.dataset_version_id,
        ]

        # Base case: direct dependencies
        base_query = select(
            *dep_fields,
            literal(0).label("depth"),
        ).where(
            (dd.c.source_dataset_id == dataset_id)
            & (dd.c.source_dataset_version_id == version_id)
        )

        cte = base_query.cte(name="dependency_tree", recursive=True)

        # Recursive case: dependencies of dependencies
        # Limit depth to 100 to prevent infinite loops in case of circular dependencies
        recursive_query = (
            select(
                *dep_fields,
                (cte.c.depth + 1).label("depth"),
            )
            .select_from(
                cte.join(
                    dd,
                    (cte.c.dataset_id == dd.c.source_dataset_id)
                    & (cte.c.dataset_version_id == dd.c.source_dataset_version_id),
                )
            )
            .where(cte.c.depth < depth_limit)
        )

        cte = cte.union(recursive_query)

        # Fetch all with full details
        select_cols = self._dataset_dependency_nodes_select_columns(
            namespaces_subquery=n,
            dependency_tree_cte=cte,
            datasets_subquery=d,
        )
        final_query = self._datasets_dependencies_select(*select_cols).select_from(
            # Use outer joins to handle cases where dependent datasets have been
            # physically deleted. This allows us to return dependency records with
            # None values instead of silently omitting them, making broken
            # dependencies visible to callers.
            cte.join(d, cte.c.dataset_id == d.c.id, isouter=True)
            .join(dv, cte.c.dataset_version_id == dv.c.id, isouter=True)
            .join(p, d.c.project_id == p.c.id, isouter=True)
            .join(n, p.c.namespace_id == n.c.id, isouter=True)
        )

        return [
            self.dependency_node_class.parse(*r) for r in self.db.execute(final_query)
        ]

    def remove_dataset_dependencies(
        self, dataset: DatasetRecord, version: str | None = None
    ) -> None:
        """
        When we remove dataset, we need to clean up it's dependencies as well
        """
        dd = self._datasets_dependencies

        q = self._datasets_dependencies_delete().where(
            dd.c.source_dataset_id == dataset.id
        )

        if version:
            q = q.where(
                dd.c.source_dataset_version_id == dataset.get_version(version).id
            )

        self.db.execute(q)

    def remove_dataset_dependants(
        self, dataset: DatasetRecord, version: str | None = None
    ) -> None:
        """
        When we remove dataset, we need to clear its references in other dataset
        dependencies
        """
        dd = self._datasets_dependencies

        q = self._datasets_dependencies_update().where(dd.c.dataset_id == dataset.id)
        if version:
            q = q.where(dd.c.dataset_version_id == dataset.get_version(version).id)

        q = q.values(dataset_id=None, dataset_version_id=None)

        self.db.execute(q)

    #
    # Jobs
    #

    @staticmethod
    def _jobs_columns() -> "list[SchemaItem]":
        return [
            Column(
                "id",
                Text,
                default=uuid4,
                primary_key=True,
                nullable=False,
            ),
            Column("name", Text, nullable=False, default=""),
            Column("status", Integer, nullable=False, default=JobStatus.CREATED),
            # When this Job was created
            Column("created_at", DateTime(timezone=True), nullable=False),
            # When this Job finished (or failed)
            Column("finished_at", DateTime(timezone=True), nullable=True),
            # This is the workers value from query settings, and determines both
            # the default and maximum number of workers for distributed UDFs.
            Column("query", Text, nullable=False, default=""),
            Column(
                "query_type",
                Integer,
                nullable=False,
                default=JobQueryType.PYTHON,
            ),
            Column("workers", Integer, nullable=False, default=1),
            Column("python_version", Text, nullable=True),
            Column("error_message", Text, nullable=False, default=""),
            Column("error_stack", Text, nullable=False, default=""),
            Column("params", JSON, nullable=False),
            Column("metrics", JSON, nullable=False),
            Column("parent_job_id", Text, nullable=True),
            Column("rerun_from_job_id", Text, nullable=True),
            Column("run_group_id", Text, nullable=True),
            Column("is_remote_execution", Boolean, nullable=False, default=False),
            Index("idx_jobs_parent_job_id", "parent_job_id"),
            Index("idx_jobs_rerun_from_job_id", "rerun_from_job_id"),
            Index("idx_jobs_run_group_id", "run_group_id"),
        ]

    @cached_property
    def _job_fields(self) -> list[str]:
        return [c.name for c in self._jobs_columns() if isinstance(c, Column)]  # type: ignore[attr-defined]

    @cached_property
    def _jobs(self) -> "Table":
        return Table(self.JOBS_TABLE, self.db.metadata, *self._jobs_columns())

    @abstractmethod
    def _jobs_insert(self) -> "Insert": ...

    def _jobs_select(self, *columns) -> "Select":
        if not columns:
            return self._jobs.select()
        return select(*columns)

    def _jobs_update(self, *where) -> "Update":
        if not where:
            return self._jobs.update()
        return self._jobs.update().where(*where)

    def _parse_job(self, rows) -> Job:
        return self.job_class.parse(*rows)

    def _parse_jobs(self, rows) -> Iterator["Job"]:
        for _, g in groupby(rows, lambda r: r[0]):
            yield self._parse_job(*list(g))

    def _jobs_query(self):
        return self._jobs_select(*[getattr(self._jobs.c, f) for f in self._job_fields])

    def list_jobs_by_ids(self, ids: list[str]) -> Iterator["Job"]:
        """List jobs by ids."""
        query = self._jobs_query().where(self._jobs.c.id.in_(ids))
        yield from self._parse_jobs(self.db.execute(query))

    def get_last_job_by_name(
        self, name: str, is_remote_execution: bool = False
    ) -> "Job | None":
        query = (
            self._jobs_query()
            .where(self._jobs.c.name == name)
            .where(self._jobs.c.is_remote_execution == is_remote_execution)
            .order_by(self._jobs.c.created_at.desc())
            .limit(1)
        )
        results = list(self.db.execute(query))
        if not results:
            return None
        return self._parse_job(results[0])

    def create_job(
        self,
        name: str,
        query: str,
        query_type: JobQueryType = JobQueryType.PYTHON,
        status: JobStatus = JobStatus.CREATED,
        workers: int = 1,
        python_version: str | None = None,
        params: dict[str, str] | None = None,
        parent_job_id: str | None = None,
        rerun_from_job_id: str | None = None,
        run_group_id: str | None = None,
        is_remote_execution: bool = False,
        job_id: str | None = None,
    ) -> str:
        """
        Creates a new job.
        Returns the job id.

        Args:
            job_id: If provided, uses this ID instead of generating a new one.
                    Used for saving Studio jobs locally with their original IDs.
        """
        if job_id is None:
            job_id = str(uuid4())

        if run_group_id is None:
            run_group_id = job_id

        self.db.execute(
            self._jobs_insert().values(
                id=job_id,
                name=name,
                status=status,
                created_at=datetime.now(timezone.utc),
                query=query,
                query_type=query_type.value,
                workers=workers,
                python_version=python_version,
                error_message="",
                error_stack="",
                params=json.dumps(params or {}),
                metrics=json.dumps({}),
                parent_job_id=parent_job_id,
                rerun_from_job_id=rerun_from_job_id,
                run_group_id=run_group_id,
                is_remote_execution=is_remote_execution,
            ),
        )
        return job_id

    def get_job(self, job_id: str) -> Job | None:
        """Returns the job with the given ID."""
        query = self._jobs_select(self._jobs).where(self._jobs.c.id == job_id)
        results = list(self.db.execute(query))
        if not results:
            return None
        return self._parse_job(results[0])

    def _has_active_checkpoints_in_run_group(
        self,
        run_group_id: str,
        ttl_threshold: datetime,
    ) -> bool:
        if not run_group_id:
            return False

        job_id_cast = cast(self._checkpoints.c.job_id, self._jobs.c.id.type)

        query = (
            self._jobs_select(f.count(self._checkpoints.c.id))
            .select_from(
                self._jobs.join(self._checkpoints, self._jobs.c.id == job_id_cast)
            )
            .where(
                and_(
                    self._jobs.c.run_group_id == run_group_id,
                    self._checkpoints.c.status != CheckpointStatus.DELETED,
                    self._checkpoints.c.created_at >= ttl_threshold,
                )
            )
        )

        return next(self.db.execute(query))[0] > 0

    def update_job(
        self,
        job_id: str,
        status: JobStatus | None = None,
        error_message: str | None = None,
        error_stack: str | None = None,
        finished_at: datetime | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> Job | None:
        """Updates job fields."""
        values: dict = {}
        if status is not None:
            values["status"] = status
        if error_message is not None:
            values["error_message"] = error_message
        if error_stack is not None:
            values["error_stack"] = error_stack
        if finished_at is not None:
            values["finished_at"] = finished_at
        if metrics:
            values["metrics"] = json.dumps(metrics)

        if values:
            j = self._jobs
            self.db.execute(
                self._jobs_update().where(j.c.id == job_id).values(**values),
            )  # type: ignore [attr-defined]

        return self.get_job(job_id)

    def set_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: str | None = None,
        error_stack: str | None = None,
    ) -> None:
        """Set the status of the given job."""
        values: dict = {"status": status}
        if status in JobStatus.finished():
            values["finished_at"] = datetime.now(timezone.utc)
        if error_message:
            values["error_message"] = error_message
        if error_stack:
            values["error_stack"] = error_stack
        self.db.execute(
            self._jobs_update(self._jobs.c.id == job_id).values(**values),
        )

    def get_job_status(
        self,
        job_id: str,
    ) -> JobStatus | None:
        """Returns the status of the given job."""
        results = list(
            self.db.execute(
                self._jobs_select(self._jobs.c.status).where(self._jobs.c.id == job_id),
            ),
        )
        if not results:
            return None
        return results[0][0]

    #
    # Checkpoints
    #

    @staticmethod
    def _checkpoints_columns() -> "list[SchemaItem]":
        return [
            Column(
                "id",
                Text,
                default=uuid4,
                primary_key=True,
                nullable=False,
            ),
            Column("job_id", Text, nullable=True),
            Column("hash", Text, nullable=False),
            Column("partial", Boolean, default=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("status", Integer, default=CheckpointStatus.ACTIVE, nullable=False),
            UniqueConstraint("job_id", "hash"),
        ]

    @cached_property
    def _checkpoints_fields(self) -> list[str]:
        return [c.name for c in self._checkpoints_columns() if c.name]  # type: ignore[attr-defined]

    @cached_property
    def _checkpoints(self) -> "Table":
        return Table(
            self.CHECKPOINTS_TABLE,
            self.db.metadata,
            *self._checkpoints_columns(),
        )

    @abstractmethod
    def _checkpoints_insert(self) -> "Insert": ...

    #
    # Checkpoint Events
    #

    @staticmethod
    def _checkpoint_events_columns() -> "list[SchemaItem]":
        return [
            Column(
                "id",
                Text,
                default=uuid4,
                primary_key=True,
                nullable=False,
            ),
            Column("job_id", Text, nullable=False),
            Column("run_group_id", Text, nullable=True),
            Column("timestamp", DateTime(timezone=True), nullable=False),
            Column("event_type", Text, nullable=False),
            Column("step_type", Text, nullable=False),
            Column("udf_name", Text, nullable=True),
            Column("dataset_name", Text, nullable=True),
            Column("checkpoint_hash", Text, nullable=True),
            Column("hash_partial", Text, nullable=True),
            Column("hash_input", Text, nullable=True),
            Column("hash_output", Text, nullable=True),
            Column("rows_input", BigInteger, nullable=True),
            Column("rows_processed", BigInteger, nullable=True),
            Column("rows_output", BigInteger, nullable=True),
            Column("rows_input_reused", BigInteger, nullable=True),
            Column("rows_output_reused", BigInteger, nullable=True),
            Column("rerun_from_job_id", Text, nullable=True),
            Column("details", JSON, nullable=True),
            Index("dc_idx_ce_job_id", "job_id"),
            Index("dc_idx_ce_run_group_id", "run_group_id"),
        ]

    @cached_property
    def _checkpoint_events_fields(self) -> list[str]:
        return [
            c.name  # type: ignore[attr-defined]
            for c in self._checkpoint_events_columns()
            if isinstance(c, Column)
        ]

    @cached_property
    def _checkpoint_events(self) -> "Table":
        return Table(
            self.CHECKPOINT_EVENTS_TABLE,
            self.db.metadata,
            *self._checkpoint_events_columns(),
        )

    @abstractmethod
    def _checkpoint_events_insert(self) -> "Insert": ...

    def _checkpoint_events_select(self, *columns) -> "Select":
        if not columns:
            return self._checkpoint_events.select()
        return select(*columns)

    def _checkpoint_events_query(self):
        return self._checkpoint_events_select(
            *[
                getattr(self._checkpoint_events.c, f)
                for f in self._checkpoint_events_fields
            ]
        )

    @classmethod
    def _dataset_version_jobs_columns(cls) -> "list[SchemaItem]":
        """Junction table for dataset versions and jobs many-to-many relationship."""
        return [
            Column("id", Integer, primary_key=True),
            Column(
                "dataset_version_id",
                Integer,
                ForeignKey(f"{cls.DATASET_VERSION_TABLE}.id", ondelete="CASCADE"),
                nullable=False,
            ),
            Column("job_id", Text, nullable=False),
            Column("is_creator", Boolean, nullable=False, default=False),
            Column("created_at", DateTime(timezone=True)),
            UniqueConstraint("dataset_version_id", "job_id"),
            Index("dc_idx_dvj_query", "job_id", "is_creator", "created_at"),
        ]

    @cached_property
    def _dataset_version_jobs_fields(self) -> list[str]:
        return [c.name for c in self._dataset_version_jobs_columns() if c.name]  # type: ignore[attr-defined]

    @cached_property
    def _dataset_version_jobs(self) -> "Table":
        return Table(
            self.DATASET_VERSION_JOBS_TABLE,
            self.db.metadata,
            *self._dataset_version_jobs_columns(),
        )

    @abstractmethod
    def _dataset_version_jobs_insert(self) -> "Insert": ...

    def _dataset_version_jobs_select(self, *columns) -> "Select":
        if not columns:
            return self._dataset_version_jobs.select()
        return select(*columns)

    def _dataset_version_jobs_delete(self) -> "Delete":
        return self._dataset_version_jobs.delete()

    def _checkpoints_select(self, *columns) -> "Select":
        query = self._checkpoints.select() if not columns else select(*columns)
        return query.where(self._checkpoints.c.status != CheckpointStatus.DELETED)

    def _checkpoints_query(self):
        return self._checkpoints_select(
            *[getattr(self._checkpoints.c, f) for f in self._checkpoints_fields]
        )

    def get_or_create_checkpoint(
        self,
        job_id: str,
        _hash: str,
        partial: bool = False,
    ) -> Checkpoint:
        with self.db.transaction():
            query = self._checkpoints_insert().values(
                id=str(uuid4()),
                job_id=job_id,
                hash=_hash,
                partial=partial,
                created_at=datetime.now(timezone.utc),
                status=CheckpointStatus.ACTIVE,
            )

            # Use on_conflict_do_nothing to handle race conditions
            if not hasattr(query, "on_conflict_do_nothing"):
                raise RuntimeError("Database must support on_conflict_do_nothing")
            query = query.on_conflict_do_nothing(index_elements=["job_id", "hash"])

            self.db.execute(query)

            checkpoint = self.find_checkpoint(job_id, _hash, partial=partial)
            if checkpoint is None:
                raise RuntimeError(
                    f"Checkpoint should exist after get_or_create for job_id={job_id}, "
                    f"hash={_hash}, partial={partial}"
                )

            return checkpoint

    def list_checkpoints(
        self,
        job_ids: list[str] | None = None,
        status: CheckpointStatus | None = None,
    ) -> Iterator[Checkpoint]:
        query = self._checkpoints_query()
        if job_ids is not None:
            query = query.where(self._checkpoints.c.job_id.in_(job_ids))
        if status is not None:
            query = query.where(self._checkpoints.c.status == status)
        rows = list(self.db.execute(query))

        yield from [self.checkpoint_class.parse(*r) for r in rows]

    def get_checkpoint_by_id(self, checkpoint_id: str) -> Checkpoint:
        """Returns the checkpoint with the given ID."""
        ch = self._checkpoints
        query = self._checkpoints_select(ch).where(ch.c.id == checkpoint_id)
        rows = list(self.db.execute(query))
        if not rows:
            raise CheckpointNotFoundError(f"Checkpoint {checkpoint_id} not found")
        return self.checkpoint_class.parse(*rows[0])

    def find_checkpoint(
        self, job_id: str, _hash: str, partial: bool = False
    ) -> Checkpoint | None:
        """
        Tries to find an active checkpoint for a job with specific hash.
        Ignores expired and deleted checkpoints.
        """
        ch = self._checkpoints
        query = self._checkpoints_select(ch).where(
            ch.c.job_id == job_id,
            ch.c.hash == _hash,
            ch.c.partial == partial,
            ch.c.status == CheckpointStatus.ACTIVE,
        )
        rows = list(self.db.execute(query))
        if not rows:
            return None
        return self.checkpoint_class.parse(*rows[0])

    def get_last_checkpoint(self, job_id: str) -> Checkpoint | None:
        query = (
            self._checkpoints_query()
            .where(self._checkpoints.c.job_id == job_id)
            .order_by(desc(self._checkpoints.c.created_at))
            .limit(1)
        )
        rows = list(self.db.execute(query))
        if not rows:
            return None
        return self.checkpoint_class.parse(*rows[0])

    def log_checkpoint_event(  # noqa: PLR0913
        self,
        job_id: str,
        event_type: CheckpointEventType,
        step_type: CheckpointStepType,
        run_group_id: str | None = None,
        udf_name: str | None = None,
        dataset_name: str | None = None,
        checkpoint_hash: str | None = None,
        hash_partial: str | None = None,
        hash_input: str | None = None,
        hash_output: str | None = None,
        rows_input: int | None = None,
        rows_processed: int | None = None,
        rows_output: int | None = None,
        rows_input_reused: int | None = None,
        rows_output_reused: int | None = None,
        rerun_from_job_id: str | None = None,
        details: dict | None = None,
    ) -> CheckpointEvent:
        """Log a checkpoint event."""
        event_id = str(uuid4())
        timestamp = datetime.now(timezone.utc)

        query = self._checkpoint_events_insert().values(
            id=event_id,
            job_id=job_id,
            run_group_id=run_group_id,
            timestamp=timestamp,
            event_type=event_type.value,
            step_type=step_type.value,
            udf_name=udf_name,
            dataset_name=dataset_name,
            checkpoint_hash=checkpoint_hash,
            hash_partial=hash_partial,
            hash_input=hash_input,
            hash_output=hash_output,
            rows_input=rows_input,
            rows_processed=rows_processed,
            rows_output=rows_output,
            rows_input_reused=rows_input_reused,
            rows_output_reused=rows_output_reused,
            rerun_from_job_id=rerun_from_job_id,
            details=details,
        )
        self.db.execute(query)

        return CheckpointEvent(
            id=event_id,
            job_id=job_id,
            run_group_id=run_group_id,
            timestamp=timestamp,
            event_type=event_type,
            step_type=step_type,
            udf_name=udf_name,
            dataset_name=dataset_name,
            checkpoint_hash=checkpoint_hash,
            hash_partial=hash_partial,
            hash_input=hash_input,
            hash_output=hash_output,
            rows_input=rows_input,
            rows_processed=rows_processed,
            rows_output=rows_output,
            rows_input_reused=rows_input_reused,
            rows_output_reused=rows_output_reused,
            rerun_from_job_id=rerun_from_job_id,
            details=details,
        )

    def get_checkpoint_events(
        self,
        job_id: str | None = None,
        run_group_id: str | None = None,
    ) -> Iterator[CheckpointEvent]:
        """Get checkpoint events, optionally filtered by job_id or run_group_id."""
        query = self._checkpoint_events_query()

        if job_id is not None:
            query = query.where(self._checkpoint_events.c.job_id == job_id)
        if run_group_id is not None:
            query = query.where(self._checkpoint_events.c.run_group_id == run_group_id)

        query = query.order_by(self._checkpoint_events.c.timestamp)
        rows = list(self.db.execute(query))

        yield from [self.checkpoint_event_class.parse(*r) for r in rows]

    def link_dataset_version_to_job(
        self,
        dataset_version_id: int,
        job_id: str,
        is_creator: bool = False,
    ) -> None:
        with self.db.transaction():
            # Insert into junction table
            query = self._dataset_version_jobs_insert().values(
                dataset_version_id=dataset_version_id,
                job_id=job_id,
                is_creator=is_creator,
                created_at=datetime.now(timezone.utc),
            )
            if hasattr(query, "on_conflict_do_nothing"):
                query = query.on_conflict_do_nothing(
                    index_elements=["dataset_version_id", "job_id"]
                )
            self.db.execute(query)

            # Also update dataset_version.job_id to point to this job
            update_query = (
                self._datasets_versions.update()
                .where(self._datasets_versions.c.id == dataset_version_id)
                .values(job_id=job_id)
            )
            self.db.execute(update_query)

    def get_ancestor_job_ids(self, job_id: str) -> list[str]:
        """Get all ancestor job IDs using recursive CTE."""
        ancestors_cte = (
            self._jobs_select(
                self._jobs.c.id.label("id"),
                self._jobs.c.rerun_from_job_id.label("rerun_from_job_id"),
                self._jobs.c.run_group_id.label("run_group_id"),
                literal(0).label("depth"),
            )
            .where(self._jobs.c.id == job_id)
            .cte(name="ancestors", recursive=True)
        )

        ancestors_recursive = ancestors_cte.union_all(
            self._jobs_select(
                self._jobs.c.id.label("id"),
                self._jobs.c.rerun_from_job_id.label("rerun_from_job_id"),
                self._jobs.c.run_group_id.label("run_group_id"),
                (ancestors_cte.c.depth + 1).label("depth"),
            ).select_from(
                self._jobs.join(
                    ancestors_cte,
                    (
                        self._jobs.c.id
                        == cast(ancestors_cte.c.rerun_from_job_id, self._jobs.c.id.type)
                    )
                    & (
                        ancestors_cte.c.rerun_from_job_id.isnot(None)
                    )  # Stop at root jobs
                    & (ancestors_cte.c.depth < JOB_ANCESTRY_MAX_DEPTH)
                    & (
                        self._jobs.c.run_group_id
                        == cast(
                            ancestors_cte.c.run_group_id, self._jobs.c.run_group_id.type
                        )
                    ),  # Safety: only traverse within same run group
                )
            )
        )

        # Select all ancestor IDs and depths except the starting job itself
        query = select(ancestors_recursive.c.id, ancestors_recursive.c.depth).where(
            ancestors_recursive.c.id != job_id
        )

        results = list(self.db.execute(query))

        # Check if we hit the depth limit
        if results:
            max_found_depth = max(row[1] for row in results)
            if max_found_depth >= JOB_ANCESTRY_MAX_DEPTH:
                from datachain.error import JobAncestryDepthExceededError

                raise JobAncestryDepthExceededError(
                    f"Job ancestry chain exceeds maximum depth of "
                    f"{JOB_ANCESTRY_MAX_DEPTH}. Job ID: {job_id}"
                )

        return [str(row[0]) for row in results]

    def _get_dataset_version_for_job_ancestry_query(
        self,
        dataset_name: str,
        namespace_name: str,
        project_name: str,
        job_ancestry: list[str],
    ) -> "Select":
        """Find most recent dataset version created by any job in ancestry.

        Searches job ancestry (current + parents) for the newest version of
        the dataset where is_creator=True. Returns newest by created_at, or
        None if no version was created by any job in the ancestry chain.

        Used for checkpoint resolution to find which version to reuse when
        continuing from a parent job.
        """
        return (
            self._datasets_versions_select()
            .select_from(
                self._dataset_version_jobs.join(
                    self._datasets_versions,
                    self._dataset_version_jobs.c.dataset_version_id
                    == self._datasets_versions.c.id,
                )
                .join(
                    self._datasets,
                    self._datasets_versions.c.dataset_id == self._datasets.c.id,
                )
                .join(
                    self._projects,
                    self._datasets.c.project_id == self._projects.c.id,
                )
                .join(
                    self._namespaces,
                    self._projects.c.namespace_id == self._namespaces.c.id,
                )
            )
            .where(
                self._datasets.c.name == dataset_name,
                self._namespaces.c.name == namespace_name,
                self._projects.c.name == project_name,
                self._dataset_version_jobs.c.job_id.in_(job_ancestry),
                self._dataset_version_jobs.c.is_creator.is_(True),
            )
            .order_by(desc(self._dataset_version_jobs.c.created_at))
            .limit(1)
        )

    def get_dataset_version_for_job_ancestry(
        self,
        dataset_name: str,
        namespace_name: str,
        project_name: str,
        job_id: str,
    ) -> DatasetVersion | None:
        # Get job ancestry (current job + all ancestors)
        job_ancestry = [job_id, *self.get_ancestor_job_ids(job_id)]

        query = self._get_dataset_version_for_job_ancestry_query(
            dataset_name, namespace_name, project_name, job_ancestry
        )

        results = list(self.db.execute(query))
        if not results:
            return None

        if len(results) > 1:
            raise DataChainError(
                f"Expected at most 1 dataset version, found {len(results)}"
            )

        return self.dataset_version_class.parse(*results[0])

    def expire_checkpoints(
        self, ttl_threshold: datetime
    ) -> tuple[list[Checkpoint], list[str]]:
        ch = self._checkpoints
        jobs = self._jobs
        job_id_cast = cast(ch.c.job_id, jobs.c.id.type)

        # Step 1: Atomic UPDATE — after this, find_checkpoint won't return
        # these checkpoints, so running jobs are safe.

        # Subquery: finished jobs where ALL active checkpoints are older than TTL
        expired_job_ids = (
            select(job_id_cast)
            .where(ch.c.status == CheckpointStatus.ACTIVE)
            .group_by(ch.c.job_id)
            .having(f.max(ch.c.created_at) < ttl_threshold)
        )
        # Exclude run groups that have any active (non-finished) job — a rerun
        # may be using checkpoints from a finished parent via find_checkpoint
        active_run_groups = (
            select(jobs.c.run_group_id)
            .where(~jobs.c.status.in_(JobStatus.finished()))
            .where(jobs.c.run_group_id.isnot(None))
        )
        finished_job_ids = (
            select(jobs.c.id)
            .where(jobs.c.status.in_(JobStatus.finished()))
            .where(jobs.c.id.in_(expired_job_ids))
            .where(
                or_(
                    jobs.c.run_group_id.is_(None),
                    ~jobs.c.run_group_id.in_(active_run_groups),
                )
            )
        )

        self.db.execute(
            ch.update()
            .where(ch.c.status == CheckpointStatus.ACTIVE)
            .where(job_id_cast.in_(finished_job_ids))
            .values(status=CheckpointStatus.EXPIRED)
        )

        # Step 2: Collect ALL expired checkpoints (including leftovers from
        # prior crashed cleanup runs).
        checkpoints = list(self.list_checkpoints(status=CheckpointStatus.EXPIRED))
        if not checkpoints:
            return [], []

        # Step 3: Find run groups where no ACTIVE checkpoints remain
        # (safe to remove shared input tables).
        expired_job_ids_set = {cp.job_id for cp in checkpoints}
        run_group_ids: set[str] = set()
        for job in self.list_jobs_by_ids(list(expired_job_ids_set)):
            if job.run_group_id:
                run_group_ids.add(job.run_group_id)

        inactive_group_ids = [
            gid
            for gid in run_group_ids
            if not self._has_active_checkpoints_in_run_group(gid, ttl_threshold)
        ]

        return checkpoints, inactive_group_ids

    def update_checkpoints(
        self,
        checkpoint_ids: list[str],
        status: CheckpointStatus | None = None,
    ) -> None:
        if not checkpoint_ids:
            return
        values: dict = {}
        if status is not None:
            values["status"] = status
        if not values:
            return
        self.db.execute(
            self._checkpoints.update()
            .where(self._checkpoints.c.id.in_(checkpoint_ids))
            .values(**values),
        )
