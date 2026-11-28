import atexit
import logging
import os
import re
import sys
import traceback
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from copy import copy
from typing import TYPE_CHECKING, ClassVar
from uuid import uuid4
from weakref import WeakKeyDictionary, WeakSet

from datachain.catalog import get_catalog
from datachain.data_storage import JobQueryType, JobStatus
from datachain.dataset import SESSION_DATASET_PREFIX
from datachain.error import DataChainError, JobNotFoundError, TableMissingError
from datachain.utils import internal

if TYPE_CHECKING:
    from datachain.catalog import Catalog
    from datachain.data_storage import AbstractMetastore
    from datachain.job import Job

logger = logging.getLogger("datachain")


def is_script_run() -> bool:
    """
    Returns True if this was ran as python script, e.g python my_script.py.
    Otherwise (if interactive or module run) returns False.
    """
    try:
        argv0 = sys.argv[0]
    except (IndexError, AttributeError):
        return False
    return bool(argv0) and argv0 not in ("-c", "-m", "ipython")


class Session:
    """The context that chains run in. It sets the catalog where they read and
    save datasets, the client configuration they read storage with, and how long
    the temporary datasets they create live.

    Temporary datasets are the ones made without a name, such as by `persist()`.
    A session opened with `with` is used by every call inside the block that has
    no `session=`, and removes its temporary datasets when the block ends. Calls
    outside any `with` use a default session, which removes them when the
    process exits.

    Parameters:
        name (str): Letters and numbers only, used in the names of the
            session's temporary datasets. Optional.
        catalog (Catalog): Where the session's chains read and save datasets.
            By default, the one the environment configures, created with
            `client_config` and `in_memory`.
        client_config (dict): Client configuration, such as credentials or an
            endpoint, that the session's chains read storage with.
        in_memory (bool): Keep datasets in memory instead of the configured
            storage. A process's in-memory sessions share them, and they are
            gone once the last of these sessions closes.

    Example:
        ```py
        import datachain as dc

        with dc.Session("anon", client_config={"anon": True}):
            dc.read_storage("s3://public-bucket/images/").save("images")
        ```
    """

    GLOBAL_SESSION_CTX: "Session | None" = None
    # The session `in_memory=True` resolves to; never the global session.
    IN_MEMORY_SESSION: "Session | None" = None
    SESSION_CONTEXTS: ClassVar[list["Session"]] = []
    _ALL_SESSIONS: ClassVar[WeakSet["Session"]] = WeakSet()
    # The job of an in-memory catalog lives in that catalog's database.
    _IN_MEMORY_JOBS: ClassVar["WeakKeyDictionary[AbstractMetastore, Job]"] = (
        WeakKeyDictionary()
    )
    ORIGINAL_EXCEPT_HOOK = None

    # Job management - class-level to ensure one job per process
    _CURRENT_JOB: ClassVar["Job | None"] = None
    _JOB_STATUS: ClassVar[JobStatus | None] = None
    _OWNS_JOB: ClassVar[bool | None] = None
    _JOB_HOOKS_REGISTERED: ClassVar[bool] = False
    # The session that created the process job and finalizes it.
    _JOB_SESSION: ClassVar["Session | None"] = None
    _EXCEPT_HOOK_INSTALLED: ClassVar[bool] = False

    GLOBAL_SESSION_NAME = "global"
    SESSION_UUID_LEN = 6
    TEMP_TABLE_UUID_LEN = 6

    def __init__(
        self,
        name="",
        catalog: "Catalog | None" = None,
        client_config: dict | None = None,
        in_memory: bool = False,
    ):
        if re.match(r"^[0-9a-zA-Z]*$", name) is None:
            raise ValueError(
                f"Session name can contain only letters or numbers - '{name}' given."
            )

        if not name:
            name = self.GLOBAL_SESSION_NAME

        session_uuid = uuid4().hex[: self.SESSION_UUID_LEN]
        self.name = f"{name}_{session_uuid}"
        self.is_new_catalog = not catalog
        self._parent: Session | None = None
        self.catalog = catalog or get_catalog(
            client_config=client_config, in_memory=in_memory
        )
        self._cleanup_catalogs: list[Catalog] = [self.catalog]
        Session._ALL_SESSIONS.add(self)

    def _with_catalog(self, catalog: "Catalog") -> "Session":
        """Use another catalog without creating a new temporary-dataset scope."""
        if catalog is self.catalog:
            return self
        view = copy(self)
        view.catalog = catalog
        # Keep the cleanup owner alive; only owners are in _ALL_SESSIONS.
        view._parent = self._parent or self
        # The owner cleans up temporary datasets created through this view.
        view._cleanup_catalogs = self._cleanup_catalogs
        view.is_new_catalog = False
        return view

    def __enter__(self):
        if (
            self._parent is not None
            or self is Session.GLOBAL_SESSION_CTX
            or self is Session.IN_MEMORY_SESSION
            or self in Session.SESSION_CONTEXTS
        ):
            # Exiting closes the session, so enter only one this code owns.
            raise RuntimeError(
                "Cannot enter a view, the global or shared in-memory session, or "
                "a session already entered; open one with `with dc.Session(name)`"
            )
        # Push the current context onto the stack
        Session.SESSION_CONTEXTS.append(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._close()
        finally:
            if Session.SESSION_CONTEXTS:
                Session.SESSION_CONTEXTS.pop()

    def _close(self) -> None:
        """Clean up this scope without changing the active session stack."""
        # Don't cleanup created versions on exception
        # Datasets should persist even if the session fails
        self._cleanup_temp_datasets()
        if self.is_new_catalog:
            self.catalog.metastore.close_on_exit()
            self.catalog.warehouse.close_on_exit()

        Session._ALL_SESSIONS.discard(self)
        if Session.IN_MEMORY_SESSION is self:
            Session.IN_MEMORY_SESSION = None

    @internal
    def get_job(self) -> "Job | None":
        """
        Return the current job if one exists, without creating a new one.

        Checks the cached ``_CURRENT_JOB`` and ``DATACHAIN_JOB_ID`` env var.
        Returns None if no job is found. An in-memory catalog has a job of its own.
        """
        if self.catalog.in_memory:
            return Session._IN_MEMORY_JOBS.get(self.catalog.metastore)

        if Session._CURRENT_JOB:
            return Session._CURRENT_JOB

        if env_job_id := os.getenv("DATACHAIN_JOB_ID"):
            Session._CURRENT_JOB = self.catalog.metastore.get_job(env_job_id)
            if Session._CURRENT_JOB:
                Session._OWNS_JOB = False
            return Session._CURRENT_JOB

        return None

    @internal
    def get_or_create_job(self) -> "Job":
        """
        Get or create a Job for this process.

        Returns:
            Job: The active Job instance.

        Behavior:
            - If a job already exists, it is returned.
            - If in Studio without DATACHAIN_JOB_ID, raises an error.
            - If ``DATACHAIN_JOB_ID`` is set, the corresponding job is fetched.
            - Otherwise, a new job is created:
                * Name = absolute path to the Python script.
                * Query = empty string.
                * Parent = last job with the same name, if available.
                * Status = "running".
              Exit hooks are registered to finalize the job.

        Note:
            Job is shared across all Session instances to ensure one job per process.
            An in-memory catalog has a job of its own, in that catalog.
        """
        if self.catalog.in_memory:
            metastore = self.catalog.metastore
            job = Session._IN_MEMORY_JOBS.get(metastore)
            if job is None:
                job_id = metastore.create_job("in-memory", "", status=JobStatus.RUNNING)
                job = metastore.get_job(job_id)
                assert job is not None
                Session._IN_MEMORY_JOBS[metastore] = job
            return job

        if Session._CURRENT_JOB:
            return Session._CURRENT_JOB

        from datachain.lib.dc.utils import is_studio

        if env_job_id := os.getenv("DATACHAIN_JOB_ID"):
            # SaaS run: just fetch existing job
            Session._CURRENT_JOB = self.catalog.metastore.get_job(env_job_id)
            if not Session._CURRENT_JOB:
                raise JobNotFoundError(
                    f"Job {env_job_id} from DATACHAIN_JOB_ID env not found"
                )
            Session._OWNS_JOB = False
        elif is_studio():
            raise DataChainError(
                "Cannot create job in Studio without DATACHAIN_JOB_ID. "
                "This usually means an internal operation is missing "
                "ephemeral mode."
            )
        else:
            # Local run: create new job
            query = ""
            if is_script_run():
                script = os.path.abspath(sys.argv[0])
                try:
                    with open(script) as f:
                        query = f.read()
                except OSError:
                    pass
            else:
                # Interactive session or module run - use unique name to avoid
                # linking unrelated sessions
                script = str(uuid4())
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

            # try to find the parent job for checkpoint/rerun chain
            parent = self.catalog.metastore.get_last_job_by_name(script)

            job_id = self.catalog.metastore.create_job(
                name=script,
                query=query,
                query_type=JobQueryType.PYTHON,
                status=JobStatus.RUNNING,
                python_version=python_version,
                rerun_from_job_id=parent.id if parent else None,
                run_group_id=parent.run_group_id if parent else None,
            )
            Session._CURRENT_JOB = self.catalog.metastore.get_job(job_id)
            Session._OWNS_JOB = True
            Session._JOB_STATUS = JobStatus.RUNNING

            # register cleanup hooks only once
            if not Session._JOB_HOOKS_REGISTERED:
                atexit.register(Session._finalize_job_at_exit)
                Session._install_except_hook()
                Session._JOB_HOOKS_REGISTERED = True
            Session._JOB_SESSION = self

        assert Session._CURRENT_JOB is not None
        return Session._CURRENT_JOB

    @contextmanager
    def _job_catalog(self) -> Iterator["Catalog"]:
        """This session's catalog on new connections: the session may be closed
        already, or belong to another thread."""
        with ExitStack() as connections:
            catalog = self.catalog.copy()
            catalog.metastore = self.catalog.metastore.clone(use_new_connection=True)
            connections.callback(catalog.metastore.close_on_exit)
            catalog.warehouse = self.catalog.warehouse.clone(use_new_connection=True)
            connections.callback(catalog.warehouse.close_on_exit)
            yield catalog

    @classmethod
    def _finalize_job_at_exit(cls) -> None:
        if cls._JOB_SESSION is not None:
            cls._JOB_SESSION._finalize_job_success()

    def _finalize_job_success(self):
        """Mark the current job as completed."""
        if (
            Session._CURRENT_JOB
            and Session._OWNS_JOB
            and Session._JOB_STATUS == JobStatus.RUNNING
        ):
            with self._job_catalog() as catalog:
                catalog.metastore.set_job_status(
                    Session._CURRENT_JOB.id, JobStatus.COMPLETE
                )
            Session._JOB_STATUS = JobStatus.COMPLETE

    def _finalize_job_as_canceled(self):
        """Mark the current job as canceled."""
        if (
            Session._CURRENT_JOB
            and Session._OWNS_JOB
            and Session._JOB_STATUS == JobStatus.RUNNING
        ):
            with self._job_catalog() as catalog:
                catalog.metastore.set_job_status(
                    Session._CURRENT_JOB.id, JobStatus.CANCELED
                )
            Session._JOB_STATUS = JobStatus.CANCELED

    def _finalize_job_as_failed(self, exc_type, exc_value, tb):
        """Mark the current job as failed with error details."""
        if (
            Session._CURRENT_JOB
            and Session._OWNS_JOB
            and Session._JOB_STATUS == JobStatus.RUNNING
        ):
            error_stack = "".join(traceback.format_exception(exc_type, exc_value, tb))
            with self._job_catalog() as catalog:
                catalog.metastore.set_job_status(
                    Session._CURRENT_JOB.id,
                    JobStatus.FAILED,
                    error_message=str(exc_value),
                    error_stack=error_stack,
                )

                # Mark any incomplete dataset versions created by this job as FAILED
                catalog.metastore.mark_job_dataset_versions_as_failed(
                    Session._CURRENT_JOB.id
                )
                # Finally clean all incomplete dataset versions
                catalog.cleanup_dataset_versions(job_id=Session._CURRENT_JOB.id)

            Session._JOB_STATUS = JobStatus.FAILED

    def generate_temp_dataset_name(self) -> str:
        if not any(
            c.metastore is self.catalog.metastore
            and c.warehouse is self.catalog.warehouse
            for c in self._cleanup_catalogs
        ):
            self._cleanup_catalogs.append(self.catalog)
        return self.get_temp_prefix() + uuid4().hex[: self.TEMP_TABLE_UUID_LEN]

    def get_temp_prefix(self) -> str:
        return f"{SESSION_DATASET_PREFIX}{self.name}_"

    @classmethod
    def is_temp_dataset(cls, name) -> bool:
        return name.startswith(SESSION_DATASET_PREFIX)

    def _cleanup_temp_datasets(self) -> None:
        prefix = self.get_temp_prefix()
        for catalog in self._cleanup_catalogs:
            try:
                with ExitStack() as connections:
                    if catalog is not self.catalog:
                        # Borrowed catalogs may already be closed, or belong to
                        # another thread. Cleanup owns only these new connections.
                        catalog = catalog.copy()
                        catalog.metastore = catalog.metastore.clone(
                            use_new_connection=True
                        )
                        connections.callback(catalog.metastore.close_on_exit)
                        catalog.warehouse = catalog.warehouse.clone(
                            use_new_connection=True
                        )
                        connections.callback(catalog.warehouse.close_on_exit)
                    for dataset in list(
                        catalog.metastore.list_datasets_by_prefix(
                            prefix, include_incomplete=True
                        )
                    ):
                        # Session datasets are always wiped — never soft-deleted.
                        catalog.remove_dataset(
                            dataset.name,
                            dataset.project,
                            force=True,
                            keep_metadata=False,
                        )
            # suppress error when metastore has been reset during testing
            except TableMissingError:
                pass
        # A retained chain/view must not keep other catalogs alive after cleanup.
        self._cleanup_catalogs[:] = [self.catalog]

    @classmethod
    @internal
    def active(cls) -> "Session | None":
        """The session ``get()`` returns without arguments, without creating
        one: None when no session exists yet."""
        if cls.SESSION_CONTEXTS:
            return cls.SESSION_CONTEXTS[-1]
        return cls.GLOBAL_SESSION_CTX

    @classmethod
    @internal
    def get(
        cls,
        session: "Session | None" = None,
        catalog: "Catalog | None" = None,
        *,
        in_memory: bool = False,
    ) -> "Session":
        """The session a call without `session=` uses. Internal: to open a session,
        or to read with another `client_config`, use `dc.Session(name, ...)`.

        Parameters:
            session (Session): Returned as is when given.
            catalog (Catalog): The catalog of the global session, if this call
                    creates it. An in-memory catalog gets the shared in-memory
                    session instead. By default, the catalog the environment
                    configures.
            in_memory (bool): The active session if it is in memory, else one
                    in-memory session shared by the process. Raises ValueError
                    with a persistent `session`.
        """
        if session:
            if in_memory and not session.catalog.in_memory:
                raise ValueError(
                    "in_memory=True can't be used with the persistent session "
                    f"{session.name!r}. Drop session=, or pass an in-memory "
                    "session: dc.Session(name, in_memory=True)."
                )
            return session

        active = cls.active()
        # The global session is never in memory.
        if active is None and catalog is not None and catalog.in_memory:
            in_memory = True
        if in_memory and (active is None or not active.catalog.in_memory):
            if cls.IN_MEMORY_SESSION is None:
                cls.IN_MEMORY_SESSION = Session("inmemory", in_memory=True)
            session = cls.IN_MEMORY_SESSION

        # Access the active (most recent) context from the stack
        elif cls.SESSION_CONTEXTS:
            session = cls.SESSION_CONTEXTS[-1]

        elif cls.GLOBAL_SESSION_CTX is None:
            cls.GLOBAL_SESSION_CTX = Session(cls.GLOBAL_SESSION_NAME, catalog)
            session = cls.GLOBAL_SESSION_CTX

            atexit.register(cls._global_cleanup)
            cls._install_except_hook()
        else:
            session = cls.GLOBAL_SESSION_CTX

        return session

    @classmethod
    def _install_except_hook(cls) -> None:
        # Only once: a hook set after ours may call ours back.
        if not cls._EXCEPT_HOOK_INSTALLED:
            cls.ORIGINAL_EXCEPT_HOOK = sys.excepthook
            sys.excepthook = cls.except_hook
            cls._EXCEPT_HOOK_INSTALLED = True

    @staticmethod
    def except_hook(exc_type, exc_value, exc_traceback):
        if getattr(sys, "ps1", None):
            original_hook = Session.ORIGINAL_EXCEPT_HOOK or sys.__excepthook__
            original_hook(exc_type, exc_value, exc_traceback)
            return

        # Handle KeyboardInterrupt specially - mark as canceled and exit with
        # signal code
        job_session = Session._JOB_SESSION
        if job_session and exc_type is KeyboardInterrupt:
            job_session._finalize_job_as_canceled()
        elif job_session:
            job_session._finalize_job_as_failed(exc_type, exc_value, exc_traceback)

        Session._global_cleanup()

        # Always delegate to original hook if it exists
        if Session.ORIGINAL_EXCEPT_HOOK:
            Session.ORIGINAL_EXCEPT_HOOK(exc_type, exc_value, exc_traceback)

        if exc_type is KeyboardInterrupt:
            # Exit with SIGINT signal code (128 + 2 = 130, or -2 in subprocess terms)
            sys.exit(130)

    @classmethod
    def cleanup_for_tests(cls):
        cls._close_all_contexts()
        if cls.IN_MEMORY_SESSION is not None:
            cls.IN_MEMORY_SESSION._close()
        cls._IN_MEMORY_JOBS.clear()
        if cls.GLOBAL_SESSION_CTX is not None:
            cls.GLOBAL_SESSION_CTX.__exit__(None, None, None)
            cls.GLOBAL_SESSION_CTX = None
            atexit.unregister(cls._global_cleanup)

        # Reset job-related class variables
        atexit.unregister(cls._finalize_job_at_exit)
        cls._CURRENT_JOB = None
        cls._JOB_STATUS = None
        cls._OWNS_JOB = None
        cls._JOB_HOOKS_REGISTERED = False
        cls._JOB_SESSION = None

        if cls.ORIGINAL_EXCEPT_HOOK:
            sys.excepthook = cls.ORIGINAL_EXCEPT_HOOK
        cls._EXCEPT_HOOK_INSTALLED = False

    @staticmethod
    def _global_cleanup():
        Session._close_all_contexts()
        if Session.GLOBAL_SESSION_CTX is not None:
            Session.GLOBAL_SESSION_CTX.__exit__(None, None, None)

        for session in list(Session._ALL_SESSIONS):
            try:
                session.__exit__(None, None, None)
            except ReferenceError:
                continue  # Object has been finalized already
            except Exception as e:  # noqa: BLE001
                logger.error(f"Exception while cleaning up session: {e}")  # noqa: G004

    @classmethod
    def _close_all_contexts(cls) -> None:
        while cls.SESSION_CONTEXTS:
            session = cls.SESSION_CONTEXTS.pop()
            try:
                session.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Exception while closing session context during cleanup: %s",
                    exc,
                )
