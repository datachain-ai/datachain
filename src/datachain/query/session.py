import atexit
import logging
import os
import re
import sys
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar
from uuid import uuid4
from weakref import WeakSet

from datachain.catalog import get_catalog
from datachain.data_storage import JobQueryType, JobStatus
from datachain.dataset import SESSION_DATASET_PREFIX
from datachain.error import DataChainError, JobNotFoundError, TableMissingError

if TYPE_CHECKING:
    from datachain.catalog import Catalog
    from datachain.job import Job

logger = logging.getLogger("datachain")


def _copy_client_config(value):
    """Recursively copy the mapping/sequence structure of a client config so
    later caller-side mutation (including nested ``client_kwargs`` etc.)
    cannot change a session's configuration. Leaf objects (credential
    providers, SSL contexts, ...) are kept by reference."""
    if isinstance(value, dict):
        return {k: _copy_client_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_copy_client_config(v) for v in value)
    return value


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
    """
    Session is a context that keeps track of temporary DataChain datasets for a proper
    cleanup. By default, a global session is created.

    Temporary or ephemeral datasets are the ones created without specified name.
    They are useful for optimization purposes and should be automatically removed.

    Temp dataset has specific name format:
        "session_<name>_<session_uuid>_<dataset_uuid>"
    The <name> suffix is optional. Both <uuid>s are auto-generated.

    Temp dataset examples:
        session_myname_624b41_48e8b4
        session_4b962d_2a5dff

    Parameters:

    name (str): The name of the session. Only latters and numbers are supported.
           It can be empty.
    catalog (Catalog): Catalog object.
    """

    GLOBAL_SESSION_CTX: "Session | None" = None
    # The implicit in-memory session; see _get_in_memory_session.
    IN_MEMORY_SESSION_CTX: "Session | None" = None
    # Owned non-context sessions: per-call client_config overrides and
    # wrappers for explicitly provided in-memory catalogs.
    OVERRIDE_SESSIONS: ClassVar[dict[str, "Session"]] = {}
    SESSION_CONTEXTS: ClassVar[list["Session"]] = []
    _ALL_SESSIONS: ClassVar[WeakSet["Session"]] = WeakSet()
    ORIGINAL_EXCEPT_HOOK = None

    # Job management - class-level to ensure one job per process
    _CURRENT_JOB: ClassVar["Job | None"] = None
    _JOB_STATUS: ClassVar[JobStatus | None] = None
    _OWNS_JOB: ClassVar[bool | None] = None
    _JOB_HOOKS_REGISTERED: ClassVar[bool] = False
    _JOB_FINALIZE_HOOK: ClassVar[Callable[[], None] | None] = None

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
        self.catalog = catalog or get_catalog(
            client_config=client_config, in_memory=in_memory
        )
        # Session-local job for in-memory catalogs; see get_or_create_job.
        self._session_job: Job | None = None
        self._closed = False
        Session._ALL_SESSIONS.add(self)

    def __enter__(self):
        # Push the current context onto the stack
        Session.SESSION_CONTEXTS.append(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Idempotent: a session may be exited by both its `with` block and a
        # cleanup sweep; a second exit would reconnect the already-closed
        # database just to query temp datasets, leaking the new connection.
        if self._closed:
            return
        self._closed = True

        # Don't cleanup created versions on exception
        # Datasets should persist even if the session fails
        if not getattr(self.catalog.metastore.db, "is_closed", False):
            # A session over an already-closed database (e.g. a wrapper whose
            # owner was cleaned up first) has nothing left to clean; querying
            # would silently reconnect and leak the connection.
            self._cleanup_temp_datasets()
        if self.is_new_catalog:
            self.catalog.metastore.close_on_exit()
            self.catalog.warehouse.close_on_exit()

        # Only ever remove *this* session: exiting a non-context session (or
        # exiting out of order) must not corrupt the context stack.
        if self in Session.SESSION_CONTEXTS:
            Session.SESSION_CONTEXTS.remove(self)
        Session._ALL_SESSIONS.discard(self)

    def get_job(self) -> "Job | None":
        """
        Return the current job if one exists, without creating a new one.

        Checks the cached ``_CURRENT_JOB`` and ``DATACHAIN_JOB_ID`` env var.
        Returns None if no job is found. Sessions with an in-memory catalog
        only see their session-local job (see get_or_create_job).
        """
        if self.catalog.in_memory:
            return self._session_job

        if Session._CURRENT_JOB:
            return Session._CURRENT_JOB

        if env_job_id := os.getenv("DATACHAIN_JOB_ID"):
            Session._CURRENT_JOB = self.catalog.metastore.get_job(env_job_id)
            if Session._CURRENT_JOB:
                Session._OWNS_JOB = False
            return Session._CURRENT_JOB

        return None

    def get_or_create_job(self) -> "Job":
        """
        Get or create the Job for this process.

        Resolution order: the already-active job, then ``DATACHAIN_JOB_ID``
        (required in Studio), then a new job created via _create_job with
        exit hooks registered to finalize it. Sessions with an in-memory
        catalog are the exception: they use a session-local, never-finalized
        job in the throwaway metastore — ``DATACHAIN_JOB_ID`` points into the
        configured metastore and is deliberately ignored, and the job is not
        shared through the process-wide cache in either direction.
        """
        if self.catalog.in_memory:
            if self._session_job is None:
                self._session_job = self._create_job()
            return self._session_job

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
            Session._CURRENT_JOB = self._create_job()
            Session._OWNS_JOB = True
            Session._JOB_STATUS = JobStatus.RUNNING

            # register cleanup hooks only once
            if not Session._JOB_HOOKS_REGISTERED:

                def _finalize_success_hook() -> None:
                    self._finalize_job_success()

                Session._JOB_FINALIZE_HOOK = _finalize_success_hook
                atexit.register(Session._JOB_FINALIZE_HOOK)
                Session._JOB_HOOKS_REGISTERED = True

        assert Session._CURRENT_JOB is not None
        return Session._CURRENT_JOB

    def _create_job(self) -> "Job":
        """Create a new job for a local run in this session's metastore."""
        query = ""
        if is_script_run():
            script = os.path.abspath(sys.argv[0])
            try:
                with open(script) as f:
                    query = f.read()
            except (OSError, UnicodeDecodeError):
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
        job = self.catalog.metastore.get_job(job_id)
        assert job is not None
        return job

    def _finalize_job_success(self):
        """Mark the current job as completed."""
        if (
            Session._CURRENT_JOB
            and Session._OWNS_JOB
            and Session._JOB_STATUS == JobStatus.RUNNING
        ):
            self.catalog.metastore.set_job_status(
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
            self.catalog.metastore.set_job_status(
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
            self.catalog.metastore.set_job_status(
                Session._CURRENT_JOB.id,
                JobStatus.FAILED,
                error_message=str(exc_value),
                error_stack=error_stack,
            )

            # Mark any incomplete dataset versions created by this job as FAILED
            self.catalog.metastore.mark_job_dataset_versions_as_failed(
                Session._CURRENT_JOB.id
            )
            # Finally clean all incomplete dataset versions
            self.catalog.cleanup_dataset_versions(job_id=Session._CURRENT_JOB.id)

            Session._JOB_STATUS = JobStatus.FAILED

    def generate_temp_dataset_name(self) -> str:
        return self.get_temp_prefix() + uuid4().hex[: self.TEMP_TABLE_UUID_LEN]

    def get_temp_prefix(self) -> str:
        return f"{SESSION_DATASET_PREFIX}{self.name}_"

    @classmethod
    def is_temp_dataset(cls, name) -> bool:
        return name.startswith(SESSION_DATASET_PREFIX)

    def _cleanup_temp_datasets(self) -> None:
        prefix = self.get_temp_prefix()
        try:
            for dataset in list(
                self.catalog.metastore.list_datasets_by_prefix(
                    prefix, include_incomplete=True
                )
            ):
                # Session datasets are always wiped — never soft-deleted.
                self.catalog.remove_dataset(
                    dataset.name, dataset.project, force=True, keep_metadata=False
                )
        # suppress error when metastore has been reset during testing
        except TableMissingError:
            pass

    @classmethod
    def get(
        cls,
        session: "Session | None" = None,
        catalog: "Catalog | None" = None,
        client_config: dict | None = None,
        in_memory: bool = False,
    ) -> "Session":
        """Resolve the session to use, by precedence:

        1. An explicit ``session=`` (validated against ``in_memory``).
        2. An explicit in-memory ``catalog=``: a wrapper session for exactly
           that catalog — never the global slot.
        3. ``in_memory=True``: the process in-memory session — never the
           global slot.
        4. The ambient session: active context, else the process-global
           session created on first use.
        5. A ``client_config`` differing from the ambient session's returns
           an owned per-config override session.

        Implicit resolution never enters contexts.
        """
        if session:
            if in_memory and not session.catalog.in_memory:
                raise ValueError(
                    "in_memory=True conflicts with the provided persistent session"
                )
            return session

        if catalog is not None and catalog.in_memory:
            return cls._get_catalog_session(catalog)

        if in_memory:
            if catalog is not None:
                raise ValueError(
                    "in_memory=True conflicts with the provided persistent catalog"
                )
            return cls._get_in_memory_session(client_config)

        session = cls._get_ambient_session(catalog, client_config)

        if client_config and session.catalog.client_config != client_config:
            session = cls._get_override_session(client_config)

        return session

    @classmethod
    def _get_ambient_session(
        cls, catalog: "Catalog | None", client_config: dict | None
    ) -> "Session":
        """The active context if any, else the process-global session,
        created on first use (this is the only place that sets it)."""
        if cls.SESSION_CONTEXTS:
            return cls.SESSION_CONTEXTS[-1]

        if cls.GLOBAL_SESSION_CTX is None:
            cls.GLOBAL_SESSION_CTX = Session(
                cls.GLOBAL_SESSION_NAME, catalog, client_config=client_config
            )
            atexit.register(cls._global_cleanup)
            cls.ORIGINAL_EXCEPT_HOOK = sys.excepthook
            sys.excepthook = cls.except_hook
        return cls.GLOBAL_SESSION_CTX

    @classmethod
    def _get_catalog_session(cls, catalog: "Catalog") -> "Session":
        """A session for an explicitly provided in-memory catalog. Cached per
        catalog; owns nothing (the caller owns the catalog)."""
        key = f"catalog:{id(catalog)}"
        session = cls.OVERRIDE_SESSIONS.get(key)
        if session is None:
            session = Session("inmemory", catalog=catalog)
            cls.OVERRIDE_SESSIONS[key] = session
        return session

    @classmethod
    def _get_in_memory_session(cls, client_config: dict | None) -> "Session":
        """Resolve an ``in_memory=True`` request.

        One implicit in-memory session exists per process; its
        ``client_config`` is frozen at creation (explicit, else inherited
        from the ambient session). A call whose effective config differs
        raises — the process-wide throwaway database cannot isolate data per
        config. Never entered as a context, never the global session.
        """
        ambient = (
            cls.SESSION_CONTEXTS[-1] if cls.SESSION_CONTEXTS else cls.GLOBAL_SESSION_CTX
        )
        if ambient is not None and ambient.catalog.in_memory:
            session, effective = ambient, client_config
        else:
            effective = client_config
            if effective is None and ambient is not None:
                effective = ambient.catalog.client_config
            if cls.IN_MEMORY_SESSION_CTX is None:
                cls.IN_MEMORY_SESSION_CTX = Session(
                    "inmemory",
                    client_config=_copy_client_config(effective),
                    in_memory=True,
                )
            session = cls.IN_MEMORY_SESSION_CTX

        if effective is not None and session.catalog.client_config != effective:
            raise ValueError(
                "the process-wide in-memory catalog is bound to a single "
                "client_config, and a different one was requested "
                "(explicitly or inherited from the ambient session)"
            )
        return session

    @classmethod
    def _get_override_session(cls, client_config: dict) -> "Session":
        """An owned, non-context session for a per-call ``client_config``
        override, so later calls' default resolution is unaffected."""
        config = _copy_client_config(client_config)
        key = repr(sorted(config.items()))
        override = cls.OVERRIDE_SESSIONS.get(key)
        if override is None:
            override = Session("sideconfig", client_config=config)
            cls.OVERRIDE_SESSIONS[key] = override
        return override

    @staticmethod
    def except_hook(exc_type, exc_value, exc_traceback):
        if Session.GLOBAL_SESSION_CTX:
            # Handle KeyboardInterrupt specially - mark as canceled and exit with
            # signal code
            if exc_type is KeyboardInterrupt:
                Session.GLOBAL_SESSION_CTX._finalize_job_as_canceled()
            else:
                Session.GLOBAL_SESSION_CTX._finalize_job_as_failed(
                    exc_type, exc_value, exc_traceback
                )
            Session.GLOBAL_SESSION_CTX.__exit__(exc_type, exc_value, exc_traceback)

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
        for override_session in cls.OVERRIDE_SESSIONS.values():
            override_session.__exit__(None, None, None)
        cls.OVERRIDE_SESSIONS.clear()
        if cls.IN_MEMORY_SESSION_CTX is not None:
            cls.IN_MEMORY_SESSION_CTX.__exit__(None, None, None)
            cls.IN_MEMORY_SESSION_CTX = None
        if cls.GLOBAL_SESSION_CTX is not None:
            cls.GLOBAL_SESSION_CTX.__exit__(None, None, None)
            cls.GLOBAL_SESSION_CTX = None
            atexit.unregister(cls._global_cleanup)

        # Reset job-related class variables
        if cls._JOB_FINALIZE_HOOK:
            try:
                atexit.unregister(cls._JOB_FINALIZE_HOOK)
            except ValueError:
                pass  # Hook was not registered
        cls._CURRENT_JOB = None
        cls._JOB_STATUS = None
        cls._OWNS_JOB = None
        cls._JOB_HOOKS_REGISTERED = False
        cls._JOB_FINALIZE_HOOK = None

        if cls.ORIGINAL_EXCEPT_HOOK:
            sys.excepthook = cls.ORIGINAL_EXCEPT_HOOK

    @staticmethod
    def _global_cleanup():
        # IN_MEMORY_SESSION_CTX is closed by the _ALL_SESSIONS loop below.
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
