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
from datachain.error import JobNotFoundError, TableMissingError

if TYPE_CHECKING:
    from datachain.catalog import Catalog
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
    SESSION_CONTEXTS: ClassVar[list["Session"]] = []
    _ALL_SESSIONS: ClassVar[WeakSet["Session"]] = WeakSet()
    ORIGINAL_EXCEPT_HOOK = None

    # Job management - class-level to ensure one job per process
    _CURRENT_JOB: ClassVar["Job | None"] = None
    _JOB_STATUS: ClassVar[JobStatus | None] = None
    _OWNS_JOB: ClassVar[bool | None] = None
    _JOB_HOOKS_REGISTERED: ClassVar[bool] = False
    _JOB_FINALIZE_HOOK: ClassVar[Callable[[], None] | None] = None

    DATASET_PREFIX = "session_"
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
        Session._ALL_SESSIONS.add(self)

    def __enter__(self):
        # Push the current context onto the stack
        Session.SESSION_CONTEXTS.append(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Don't cleanup created versions on exception
        # Datasets should persist even if the session fails
        self._cleanup_temp_datasets()
        if self.is_new_catalog:
            self.catalog.metastore.close_on_exit()
            self.catalog.warehouse.close_on_exit()

        if Session.SESSION_CONTEXTS:
            Session.SESSION_CONTEXTS.pop()
        Session._ALL_SESSIONS.discard(self)

    def get_or_create_job(self) -> "Job":
        """
        Get or create a Job for this process.

        Returns:
            Job: The active Job instance.

        Behavior:
            - If a job already exists, it is returned.
            - If ``DATACHAIN_JOB_ID`` is set, the corresponding job is fetched.
            - Otherwise, a new job is created:
                * Name = absolute path to the Python script.
                * Query = empty string.
                * Parent = last job with the same name, if available.
                * Status = "running".
              Exit hooks are registered to finalize the job.

        Note:
            Job is shared across all Session instances to ensure one job per process.
        """
        if Session._CURRENT_JOB:
            return Session._CURRENT_JOB

        if env_job_id := os.getenv("DATACHAIN_JOB_ID"):
            # SaaS run: just fetch existing job
            Session._CURRENT_JOB = self.catalog.metastore.get_job(env_job_id)
            if not Session._CURRENT_JOB:
                raise JobNotFoundError(
                    f"Job {env_job_id} from DATACHAIN_JOB_ID env not found"
                )
            Session._OWNS_JOB = False
        else:
            # Local run: create new job
            if is_script_run():
                script = os.path.abspath(sys.argv[0])
            else:
                # Interactive session or module run - use unique name to avoid
                # linking unrelated sessions
                script = str(uuid4())
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

            # try to find the parent job
            parent = self.catalog.metastore.get_last_job_by_name(script)

            job_id = self.catalog.metastore.create_job(
                name=script,
                query="",
                query_type=JobQueryType.PYTHON,
                status=JobStatus.RUNNING,
                python_version=python_version,
                parent_job_id=parent.id if parent else None,
            )
            Session._CURRENT_JOB = self.catalog.metastore.get_job(job_id)
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
            self.catalog.cleanup_failed_dataset_versions(job_id=Session._CURRENT_JOB.id)

            Session._JOB_STATUS = JobStatus.FAILED

    def generate_temp_dataset_name(self) -> str:
        return self.get_temp_prefix() + uuid4().hex[: self.TEMP_TABLE_UUID_LEN]

    def get_temp_prefix(self) -> str:
        return f"{self.DATASET_PREFIX}{self.name}_"

    @classmethod
    def is_temp_dataset(cls, name) -> bool:
        return name.startswith(cls.DATASET_PREFIX)

    def _cleanup_temp_datasets(self) -> None:
        prefix = self.get_temp_prefix()
        try:
            for dataset in list(
                self.catalog.metastore.list_datasets_by_prefix(
                    prefix, include_incomplete=True
                )
            ):
                self.catalog.remove_dataset(dataset.name, dataset.project, force=True)
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
        """Creates a Session() object from a catalog.

        Parameters:
            session (Session): Optional Session(). If not provided a new session will
                    be created. It's needed mostly for simple API purposes.
            catalog (Catalog): Optional catalog. By default, a new catalog is created.
        """
        if session:
            return session

        # Access the active (most recent) context from the stack
        if cls.SESSION_CONTEXTS:
            session = cls.SESSION_CONTEXTS[-1]

        elif cls.GLOBAL_SESSION_CTX is None:
            cls.GLOBAL_SESSION_CTX = Session(
                cls.GLOBAL_SESSION_NAME,
                catalog,
                client_config=client_config,
                in_memory=in_memory,
            )
            session = cls.GLOBAL_SESSION_CTX

            atexit.register(cls._global_cleanup)
            cls.ORIGINAL_EXCEPT_HOOK = sys.excepthook
            sys.excepthook = cls.except_hook
        else:
            session = cls.GLOBAL_SESSION_CTX

        if client_config and session.catalog.client_config != client_config:
            session = Session(
                "session" + uuid4().hex[:4],
                catalog,
                client_config=client_config,
                in_memory=in_memory,
            )
            session.__enter__()

        return session

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
