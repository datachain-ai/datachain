import os
import os.path
import signal
import subprocess  # nosec B404
import uuid
from collections.abc import Generator
from datetime import datetime
from pathlib import PosixPath
from time import sleep
from typing import NamedTuple

import attrs
import pytest
import sqlalchemy
from pytest import MonkeyPatch, TempPathFactory
from upath.implementations.cloud import CloudPath

import datachain as dc
from datachain.catalog import Catalog
from datachain.catalog.loader import get_metastore, get_warehouse
from datachain.cli.utils import CommaSeparatedArgs
from datachain.config import Config, ConfigLevel
from datachain.data_storage import AbstractMetastore, JobQueryType, JobStatus
from datachain.data_storage.sqlite import (
    SQLiteDatabaseEngine,
    SQLiteMetastore,
    SQLiteWarehouse,
)
from datachain.dataset import DatasetRecord, DatasetVersion
from datachain.lib.dc import Sys
from datachain.namespace import Namespace
from datachain.project import Project
from datachain.query.session import Session
from datachain.utils import (
    ENV_DATACHAIN_GLOBAL_CONFIG_DIR,
    ENV_DATACHAIN_SYSTEM_CONFIG_DIR,
    STUDIO_URL,
    DataChainDir,
)

from .utils import DEFAULT_TREE, instantiate_tree, reset_session_job_state

DEFAULT_DATACHAIN_BIN = "datachain"
DEFAULT_DATACHAIN_GIT_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

collect_ignore = ["setup.py"]


@pytest.fixture(scope="session", autouse=True)
def add_test_env():
    os.environ["DATACHAIN_TEST"] = "true"


@pytest.fixture(autouse=True)
def global_config_dir(monkeypatch, tmp_path_factory):
    global_dir = str(tmp_path_factory.mktemp("studio-login-global"))
    monkeypatch.setenv(ENV_DATACHAIN_GLOBAL_CONFIG_DIR, global_dir)
    yield global_dir


@pytest.fixture(autouse=True)
def system_config_dir(monkeypatch, tmp_path_factory):
    system_dir = str(tmp_path_factory.mktemp("studio-login-system"))
    monkeypatch.setenv(ENV_DATACHAIN_SYSTEM_CONFIG_DIR, system_dir)
    yield system_dir


@pytest.fixture(autouse=True)
def virtual_memory(mocker):
    class VirtualMemory(NamedTuple):
        total: int
        available: int
        percent: int
        used: int
        free: int

    return mocker.patch(
        "psutil.virtual_memory",
        return_value=VirtualMemory(
            total=1073741824,
            available=5368709120,
            # prevent dumping of smaller batches to db in process_udf_outputs
            # we want to avoid this due to tests running concurrently (xdist)
            percent=50,
            used=5368709120,
            free=5368709120,
        ),
    )


@pytest.fixture(autouse=True)
def per_thread_im_mem_db(mocker, worker_id):
    if worker_id == "master":
        return
    mocker.patch(
        "datachain.data_storage.sqlite._get_in_memory_uri",
        return_value=f"file:in-mem-db-{worker_id}?mode=memory&cache=shared",
    )


@pytest.fixture(scope="session")
def monkeypatch_session() -> Generator[MonkeyPatch, None, None]:
    """
    Like monkeypatch, but for session scope.
    """
    mpatch = pytest.MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(autouse=True)
def clean_session() -> Generator[None, None, None]:
    """
    Clean leftover sessions after each test while storage handles are still open.
    """
    yield
    Session.cleanup_for_tests()


@pytest.fixture(scope="session", autouse=True)
def clean_environment(
    monkeypatch_session: MonkeyPatch,
    tmp_path_factory: TempPathFactory,
) -> None:
    """
    Make sure we have a clean environment and won't write to userspace.
    """
    working_dir = str(tmp_path_factory.mktemp("default_working_dir"))
    monkeypatch_session.chdir(working_dir)
    monkeypatch_session.delenv(DataChainDir.ENV_VAR, raising=False)
    monkeypatch_session.delenv(DataChainDir.ENV_VAR_DATACHAIN_ROOT, raising=False)


def _create_job(metastore: AbstractMetastore) -> str:
    return metastore.create_job(
        "my-job",
        'import datachain as dc; dc.read_values(num=[1, 2, 3]).save("nums")',
        query_type=JobQueryType.PYTHON,
        status=JobStatus.RUNNING,
    )


@pytest.fixture
def sqlite_db():
    if os.environ.get("DATACHAIN_METASTORE") or os.environ.get("DATACHAIN_WAREHOUSE"):
        pytest.skip("This test only runs on SQLite")
    with SQLiteDatabaseEngine.from_db_file(":memory:") as db:
        yield db
        cleanup_sqlite_db(db, [])


def cleanup_sqlite_db(
    db: SQLiteDatabaseEngine,
    cleanup_tables: list[str],
):
    # Wipe the DB after the test
    # Using new connection to check that the DB isn't locked
    tables = db.execute_str(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()

    # removing in reversed order because of foreign keys
    for table in reversed(cleanup_tables):
        db.execute_str(f"DROP TABLE IF EXISTS '{table}'")

    # Disable foreign key checks for cleanup to avoid constraint errors
    db.execute_str("PRAGMA foreign_keys = OFF")
    for (table,) in tables:
        name = table.replace("'", "''")
        db.execute_str(f"DROP TABLE IF EXISTS '{name}'")

    # Close the connection so that the SQLite file is no longer open, to avoid
    # pytest throwing: OSError: [Errno 24] Too many open files
    db.close()


@pytest.fixture
def metastore(monkeypatch):
    if os.environ.get("DATACHAIN_METASTORE"):
        _metastore = get_metastore()

        job_id = _create_job(_metastore)
        monkeypatch.setenv("DATACHAIN_JOB_ID", job_id)

        yield _metastore

        Session.cleanup_for_tests()
        _metastore.cleanup_for_tests()
    else:
        _metastore = SQLiteMetastore(db_file=":memory:")
        yield _metastore

        Session.cleanup_for_tests()
        cleanup_sqlite_db(_metastore.db.clone(), _metastore.default_table_names)

    # Close the connection so that the SQLite file is no longer open, to avoid
    # pytest throwing: OSError: [Errno 24] Too many open files
    # Or, for other implementations, prevent "too many clients" errors.
    _metastore.close_on_exit()


def check_temp_tables_cleaned_up(original_warehouse):
    """Ensure that temporary tables are cleaned up."""
    with original_warehouse.clone() as warehouse:
        assert [
            t
            for t in sqlalchemy.inspect(warehouse.db.engine).get_table_names()
            if t.startswith(
                (warehouse.UDF_TABLE_NAME_PREFIX, warehouse.TMP_TABLE_NAME_PREFIX)
            )
        ] == []


@pytest.fixture
def warehouse(metastore):
    if os.environ.get("DATACHAIN_WAREHOUSE"):
        _warehouse = get_warehouse()
        yield _warehouse
        try:
            check_temp_tables_cleaned_up(_warehouse)
        finally:
            _warehouse.cleanup_for_tests()
    else:
        _warehouse = SQLiteWarehouse(db_file=":memory:")
        yield _warehouse
        try:
            check_temp_tables_cleaned_up(_warehouse)
        finally:
            cleanup_sqlite_db(_warehouse.db.clone(), metastore.default_table_names)

            # Close the connection so that the SQLite file is no longer open, to avoid
            # pytest throwing: OSError: [Errno 24] Too many open files
            _warehouse.db.close()


@pytest.fixture
def catalog(metastore, warehouse):
    with Catalog(metastore=metastore, warehouse=warehouse) as catalog:
        yield catalog

    # Clean up job-related atexit hooks to prevent errors during pytest shutdown
    reset_session_job_state()


@pytest.fixture
def test_session(catalog):
    with Session("TestSession", catalog=catalog) as session:
        yield session

    # Clean up job-related atexit hooks to prevent errors during pytest shutdown
    reset_session_job_state()


@pytest.fixture
def metastore_tmpfile(monkeypatch, tmp_path):
    if os.environ.get("DATACHAIN_METASTORE"):
        _metastore = get_metastore()

        job_id = _create_job(_metastore)
        monkeypatch.setenv("DATACHAIN_JOB_ID", job_id)

        yield _metastore

        Session.cleanup_for_tests()
        _metastore.cleanup_for_tests()
    else:
        _metastore = SQLiteMetastore(db_file=str(tmp_path / "test.db"))
        yield _metastore

        Session.cleanup_for_tests()
        cleanup_sqlite_db(_metastore.db.clone(), _metastore.default_table_names)

    # Close the connection so that the SQLite file is no longer open, to avoid
    # pytest throwing: OSError: [Errno 24] Too many open files
    # Or, for other implementations, prevent "too many clients" errors.
    _metastore.close_on_exit()


@pytest.fixture
def warehouse_tmpfile(tmp_path, metastore_tmpfile):
    if os.environ.get("DATACHAIN_WAREHOUSE"):
        _warehouse = get_warehouse()
        yield _warehouse
        try:
            check_temp_tables_cleaned_up(_warehouse)
        finally:
            _warehouse.cleanup_for_tests()
    else:
        _warehouse = SQLiteWarehouse(db_file=str(tmp_path / "test.db"))
        yield _warehouse
        try:
            check_temp_tables_cleaned_up(_warehouse)
        finally:
            cleanup_sqlite_db(
                _warehouse.db.clone(), metastore_tmpfile.default_table_names
            )

            # Close the connection so that the SQLite file is no longer open, to avoid
            # pytest throwing: OSError: [Errno 24] Too many open files
            _warehouse.db.close()


@pytest.fixture
def catalog_tmpfile(metastore_tmpfile, warehouse_tmpfile):
    # For testing parallel and distributed processing, as these cannot use
    # in-memory databases.
    with Catalog(metastore=metastore_tmpfile, warehouse=warehouse_tmpfile) as catalog:
        yield catalog


@pytest.fixture
def test_session_tmpfile(catalog_tmpfile):
    # For testing parallel and distributed processing, as these cannot use
    # in-memory databases.
    with Session("TestSession", catalog=catalog_tmpfile) as session:
        yield session

    # Clean up job-related atexit hooks to prevent errors during pytest shutdown
    reset_session_job_state()


@pytest.fixture
def tmp_dir(tmp_path_factory, monkeypatch):
    dpath = tmp_path_factory.mktemp("datachain-test")
    monkeypatch.chdir(dpath)
    return dpath


def pytest_addoption(parser):
    parser.addoption(
        "--disable-remotes",
        action=CommaSeparatedArgs,
        default=[],
        help="Comma separated list of remotes to disable",
    )

    parser.addoption(
        "--datachain-bin",
        type=str,
        default=DEFAULT_DATACHAIN_BIN,
        help="Path to datachain binary",
    )

    parser.addoption(
        "--datachain-revs",
        type=str,
        help="Comma-separated list of DataChain revisions to test "
        "(overrides `--datachain-bin`)",
    )

    parser.addoption(
        "--datachain-git-repo",
        type=str,
        default=DEFAULT_DATACHAIN_GIT_REPO,
        help="Path or url to datachain git repo",
    )
    parser.addoption(
        "--azure-connection-string",
        type=str,
        help=(
            "Connection string to run tests against a real, versioned "
            "Azure storage account"
        ),
    )


class DataChainTestConfig:
    def __init__(self):
        self.datachain_bin = DEFAULT_DATACHAIN_BIN
        self.datachain_revs = None
        self.datachain_git_repo = DEFAULT_DATACHAIN_GIT_REPO


@pytest.fixture(scope="session")
def test_config(request):
    return request.config.datachain_config


def pytest_configure(config):
    config.datachain_config = DataChainTestConfig()

    config.datachain_config.datachain_bin = config.getoption("--datachain-bin")
    config.datachain_config.datachain_revs = config.getoption("--datachain-revs")
    config.datachain_config.datachain_git_repo = config.getoption(
        "--datachain-git-repo"
    )


@pytest.fixture(scope="session", params=[DEFAULT_TREE])
def tree(request):
    return request.param


@attrs.define
class CloudServer:
    kind: str
    src: CloudPath
    client_config: dict[str, str]

    @property
    def src_uri(self):
        if self.kind == "file":
            return self.src.as_uri()
        return str(self.src).rstrip("/")


def make_cloud_server(src_path, cloud_type, tree):
    fs = src_path.fs
    if cloud_type == "s3":
        endpoint_url = fs.client_kwargs["endpoint_url"]
        client_config = {"aws_endpoint_url": endpoint_url}
    elif cloud_type in ("gs", "gcs"):
        endpoint_url = fs._endpoint
        client_config = {"endpoint_url": endpoint_url}
    elif cloud_type == "azure":
        client_config = fs.storage_options.copy()
    elif cloud_type == "file":
        client_config = {}
    else:
        raise ValueError(f"invalid cloud_type: {cloud_type}")

    instantiate_tree(src_path, tree)
    return CloudServer(kind=cloud_type, src=src_path, client_config=client_config)


@attrs.define
class CloudTestCatalog:
    server: CloudServer
    working_dir: PosixPath
    catalog: Catalog

    @property
    def src(self):
        return self.server.src

    @property
    def src_uri(self):
        return self.server.src_uri

    @property
    def client_config(self):
        return self.server.client_config

    @property
    def session(self) -> Session:
        return Session("CTCSession", catalog=self.catalog)


cloud_types = ["s3", "gs", "azure"]


@pytest.fixture(scope="session", params=["file", *cloud_types])
def cloud_type(request):
    return request.param


@pytest.fixture(scope="session", params=[False, True])
def version_aware(request):
    return request.param


def pytest_collection_modifyitems(config, items):
    disabled_remotes = config.getoption("--disable-remotes")
    if not disabled_remotes:
        return

    for item in items:
        if "cloud_server" in item.fixturenames:
            cloud_type = item.callspec.params.get("cloud_type")
            if cloud_type not in cloud_types:
                continue
            if "all" in disabled_remotes:
                reason = "Skipping all tests requiring cloud"
                item.add_marker(pytest.mark.skip(reason=reason))
            if cloud_type in disabled_remotes:
                reason = f"Skipping all tests for {cloud_type=}"
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def cloud_server(request, tmp_upath_factory, cloud_type, version_aware, tree):
    if cloud_type == "azure" and version_aware:
        if conn_str := request.config.getoption("--azure-connection-string"):
            src_path = tmp_upath_factory.azure(conn_str)
        else:
            pytest.skip("Can't test versioning with Azure")
    elif cloud_type == "file":
        if version_aware:
            pytest.skip("Local storage can't be versioned")
        else:
            src_path = tmp_upath_factory.mktemp("local")
    else:
        src_path = tmp_upath_factory.mktemp(cloud_type, version_aware=version_aware)
    return make_cloud_server(src_path, cloud_type, tree)


@pytest.fixture
def datachain_job_id(test_session, monkeypatch):
    yield _create_job(test_session.catalog.metastore)


@pytest.fixture(scope="session")
def cloud_server_credentials(cloud_server):
    if cloud_server.kind == "s3":
        cfg = cloud_server.src.fs.client_kwargs
        try:
            os.environ.pop("AWS_PROFILE")
        except KeyError:
            pass
        os.environ["AWS_ACCESS_KEY_ID"] = cfg.get("aws_access_key_id")
        os.environ["AWS_SECRET_ACCESS_KEY"] = cfg.get("aws_secret_access_key")
        os.environ["AWS_SESSION_TOKEN"] = cfg.get("aws_session_token")
        os.environ["AWS_DEFAULT_REGION"] = cfg.get("region_name")


def get_cloud_test_catalog(cloud_server, tmp_path, metastore, warehouse):
    cache_dir = tmp_path / ".datachain" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmpfile_dir = tmp_path / ".datachain" / "tmp"
    tmpfile_dir.mkdir(exist_ok=True)

    catalog = Catalog(
        metastore=metastore,
        warehouse=warehouse,
        cache_dir=str(cache_dir),
        tmp_dir=str(tmpfile_dir),
        client_config=cloud_server.client_config,
    )
    return CloudTestCatalog(server=cloud_server, working_dir=tmp_path, catalog=catalog)


@pytest.fixture
def cloud_test_catalog(
    cloud_server,
    cloud_server_credentials,
    tmp_path,
    metastore,
    warehouse,
):
    catalog = get_cloud_test_catalog(cloud_server, tmp_path, metastore, warehouse)
    try:
        yield catalog
    finally:
        # Ensure catalog shuts down its metastore/warehouse connections between tests
        catalog.catalog.close()
        reset_session_job_state()


@pytest.fixture
def cloud_test_catalog_upload(cloud_test_catalog):
    """This returns a version of the cloud_test_catalog that is suitable for uploading
    files, and will perform the necessary cleanup of any uploaded files."""
    from datachain.client.fsspec import Client

    src = cloud_test_catalog.src_uri
    client = Client.get_implementation(src)
    fsspec_fs = client.create_fs(**cloud_test_catalog.client_config)
    original_paths = set(fsspec_fs.ls(src))

    yield cloud_test_catalog

    # Cleanup any written files
    new_paths = set(fsspec_fs.ls(src))
    cleanup_paths = new_paths - original_paths
    for p in cleanup_paths:
        fsspec_fs.rm(p, recursive=True)


@pytest.fixture
def cloud_test_catalog_tmpfile(
    cloud_server,
    cloud_server_credentials,
    tmp_path,
    metastore_tmpfile,
    warehouse_tmpfile,
):
    catalog = get_cloud_test_catalog(
        cloud_server,
        tmp_path,
        metastore_tmpfile,
        warehouse_tmpfile,
    )
    try:
        yield catalog
    finally:
        # Clean up job-related atexit hooks to prevent errors during pytest shutdown
        from tests.utils import reset_session_job_state

        catalog.catalog.close()
        reset_session_job_state()


@pytest.fixture
def is_studio():
    return True


@pytest.fixture(autouse=True)
def mock_is_studio(monkeypatch, is_studio):
    if is_studio:
        monkeypatch.setenv("DATACHAIN_IS_STUDIO", "True")
    yield


@pytest.fixture
def project(test_session):
    return dc.create_project("dev", "animals", "Animals project")


@pytest.fixture
def listed_bucket(cloud_test_catalog):
    ctc = cloud_test_catalog
    dc.read_storage(ctc.src_uri, session=ctc.session).exec()


@pytest.fixture
def animal_dataset(listed_bucket, cloud_test_catalog):
    name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri
    dataset = catalog.create_dataset_from_sources(
        name, [src_uri], catalog.metastore.default_project, recursive=True
    )
    return catalog.update_dataset(
        dataset, {"description": "animal dataset", "attrs": ["cats", "dogs"]}
    )


@pytest.fixture
def dogs_dataset(listed_bucket, cloud_test_catalog):
    name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri
    dataset = catalog.create_dataset_from_sources(
        name,
        [f"{src_uri}/dogs/"],
        catalog.metastore.default_project,
        recursive=True,
    )
    return catalog.update_dataset(
        dataset, {"description": "dogs dataset", "attrs": ["dogs", "dataset"]}
    )


@pytest.fixture
def cats_dataset(listed_bucket, cloud_test_catalog):
    name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri
    dataset = catalog.create_dataset_from_sources(
        name, [f"{src_uri}/cats/*"], catalog.metastore.default_project, recursive=True
    )
    return catalog.update_dataset(
        dataset, {"description": "cats dataset", "attrs": ["cats", "dataset"]}
    )


@pytest.fixture
def dataset_record():
    return DatasetRecord(
        id=1,
        name=f"ds_{uuid.uuid4().hex}",
        description="",
        attrs=[],
        versions=[
            DatasetVersion(
                id=1,
                uuid=uuid.uuid4().hex,
                dataset_id=1,
                version="1.0.0",
                status=1,
                created_at=datetime.now(),
                finished_at=datetime.now(),
                error_message="",
                error_stack="",
                num_objects=6,
                size=100,
                feature_schema=None,
                script_output="",
                schema=None,
                _preview_data=[],
            )
        ],
        status=1,
        schema={},
        feature_schema={},
        project=Project(
            id=1,
            uuid=str(uuid.uuid4()),
            name="animals",
            created_at=datetime.now(),
            descr="",
            namespace=Namespace(
                id=1,
                uuid=str(uuid.uuid4()),
                name="dev",
                created_at=datetime.now(),
                descr="",
            ),
        ),
    )


@pytest.fixture
def dataset_rows():
    int_example = 25
    return [
        {
            "id": i,
            "location": "",
            "source": "s3://my-bucket",
            "dir_type": 0,
            "parent": "input/text_emd_1m",
            "version": "7e589b7d-382c-49a5-931f-2b999c930c5e",
            "is_latest": True,
            "name": f"dql_1m_meta_text_emd.parquet_3_{i}_0.snappy.parquet",
            "etag": f"72b35c8e9b8eed1636c91eb94241c2f8-{i}",
            "last_modified": "2024-02-23T10:42:31.842944+00:00",
            "size": 49807360,
            "sys__rand": 12123123123,
            "int_col": 5,
            "int_col_32": 5,
            "int_col_64": 5,
            "float_col": 0.5,
            "float_col_32": 0.5,
            "float_col_64": 0.5,
            "array_col": [0.5],
            "array_col_nested": [[0.5]],
            "array_col_32": [0.5],
            "array_col_64": [0.5],
            "string_col": "a string",
            "bool_col": True,
            "json_col": '{"a": 1}',
            "binary_col": int_example.to_bytes(2, "big"),
        }
        for i in range(19)
    ]


@pytest.fixture
def studio_token():
    with Config(ConfigLevel.GLOBAL).edit() as conf:
        conf["studio"] = {"token": "isat_access_token", "team": "team_name"}


@pytest.fixture
def studio_datasets(requests_mock, studio_token):
    common_version_info = {
        "status": 1,
        "created_at": "2024-02-23T10:42:31.842944+00:00",
        "finished_at": "2024-02-23T10:42:31.842944+00:00",
        "error_message": "",
        "error_stack": "",
        "num_objects": 6,
        "size": 100,
    }

    project = {
        "id": 1,
        "uuid": str(uuid.uuid4()),
        "name": "animals",
        "created_at": "2024-02-23T10:42:31.842944+00:00",
        "descr": "",
        "namespace": {
            "id": 1,
            "name": "dev",
            "uuid": str(uuid.uuid4()),
            "created_at": "2024-02-23T10:42:31.842944+00:00",
            "descr": "",
        },
    }

    dogs_dataset = {
        "id": 1,
        "name": "dogs",
        "description": "dogs dataset",
        "attrs": ["dogs", "dataset"],
        "project": project,
        "versions": [
            {
                "version": "1.0.0",
                "id": 1,
                "uuid": "dab73bdf-ceb3-4af3-8e01-1d44eb41acf9",
                "dataset_id": 1,
                **common_version_info,
            },
            {
                "version": "2.0.0",
                "id": 2,
                "uuid": "dab73bdf-ceb3-4af3-8e01-1d44eb41acf8",
                "dataset_id": 1,
                **common_version_info,
            },
        ],
    }

    datasets = [
        dogs_dataset,
        {
            "id": 2,
            "name": "cats",
            "description": "cats dataset",
            "attrs": ["cats", "dataset"],
            "project": project,
            "versions": [
                {
                    "version": "1.0.0",
                    "id": 3,
                    "uuid": "dab73bdf-ceb3-4af3-8e01-1d44eb41acf7",
                    "dataset_id": 2,
                    **common_version_info,
                },
            ],
        },
        {
            "id": 3,
            "name": "both",
            "description": "both dataset",
            "attrs": ["both", "dataset"],
            "project": project,
            "versions": [
                {
                    "version": "1.0.0",
                    "id": 4,
                    "uuid": "dab73bdf-ceb3-4af3-8e01-1d44eb41acf6",
                    "dataset_id": 3,
                    **common_version_info,
                },
            ],
        },
    ]

    requests_mock.get(f"{STUDIO_URL}/api/datachain/datasets", json=datasets)
    requests_mock.get(
        (
            f"{STUDIO_URL}/api/datachain/datasets/info?namespace=dev&project=animals"
            "&name=dogs&team_name=team_name"
        ),
        json=dogs_dataset,
    )


@pytest.fixture
def not_random_ds(test_session):
    # `sys__rand` column is carefully crafted to ensure that `train_test_split` func
    # will always return columns in the `sys__id` order if no seed is provided.
    return dc.read_records(
        [
            {"sys__id": 1, "sys__rand": 8025184816406567794, "fib": 0},
            {"sys__id": 2, "sys__rand": 8264763963075908010, "fib": 1},
            {"sys__id": 3, "sys__rand": 338514328625642097, "fib": 1},
            {"sys__id": 4, "sys__rand": 508807229144041274, "fib": 2},
            {"sys__id": 5, "sys__rand": 8730460072520445744, "fib": 3},
            {"sys__id": 6, "sys__rand": 154987448000528066, "fib": 5},
            {"sys__id": 7, "sys__rand": 6310705427500864020, "fib": 8},
            {"sys__id": 8, "sys__rand": 2154127460471345108, "fib": 13},
            {"sys__id": 9, "sys__rand": 2584481985215516118, "fib": 21},
            {"sys__id": 10, "sys__rand": 5771949255753972681, "fib": 34},
        ],
        session=test_session,
        schema={"sys": Sys, "fib": int},
    )


@pytest.fixture
def pseudo_random_ds(test_session):
    return dc.read_records(
        [
            {"sys__id": 1, "sys__rand": 2406827533654413759, "fib": 0},
            {"sys__id": 2, "sys__rand": 743035223448130834, "fib": 1},
            {"sys__id": 3, "sys__rand": 8572034894545971037, "fib": 1},
            {"sys__id": 4, "sys__rand": 3413911135601125438, "fib": 2},
            {"sys__id": 5, "sys__rand": 8036488725627198326, "fib": 3},
            {"sys__id": 6, "sys__rand": 2020789040280779494, "fib": 5},
            {"sys__id": 7, "sys__rand": 8478782014085172114, "fib": 8},
            {"sys__id": 8, "sys__rand": 1374262678671783922, "fib": 13},
            {"sys__id": 9, "sys__rand": 7728884931956308771, "fib": 21},
            {"sys__id": 10, "sys__rand": 5591681088079559562, "fib": 34},
        ],
        session=test_session,
        schema={"sys": Sys, "fib": int},
    )


@pytest.fixture()
def run_datachain_worker(monkeypatch):
    if not os.environ.get("DATACHAIN_DISTRIBUTED"):
        pytest.skip("Distributed tests are disabled")

    job_id = os.environ.get("DATACHAIN_JOB_ID")
    assert job_id, "DATACHAIN_JOB_ID environment variable is required for this test"

    monkeypatch.delenv("DATACHAIN_DISTRIBUTED_DISABLED", raising=False)

    # This worker can take several tasks in parallel, as it's very handy
    # for testing, where we don't want [yet] to constrain the number of
    # available workers.
    workers: list[subprocess.Popen] = []
    queues: list[str] = []
    for i in range(2):
        queue_name = f"udf-{uuid.uuid4()}"
        worker_cmd = [
            "celery",
            "-A",
            "datachain_worker.tasks",
            "worker",
            "--loglevel=INFO",
            f"--hostname=tests-datachain-worker-udf-runner-{i}",
            "--pool=solo",
            "--concurrency=1",
            "--max-tasks-per-child=1",
            "--prefetch-multiplier=1",
            "-Q",
            queue_name,
        ]
        queues.append(queue_name)
        print(f"Starting worker with command: {' '.join(worker_cmd)}")
        worker_proc = subprocess.Popen(  # noqa: S603
            worker_cmd,
            env=os.environ,
            shell=False,
        )
        workers.append(worker_proc)
    try:
        from datachain_worker.utils.celery import celery_app

        inspect = celery_app.control.inspect()
        attempts = 0
        # Wait 10 seconds for the Celery worker(s) to be up
        while not inspect.active() and attempts < 10:
            print("Waiting for Celery worker(s) to start...")
            sleep(1)
            attempts += 1

        if attempts == 10:
            raise RuntimeError("Celery worker(s) did not start in time")

        monkeypatch.setenv("DATACHAIN_STEP_ID", "1")
        monkeypatch.setenv("UDF_RUNNER_QUEUE_NAME_LIST", ",".join(queues))

        yield workers
    finally:
        for worker in workers:
            print(f"Stopping worker {worker.pid}")
            os.kill(worker.pid, signal.SIGTERM)
        for worker in workers:
            try:
                worker.wait(timeout=30)  # seconds
            except subprocess.TimeoutExpired:
                os.kill(worker.pid, signal.SIGKILL)


# Common constants for remote dataset testing
REMOTE_DATASET_UUID = "20f5a2f1-fc9a-4e36-8b91-5a530f289451"
REMOTE_DATASET_UUID_V2 = "30f5a2f1-fc9a-4e36-8b91-5a530f289452"
REMOTE_NAMESPACE_UUID = "efbc3e84-3eb6-4be1-bec1-704b939e1fe4"
REMOTE_PROJECT_UUID = "0ed3a6c6-f2f7-45aa-869b-39219c86a9d4"

REMOTE_NAMESPACE_NAME = "dev"
REMOTE_PROJECT_NAME = "animals"


@pytest.fixture
def remote_dataset_schema():
    """Common schema for remote datasets."""
    return {
        "id": {"type": "UInt64"},
        "sys__rand": {"type": "UInt64"},
        "file__path": {"type": "String"},
        "file__etag": {"type": "String"},
        "file__version": {"type": "String"},
        "file__is_latest": {"type": "Boolean"},
        "file__last_modified": {"type": "DateTime"},
        "file__size": {"type": "Int64"},
        "file__location": {"type": "String"},
        "file__source": {"type": "String"},
        "version": {"type": "String"},
    }


@pytest.fixture
def remote_file_feature_schema():
    """Common File feature schema for remote datasets."""
    return {
        "file": "File@v1",
        "version": "str",
        "_custom_types": {
            "File@v1": {
                "schema_version": 2,
                "name": "File@v1",
                "fields": {
                    "source": "str",
                    "path": "str",
                    "size": "int",
                    "version": "str",
                    "etag": "str",
                    "is_latest": "bool",
                    "last_modified": "datetime",
                    "location": "Union[dict, list[dict], NoneType]",
                },
                "bases": [
                    ["File", "datachain.lib.file", "File@v1"],
                    ["DataModel", "datachain.lib.data_model", "DataModel@v1"],
                    ["BaseModel", "pydantic.main", None],
                    ["object", "builtins", None],
                ],
                "hidden_fields": [
                    "source",
                    "version",
                    "etag",
                    "is_latest",
                    "last_modified",
                    "location",
                ],
            }
        },
    }


@pytest.fixture
def remote_namespace():
    """Remote namespace fixture for Studio API mocking."""
    return {
        "id": 1,
        "uuid": REMOTE_NAMESPACE_UUID,
        "name": REMOTE_NAMESPACE_NAME,
        "descr": "Dev namespace",
        "created_at": "2024-02-23T10:42:31.842944+00:00",
    }


@pytest.fixture
def remote_project(remote_namespace):
    """Remote project fixture for Studio API mocking."""
    return {
        "id": 1,
        "uuid": REMOTE_PROJECT_UUID,
        "name": REMOTE_PROJECT_NAME,
        "descr": "Animals project",
        "created_at": "2024-02-23T10:42:31.842944+00:00",
        "namespace": remote_namespace,
    }


@pytest.fixture
def compressed_parquet_data():
    """
    Factory fixture that creates lz4-compressed parquet for datasets.
    Returns a function that can be called with different data.
    """
    import io
    from datetime import datetime

    import lz4.frame
    import pandas as pd

    def create_compressed_parquet(data, src_uri=None):
        def _adapt_row(row):
            """
            Adjusting row values to match remote response
            """
            adapted = {}
            for k, v in row.items():
                if isinstance(v, datetime):
                    adapted[k] = v.timestamp()
                elif v is None:
                    adapted[k] = ""
                else:
                    adapted[k] = v

            adapted["sys__id"] = 1
            adapted["sys__rand"] = 1
            adapted["file__location"] = ""
            adapted["file__source"] = src_uri or ""
            adapted["file__version"] = ""
            return adapted

        adapted_data = [_adapt_row(row) for row in data]
        df = pd.DataFrame.from_records(adapted_data)
        buffer = io.BytesIO()
        df.to_parquet(buffer, engine="auto")

        return lz4.frame.compress(buffer.getvalue())

    return create_compressed_parquet


@pytest.fixture
def dog_entries():
    """Factory function to create version-specific dog entries."""
    from tests.data import ENTRIES

    def _create_dog_entries(version="1.0.0"):
        return [
            {
                "file__path": e.path,
                "file__etag": e.etag,
                "file__version": e.version,
                "file__is_latest": e.is_latest,
                "file__last_modified": e.last_modified,
                "file__size": e.size,
                "version": version,
            }
            for e in ENTRIES
            if e.name.startswith("dog")
        ]

    return _create_dog_entries


PRIMES_UP_TO_73 = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
)


@pytest.fixture
def numbers_ds(test_session) -> Generator[DatasetRecord, None, None]:
    numbers = list(range(1, 74))
    ds = dc.read_values(
        number=numbers,
        parity=["odd" if n % 2 else "even" for n in numbers],
        primality=["prime" if n in PRIMES_UP_TO_73 else "composite" for n in numbers],
        last_digit=[n % 10 for n in numbers],
        session=test_session,
    ).save("numbers_dataset")
    assert ds.dataset is not None
    yield ds.dataset
    dc.delete_dataset(ds.dataset.name, force=True)


@pytest.fixture
def numbers_table(warehouse, numbers_ds) -> Generator[sqlalchemy.Table, None, None]:
    table_name = warehouse.dataset_table_name(numbers_ds, numbers_ds.latest_version)
    yield warehouse.get_table(table_name)


@pytest.fixture
def mock_parquet_data(compressed_parquet_data, dog_entries, version="1.0.0"):
    return compressed_parquet_data(dog_entries(version))


@pytest.fixture
def mock_parquet_data_cloud(compressed_parquet_data, dog_entries, cloud_test_catalog):
    src_uri = cloud_test_catalog.src_uri
    return compressed_parquet_data(dog_entries("1.0.0"), src_uri)
