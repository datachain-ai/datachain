import json
import os
import sys
import uuid

import pytest
import sqlalchemy as sa

from datachain.dataset import DatasetStatus
from datachain.sql.types import Float32
from tests.utils import (
    run_test_subprocess,
    skip_if_not_sqlite,
    table_row_count,
    wait_for_test_subprocess,
)

tests_dir = os.path.dirname(os.path.abspath(__file__))

python_exc = sys.executable or "python3"

E2E_STEP_TIMEOUT_SEC = 90


@pytest.mark.e2e
@pytest.mark.xdist_group(name="tmpfile")
def test_atomicity_feature_file(tmp_dir, catalog_tmpfile):
    project = catalog_tmpfile.metastore.create_project("dev", "animals")

    command = (
        python_exc,
        os.path.join(tests_dir, "scripts", "feature_class_exception.py"),
    )
    dataset = catalog_tmpfile.create_dataset(
        "existing_dataset",
        project,
        query_script="script",
        columns=[sa.Column("similarity", Float32)],
    )
    catalog_tmpfile.metastore.update_dataset_status(
        dataset, DatasetStatus.COMPLETE, version="1.0.0"
    )

    process = run_test_subprocess(
        command,
        {
            **os.environ,
            "DATACHAIN__METASTORE": catalog_tmpfile.metastore.serialize(),
            "DATACHAIN__WAREHOUSE": catalog_tmpfile.warehouse.serialize(),
        },
    )

    rc, _, stderr = wait_for_test_subprocess(process, timeout=E2E_STEP_TIMEOUT_SEC)

    assert rc == 1, stderr

    # All datasets should persist even after exceptions
    dataset_versions = list(catalog_tmpfile.list_datasets_versions())
    dataset_names = sorted([d[0].name for d in dataset_versions])
    assert len(dataset_versions) == 6

    assert dataset_names == [
        "existing_dataset",
        "global_error_class_v2",
        "global_test_datachain_v1",
        "local_test_datachain",
        "local_test_datachain_v2",
        "passed_as_argument",
    ]


@skip_if_not_sqlite
@pytest.mark.e2e
@pytest.mark.xdist_group(name="tmpfile")
def test_concurrent_save_retries_auto_version(tmp_dir, catalog_tmpfile):
    dataset_name = f"concurrent_save_{uuid.uuid4().hex}"
    barrier_dir = tmp_dir / "concurrent-save-barrier"
    script = os.path.join(tests_dir, "scripts", "concurrent_save.py")

    env = {
        **os.environ,
        "DATACHAIN__METASTORE": catalog_tmpfile.metastore.serialize(),
        "DATACHAIN__WAREHOUSE": catalog_tmpfile.warehouse.serialize(),
        "DATACHAIN_CONCURRENT_SAVE_DATASET": dataset_name,
        "DATACHAIN_CONCURRENT_SAVE_BARRIER_DIR": os.fspath(barrier_dir),
        "DATACHAIN_CONCURRENT_SAVE_PARTIES": "2",
    }

    processes = [
        run_test_subprocess(
            (python_exc, script),
            {**env, "DATACHAIN_CONCURRENT_SAVE_WORKER": str(worker_id)},
        )
        for worker_id in (1, 2)
    ]

    results = []
    for process in processes:
        rc, stdout, stderr = wait_for_test_subprocess(
            process, timeout=E2E_STEP_TIMEOUT_SEC
        )
        assert rc == 0, stderr
        results.append(json.loads(stdout.strip().splitlines()[-1]))

    versions = sorted(result["version"] for result in results)
    attempts = sorted(result["attempts"] for result in results)

    assert versions == ["1.0.0", "1.0.1"]
    assert attempts == [1, 2]

    dataset = catalog_tmpfile.get_dataset(dataset_name, include_incomplete=True)
    assert [v.version for v in dataset.versions if v.version] == ["1.0.0", "1.0.1"]

    for version in ("1.0.0", "1.0.1"):
        dataset_version = dataset.get_version(version)
        table_name = catalog_tmpfile.warehouse.dataset_table_name(dataset, version)
        assert dataset_version.status == DatasetStatus.COMPLETE
        assert table_row_count(catalog_tmpfile.warehouse.db, table_name) == 1


@skip_if_not_sqlite
@pytest.mark.e2e
@pytest.mark.xdist_group(name="tmpfile")
def test_concurrent_save_fails_after_max_retries(tmp_dir, catalog_tmpfile):
    dataset_name = f"concurrent_save_max_{uuid.uuid4().hex}"
    barrier_dir = tmp_dir / "concurrent-save-max-barrier"
    script = os.path.join(tests_dir, "scripts", "concurrent_save.py")
    process_count = 10

    env = {
        **os.environ,
        "DATACHAIN__METASTORE": catalog_tmpfile.metastore.serialize(),
        "DATACHAIN__WAREHOUSE": catalog_tmpfile.warehouse.serialize(),
        "DATACHAIN_CONCURRENT_SAVE_DATASET": dataset_name,
        "DATACHAIN_CONCURRENT_SAVE_BARRIER_DIR": os.fspath(barrier_dir),
        "DATACHAIN_CONCURRENT_SAVE_PARTIES": str(process_count),
        "DATACHAIN_CONCURRENT_SAVE_SYNC_ALL": "1",
    }

    processes = [
        run_test_subprocess(
            (python_exc, script),
            {**env, "DATACHAIN_CONCURRENT_SAVE_WORKER": str(worker_id)},
        )
        for worker_id in range(1, process_count + 1)
    ]

    results = []
    for process in processes:
        _, stdout, stderr = wait_for_test_subprocess(
            process, timeout=E2E_STEP_TIMEOUT_SEC
        )
        assert stdout.strip(), stderr
        results.append(json.loads(stdout.strip().splitlines()[-1]))

    successes = [result for result in results if result["status"] == "success"]
    failures = [result for result in results if result["status"] == "error"]

    assert len(successes) == 6
    assert len(failures) == 4
    assert sorted(result["attempts"] for result in successes) == [1, 2, 3, 4, 5, 6]
    assert sorted(result["attempts"] for result in failures) == [6, 6, 6, 6]
    assert sorted(result["version"] for result in successes) == [
        "1.0.0",
        "1.0.1",
        "1.0.2",
        "1.0.3",
        "1.0.4",
        "1.0.5",
    ]
    assert all(
        result["error_type"] == "DatasetInvalidVersionError" for result in failures
    )
    assert all(
        "Failed to claim a version" in result["error_message"] for result in failures
    )

    dataset = catalog_tmpfile.get_dataset(dataset_name, include_incomplete=True)
    assert [v.version for v in dataset.versions if v.version] == [
        "1.0.0",
        "1.0.1",
        "1.0.2",
        "1.0.3",
        "1.0.4",
        "1.0.5",
    ]

    for version in ("1.0.0", "1.0.1", "1.0.2", "1.0.3", "1.0.4", "1.0.5"):
        dataset_version = dataset.get_version(version)
        table_name = catalog_tmpfile.warehouse.dataset_table_name(dataset, version)
        assert dataset_version.status == DatasetStatus.COMPLETE
        assert table_row_count(catalog_tmpfile.warehouse.db, table_name) == 1
