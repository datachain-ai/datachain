import os
import subprocess
import sys

import pytest
import sqlalchemy as sa

from datachain.dataset import DatasetStatus
from datachain.sql.types import Float32

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
    if sys.platform == "win32":
        # Windows has a different mechanism of creating a process group.
        popen_args = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        # This is STATUS_CONTROL_C_EXIT which is equivalent to 0xC000013A
    else:
        popen_args = {"start_new_session": True}

    dataset = catalog_tmpfile.create_dataset(
        "existing_dataset",
        project,
        query_script="script",
        columns=[sa.Column("similarity", Float32)],
        create_rows=True,
    )
    catalog_tmpfile.metastore.update_dataset_status(
        dataset, DatasetStatus.COMPLETE, version="1.0.0"
    )

    process = subprocess.Popen(  # noqa: S603
        command,
        shell=False,
        encoding="utf-8",
        env={
            **os.environ,
            "DATACHAIN__METASTORE": catalog_tmpfile.metastore.serialize(),
            "DATACHAIN__WAREHOUSE": catalog_tmpfile.warehouse.serialize(),
        },
        **popen_args,
    )

    process.communicate(timeout=E2E_STEP_TIMEOUT_SEC)

    assert process.returncode == 1

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
