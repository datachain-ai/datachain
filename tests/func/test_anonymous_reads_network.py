import os
import subprocess
import sys

import pytest

PUBLIC_URI = "gs://gresearch/robotics/droid_raw/1.0.1/AUTOLab/success/2023-07-07/"

SAVE = f"""
import datachain as dc
dc.read_storage({PUBLIC_URI!r}, update=True).save("ep")
"""

READ = """
import datachain as dc
files = list(dc.read_dataset("ep").limit(4).to_values("file"))
assert len(files) == 4
for file in files:
    assert file.read_bytes(1024)
"""


def _anon_env(tmp_path):
    # Hide local credentials (env vars, gcloud ADC) so reads are anonymous.
    drop_prefix = (
        "GCSFS_",
        "GOOGLE_",
        "DATACHAIN_GCP_",
        "DATACHAIN_METASTORE",
        "DATACHAIN_WAREHOUSE",
        "DATACHAIN__METASTORE",
        "DATACHAIN__WAREHOUSE",
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop_prefix)}
    env |= {
        "DATACHAIN_ROOT_DIR": str(tmp_path),
        "DATACHAIN_DIR": str(tmp_path / ".datachain"),
        "CLOUDSDK_CONFIG": str(tmp_path / "gcloud"),
    }
    return env


def _run(code, env):
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )


def test_smoke_environment_ignores_inherited_backends(tmp_path, monkeypatch):
    caller_db = tmp_path / "caller.db"
    inherited = {
        "DATACHAIN_METASTORE": "datachain.data_storage.sqlite.SQLiteMetastore",
        "DATACHAIN_WAREHOUSE": "datachain.data_storage.sqlite.SQLiteWarehouse",
        "DATACHAIN_METASTORE_ARG_DB_FILE": str(caller_db),
        "DATACHAIN_WAREHOUSE_ARG_DB_FILE": str(caller_db),
        "DATACHAIN__METASTORE": "invalid inherited metastore",
        "DATACHAIN__WAREHOUSE": "invalid inherited warehouse",
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    env = _anon_env(tmp_path / "smoke")
    assert not inherited.keys() & env.keys()
    _run('import datachain as dc; dc.read_values(x=[1]).save("ep")', env)
    assert not caller_db.exists()


@pytest.mark.skipif(
    not os.environ.get("DATACHAIN_TEST_NETWORK"),
    reason="reads a real public bucket; set DATACHAIN_TEST_NETWORK=1",
)
def test_public_bucket_reads_in_new_session(tmp_path):
    env = _anon_env(tmp_path)
    _run(SAVE, env)
    _run(READ, env)
