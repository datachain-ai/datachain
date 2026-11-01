import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATACHAIN_TEST_NETWORK"),
    reason="reads a real public bucket; set DATACHAIN_TEST_NETWORK=1",
)

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


def _anon_env(tmp_path, drop_prefix):
    # Hide local credentials (env vars, gcloud ADC) so reads are anonymous.
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop_prefix)}
    env |= {
        "DATACHAIN_ROOT_DIR": str(tmp_path),
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


def test_public_bucket_reads_in_new_session(tmp_path):
    env = _anon_env(tmp_path, ("GCSFS_", "GOOGLE_", "DATACHAIN_GCP_"))
    _run(SAVE, env)
    _run(READ, env)
