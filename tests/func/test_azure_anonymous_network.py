import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATACHAIN_TEST_NETWORK"),
    reason="reads a real public container; set DATACHAIN_TEST_NETWORK=1",
)

READ = """
import time, datachain as dc
session = dc.Session.get(client_config={"account_name": "azureopendatastorage"})
for path in ["t10k-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz"]:
    f = dc.File(source="az://mnist", path=path)
    f._set_stream(session.catalog)
    t = time.time(); f.read_bytes(1024); print(time.time() - t)
"""


def test_public_container_reads_without_credentials(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("AZURE_")}
    env |= {
        "DATACHAIN_ROOT_DIR": str(tmp_path),
        "AZURE_CONFIG_DIR": str(tmp_path / "azure"),
    }
    timings = [
        float(t)
        for t in subprocess.run(  # noqa: S603
            [sys.executable, "-c", READ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        ).stdout.split()
    ]

    assert len(timings) == 2
    assert timings[1] < 1, timings
