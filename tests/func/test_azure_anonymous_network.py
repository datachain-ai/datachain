import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATACHAIN_TEST_NETWORK"),
    reason="reads a real public container; set DATACHAIN_TEST_NETWORK=1",
)

READ = """
import json, datachain as dc
session = dc.Session.get(client_config={"account_name": "azureopendatastorage"})
prefixes = []
for path in ["t10k-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz"]:
    f = dc.File(source="az://mnist", path=path)
    f._set_stream(session.catalog)
    assert session.catalog.get_client(f.source).fs.anon
    prefixes.append(f.read_bytes(2).hex())
print(json.dumps(prefixes))
"""


def test_public_container_reads_without_credentials(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("AZURE_")}
    env |= {
        "DATACHAIN_ROOT_DIR": str(tmp_path),
        "AZURE_CONFIG_DIR": str(tmp_path / "azure"),
    }
    env["HOME"] = str(tmp_path)
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", READ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == ["1f8b", "1f8b"]
