import os
import sys

import pytest

from tests.utils import (
    run_test_subprocess,
    skip_if_not_sqlite,
    wait_for_test_subprocess,
)

python_exc = sys.executable or "python3"


_WRITE_SCRIPT = """
from pydantic import BaseModel
import datachain as dc


class Thresholds(BaseModel):
    name: str
    limit: float


class Scenario(BaseModel):
    key: int
    thresholds: Thresholds


dc.read_values(key=[1, 2]).map(
    s=lambda key: Scenario(key=key, thresholds=Thresholds(name="v1", limit=0.5)),
    output={"s": Scenario},
).save("nested_pydantic_identity")
"""


_READ_SCRIPT = """
from pydantic import BaseModel
import datachain as dc


class Thresholds(BaseModel):
    name: str
    limit: float


class Scenario(BaseModel):
    key: int
    thresholds: Thresholds


row = dc.read_dataset("nested_pydantic_identity").to_list("s")[0][0]
assert isinstance(row, Scenario), f"row is {type(row).__module__}.{type(row).__name__}"
assert isinstance(row.thresholds, Thresholds), (
    f"row.thresholds is {type(row.thresholds).__module__}."
    f"{type(row.thresholds).__name__}"
)
Scenario(key=99, thresholds=row.thresholds)
"""


@skip_if_not_sqlite
@pytest.mark.e2e
@pytest.mark.xdist_group(name="tmpfile")
def test_nested_pydantic_class_identity_cross_process(tmp_dir, catalog_tmpfile):
    env = {
        **os.environ,
        "DATACHAIN__METASTORE": catalog_tmpfile.metastore.serialize(),
        "DATACHAIN__WAREHOUSE": catalog_tmpfile.warehouse.serialize(),
    }
    for script in (_WRITE_SCRIPT, _READ_SCRIPT):
        proc = run_test_subprocess((python_exc, "-c", script), env)
        rc, _, stderr = wait_for_test_subprocess(proc, timeout=60)
        assert rc == 0, stderr
