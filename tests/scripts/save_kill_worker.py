"""Save a small chain, optionally killing this process with SIGKILL mid-UDF or
right after the save's copy into its reserved ds_ table."""

import os
import signal
import sys

import datachain as dc
from datachain.query.session import Session

kill_at, version = sys.argv[1], sys.argv[2]


def double(num: int) -> int:
    if kill_at == "udf" and num == 3:
        os.kill(os.getpid(), signal.SIGKILL)
    return num * 2


if kill_at == "copy":
    warehouse_class = type(Session.get().catalog.warehouse)
    insert_into = warehouse_class.insert_into

    def insert_then_die(self, table, *args, **kwargs):
        insert_into(self, table, *args, **kwargs)
        if table.name.startswith("ds_") and "_result_" in table.name:
            os.kill(os.getpid(), signal.SIGKILL)

    warehouse_class.insert_into = insert_then_die  # type: ignore[method-assign]

chain = dc.read_values(num=[1, 2, 3, 4, 5]).map(doubled=double)
chain.save("result", version=None if version == "auto" else version)
