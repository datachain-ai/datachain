import logging

import datachain as dc
from datachain import Session

logging.basicConfig(level=logging.INFO)

# Dataset with variable session. Will persist.
session = Session("asVariable")
dv = dc.read_values(key=["a", "b", "c"], session=session)
dv.save("passed_as_argument")

# A datachain created in global context.
# This will persist even with global exception.
dc.read_values(key=["a", "b", "c"]).save("global_test_datachain_v1")

with Session("local") as session_local:
    # A datachain created in local context.
    # This will persist.
    dc.read_values(key=["a", "b", "c"], session=session_local).save(
        "local_test_datachain"
    )

try:
    with Session("localfailure") as session_failure:
        # A datachain created in local context.
        # This will persist even with local exception.
        dc.read_values(key=["a", "b", "c"], session=session_failure).save(
            "local_test_datachain_v2"
        )
        raise ValueError("Local failure class")
except ValueError:
    pass

# We return to global context.
# This will persist even with global exception.
dc.read_values(key=["a", "b", "c"]).save("global_error_class_v2")

raise Exception("This is a test exception")
