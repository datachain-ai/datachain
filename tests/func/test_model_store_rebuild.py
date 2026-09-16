import os
import sys

import cloudpickle
import pytest
from pydantic import Field

import datachain as dc
from tests.utils import run_test_subprocess, wait_for_test_subprocess

python_exc = sys.executable or "python3"


class Metrics(dc.DataModel):
    accuracy: float = 0.0
    latency: float = 0.0


class Contents(dc.DataModel):
    metrics1: Metrics = Field(default_factory=Metrics)
    metrics2: Metrics = Field(default_factory=Metrics)
    metrics3: Metrics = Field(default_factory=Metrics)
    metrics4: Metrics = Field(default_factory=Metrics)


class Sample(dc.DataModel):
    record_id: int = 0
    contents: Contents = Field(default_factory=Contents)


class Envelope(dc.DataModel):
    sample: Sample = Field(default_factory=Sample)
    origin: str = "builder"


def build_envelopes(record_id: int):
    nested = Metrics(accuracy=0.9 + 0.01 * record_id, latency=42.0 + record_id)
    contents = Contents(
        metrics1=nested,
        metrics2=nested,
        metrics3=nested,
        metrics4=nested,
    )
    sample = Sample(record_id=record_id, contents=contents)
    yield Envelope(sample=sample, origin="built")


def process_envelopes(envelope: Envelope):
    yield envelope


def test_nested_datamodels_round_trip_parallel(
    test_session_tmpfile,
):
    import tests.func.test_model_store_rebuild as this_module  # noqa: PLW0406

    cloudpickle.register_pickle_by_value(this_module)

    chain = (
        dc.read_values(record_id=range(1, 1001), session=test_session_tmpfile)
        .settings(parallel=2, prefetch=False)
        .gen(
            envelope=build_envelopes,
            params=["record_id"],
            output={"envelope": Envelope},
        )
        .gen(
            processed_envelope=process_envelopes,
            params=["envelope"],
            output={"processed_envelope": Envelope},
        )
    )

    rows = chain.to_list("processed_envelope")

    assert len(rows) == 1000
    for (envelope,) in rows:
        assert isinstance(envelope, Envelope)
        sample = envelope.sample
        assert isinstance(sample, Sample)
        assert isinstance(sample.contents, Contents)
        assert isinstance(sample.contents.metrics1, Metrics)
        assert sample.contents.metrics1.accuracy == pytest.approx(
            0.9 + 0.01 * sample.record_id
        )
        assert sample.contents.metrics1.latency == pytest.approx(
            42.0 + sample.record_id
        )
        assert envelope.origin == "built"


_WRITE_SCRIPT = """
from pydantic import BaseModel
import datachain as dc


class Thresholds(BaseModel):
    name: str
    limit: float


class Scenario(BaseModel):
    key: int
    thresholds: Thresholds
    threshold_items: list[Thresholds]
    optional_thresholds: Thresholds | None


dc.read_values(
    settings={"prefetch": False},
    s=[
        Scenario(
            key=1,
            thresholds=Thresholds(name="v1", limit=0.5),
            threshold_items=[Thresholds(name="v2", limit=0.75)],
            optional_thresholds=Thresholds(name="v3", limit=1.0),
        )
    ],
).save("nested_pydantic_identity")
"""


_READ_SCRIPT = """
from pydantic import BaseModel, field_validator
import datachain as dc


validated_limits = []


class Thresholds(BaseModel):
    name: str
    limit: float

    @field_validator("limit")
    @classmethod
    def record_validation(cls, value):
        validated_limits.append(value)
        return value


class Scenario(BaseModel):
    key: int
    thresholds: Thresholds
    threshold_items: list[Thresholds]
    optional_thresholds: Thresholds | None


row = dc.read_dataset("nested_pydantic_identity").to_list("s")[0][0]
assert isinstance(row, Scenario), f"row is {type(row).__module__}.{type(row).__name__}"
assert isinstance(row.thresholds, Thresholds), (
    f"row.thresholds is {type(row.thresholds).__module__}."
    f"{type(row.thresholds).__name__}"
)
assert isinstance(row.threshold_items[0], Thresholds)
assert isinstance(row.optional_thresholds, Thresholds)
assert sorted(validated_limits) == [0.5, 0.75, 1.0]
Scenario(
    key=99,
    thresholds=row.thresholds,
    threshold_items=row.threshold_items,
    optional_thresholds=row.optional_thresholds,
)
"""


@pytest.mark.e2e
@pytest.mark.xdist_group(name="tmpfile")
def test_nested_pydantic_class_identity_cross_process(tmp_dir, catalog_tmpfile):
    env = {
        **os.environ,
        "ITERATIVE_DO_NOT_TRACK": "1",
        "DATACHAIN__METASTORE": catalog_tmpfile.metastore.serialize(),
        "DATACHAIN__WAREHOUSE": catalog_tmpfile.warehouse.serialize(),
    }
    for script in (_WRITE_SCRIPT, _READ_SCRIPT):
        proc = run_test_subprocess((python_exc, "-c", script), env)
        rc, _, stderr = wait_for_test_subprocess(proc, timeout=60)
        assert rc == 0, stderr
