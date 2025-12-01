import cloudpickle
import pytest
from pydantic import Field

import datachain as dc


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
