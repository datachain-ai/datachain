"""Example: Nested DataModels with parallel execution.

Demonstrates mapping a function that returns a nested DataModel (a DataModel
containing other DataModels).

The example keeps things minimal: we persist a tiny dataset, run a parallel map
that returns a nested DataModel, and display the result.
"""

from pydantic import Field

import datachain as dc


class Metric(dc.DataModel):
    """Represents a single computed metric with quality metadata."""

    value: float | None = Field(default=None, description="Computed metric value")
    confidence: float | None = Field(
        default=None, description="Confidence / quality score"
    )
    status: str | None = Field(default=None, description="Processing status label")
    metric_error: str | None = Field(
        default=None, description="Error message if metric computation failed"
    )


class SampleMetrics(dc.DataModel):
    """Container for two illustrative nested metrics.

    Each sub-field is its own DataModel instance to demonstrate nested schemas
    """

    metric_primary: Metric = Field(default_factory=Metric, description="Primary metric")
    metric_secondary: Metric = Field(
        default_factory=Metric, description="Secondary metric"
    )


def generate_sample_metrics() -> SampleMetrics:
    """Synthesize a pair of metrics.

    In real scenarios you'd compute these values; here we just return constants
    to keep the example deterministic.
    """

    return SampleMetrics(
        metric_primary=Metric(value=50.0, confidence=0.95, status="ok"),
    )


def main():
    (
        dc.read_values(record_id=[1, 2])
        .settings(parallel=2)  # Keep it parallel to test serialization
        .map(metrics=generate_sample_metrics)
        .save("nested_datamodel")
    )

    dc.read_dataset("nested_datamodel").show()

    print(dc.read_dataset("nested_datamodel").to_values("metrics"))


if __name__ == "__main__":
    main()
