import os

metrics: dict[str, str | int | float | bool | None] = {}


def set(key: str, value: str | int | float | bool | None) -> None:  # noqa: PYI041
    """Set a metric value."""
    if not isinstance(key, str):
        raise TypeError("Key must be a string")
    if not key:
        raise ValueError("Key must not be empty")
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError("Value must be a string, int, float or bool")
    metrics[key] = value

    if job_id := os.getenv("DATACHAIN_JOB_ID"):
        from datachain.catalog.loader import get_metastore
        from datachain.query.session import Session

        catalog = Session.get().catalog
        if not catalog.in_memory:
            catalog.metastore.update_job(job_id, metrics=metrics)
            return
        # The job lives in the configured metastore, not in the in-memory one.
        with get_metastore() as metastore:
            metastore.update_job(job_id, metrics=metrics)


def get(key: str) -> str | int | float | bool | None:
    """Get a metric value."""
    return metrics[key]
