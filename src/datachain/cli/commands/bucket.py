import sys

from datachain.client import bucket_status


def bucket_status_cmd(uri: str, client_config: dict | None = None) -> int:
    """Check existence and access of a bucket/container.

    Returns 0 if bucket exists, 1 if not found.
    Raises on network errors.
    """
    status = bucket_status(uri, **(client_config or {}))

    if status.exists:
        print("Status: exists")
        print(f"Access: {status.access}")
    else:
        print("Status: not found")
    if status.error:
        print(f"Error: {status.error}", file=sys.stderr)
    return 0 if status.exists else 1
