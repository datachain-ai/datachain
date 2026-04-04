"""Print max mtime of .datachain/db* files as ISO-8601 UTC."""

import os
from datetime import datetime, timezone
from glob import glob

from utils import studio_available


def cmd_db_mtime():
    matches = glob(".datachain/db*")
    if not matches:
        if studio_available():
            # No local DB but Studio is configured — signal Studio mode.
            # The skill will skip the timestamp comparison and always refresh.
            print("studio")
        else:
            # No DB, no Studio — return epoch so the graph is always stale
            print("1970-01-01T00:00:00Z")
        return
    mtime = max(os.path.getmtime(p) for p in matches)
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
    print(dt.strftime("%Y-%m-%dT%H:%M:%SZ"))


if __name__ == "__main__":
    cmd_db_mtime()
