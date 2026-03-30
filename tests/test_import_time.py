import logging
import subprocess
import sys
from pathlib import Path

import pytest

import datachain as dc

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
MAX_IMPORT_TIME_MS = 800

lazy_modules = [
    "adlfs",
    "boto3",
    "botocore",
    "gcsfs",
    "google",
    "numpy",
    "pyarrow",
    "requests",
    "s3fs",
    "torch",
]


def _import_time_chain(test_session):
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "import datachain"],
        stderr=subprocess.PIPE,
        check=True,
    )
    out = proc.stderr.replace(b"import time:", b"").strip()
    Path("import_time.csv").write_bytes(out)

    try:
        chain = dc.read_csv("import_time.csv", session=test_session, delimiter="|")
    except Exception:
        logger.error("Failed to parse output: %r", proc.stderr)
        raise

    chain = chain.save("import_time")
    # TODO: use `mutate` instead of `map`
    return (
        chain.map(cumulative_ms=lambda cumulative: cumulative // 1000, output=int)
        .map(self_ms=lambda self_us: self_us // 1000, output=int)
        .map(**{"import": lambda imported_package: imported_package.lstrip()})
        .select("self_ms", "cumulative_ms", "import")
        .order_by("cumulative_ms", descending=True)
    )


# disable coverage for this test to minimize import time overhead
@pytest.mark.no_cover
@pytest.mark.skipif(sys.platform == "win32", reason="not reliable on Windows")
def test_import_time(catalog, test_session):
    """
    Outside of the test, you can measure the import time with:
        python -Ximporttime -c 'import datachain'

    To visualize the import profile, consider using `tuna`: https://github.com/nschloe/tuna.
    """
    import_timings = []
    for attempt in range(MAX_ATTEMPTS):
        chain = _import_time_chain(test_session)
        (import_time_ms,) = chain.filter(
            dc.C("import") == "datachain",
        ).to_values("cumulative_ms")
        import_timings.append((chain, import_time_ms))
        # pass `--log-cli-level=info` to see these logs live
        logger.info("attempt %d, import time: %dms", attempt + 1, import_time_ms)

    chain, min_import_time = min(import_timings, key=lambda x: x[1])
    # If there is a regression, uncomment the following to find the culprit:
    # dc.show(limit=40)
    for module in lazy_modules:
        assert not chain.filter(dc.C("import").startswith(module)).to_list(), (
            f"found {module} at import time"
        )
    assert min_import_time < MAX_IMPORT_TIME_MS, (
        f"Possible import time regression; took {min_import_time}ms"
    )
