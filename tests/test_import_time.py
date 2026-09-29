import logging
import subprocess
import sys

import pytest

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 5
MAX_IMPORT_CPU_MS = 1

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


def _import_datachain():
    import resource

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "import datachain"],
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_ms = (
        (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    ) * 1000

    imports = []
    for line in proc.stderr.splitlines():
        if not line.startswith("import time:"):
            continue
        _, cumulative_us, name = (part.strip() for part in line[12:].split("|"))
        if cumulative_us.isdigit():
            imports.append((int(cumulative_us) // 1000, name))
    return cpu_ms, imports


# disable coverage for this test to minimize import time overhead
@pytest.mark.no_cover
@pytest.mark.skipif(sys.platform == "win32", reason="not reliable on Windows")
def test_import_time():
    """
    Outside of the test, you can measure the import time with:
        python -Ximporttime -c 'import datachain'

    To visualize the import profile, consider using `tuna`: https://github.com/nschloe/tuna.
    """
    attempts = []
    for attempt in range(MAX_ATTEMPTS):
        cpu_ms, imports = _import_datachain()
        attempts.append((cpu_ms, imports))
        # pass `--log-cli-level=info` to see these logs live
        logger.info("attempt %d, import CPU time: %dms", attempt + 1, cpu_ms)

    min_cpu_ms, imports = min(attempts, key=lambda x: x[0])
    for module in lazy_modules:
        assert not [name for _, name in imports if name.startswith(module)], (
            f"found {module} at import time"
        )

    culprits = "\n".join(
        f"  {ms}ms {name}" for ms, name in sorted(imports, reverse=True)[:10]
    )
    all_cpu_ms = ", ".join(f"{cpu_ms:.0f}" for cpu_ms, _ in attempts)
    assert min_cpu_ms < MAX_IMPORT_CPU_MS, (
        f"Possible import time regression; took {min_cpu_ms:.0f}ms CPU "
        f"(attempts: {all_cpu_ms}). Slowest imports (cumulative):\n{culprits}"
    )
