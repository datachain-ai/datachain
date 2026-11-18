"""Run a CPU-bound Python function in parallel.

Use .settings(parallel=True) for automatic process counts or parallel=N for N
processes. Without this setting, the UDF runs in the current process.
"""

import datachain as dc


def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def path_len_benchmark(path: str) -> int:
    fibonacci(35)
    if path.endswith(".json"):
        return -1
    return len(path)


(
    dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True)
    # Try to disable to see the difference in performance
    .settings(parallel=True)
    .map(path_len=path_len_benchmark, params=["file.path"])
    .show()
)
