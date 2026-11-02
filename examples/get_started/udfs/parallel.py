"""
This is a simple UDF to demonstrate local parallel processing with multiprocessing.

In add_signals specify either parallel=True to use processes equal to the number
of CPUs available to the process, or parallel=N for N processes.
The default if parallel is not specified is to run single-threaded.

The UDF specified in map will then be run in parallel across all these
worker processes, no other code changes are needed.
"""

import datachain as dc


# This is a simple single-threaded benchmark function to demonstrate the speedup
# that can be achieved with multiprocessing, by enabling parallel in add_signals.
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# Define the UDF:
def path_len_benchmark(path: str) -> int:
    # Run the fibonacci benchmark as an example of a single-threaded CPU-bound UDF
    fibonacci(35)
    if path.endswith(".json"):
        return -1
    return len(path)


# Run in chain
(
    dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True)
    # Try to disable to see the difference in performance
    .settings(parallel=True)
    .map(path_len=path_len_benchmark, params=["file.path"])
    .show()
)
