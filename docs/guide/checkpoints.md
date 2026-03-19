# Checkpoints

Checkpoints let DataChain skip work that was already done in a previous run. When you re-run a script, DataChain detects which datasets and UDF results were already created and reuses them instead of recomputing.

## Example

```python
import datachain as dc

def process(file) -> str:
    return file.path.upper()

dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True).map(
    result=process
).save("processed_files")
```

**First run:** Lists storage, runs the UDF on every row, saves the dataset.

**Second run:** DataChain detects that `processed_files` was already created with the same chain. It skips everything:

```
Checkpoint found for dataset 'processed_files', skipping creation
```

## Recovering from Failures

If your UDF fails mid-execution, DataChain saves the progress. Fix the bug and re-run — only the remaining rows are processed.

```python
def process(file) -> str:
    # Bug: crashes on certain files
    if "cat" in file.path:
        raise ValueError("can't handle cats")
    return file.path.upper()

dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True).map(
    result=process
).save("processed_files")
```

**First run:** Processes some files, then crashes on a cat image.

**Fix and re-run:**

```python
def process(file) -> str:
    return file.path.upper()  # Fixed

dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True).map(
    result=process
).save("processed_files")
```

**Second run:** Skips already-processed rows and continues with the fixed code. No progress is lost.

This works because DataChain tracks processed rows incrementally during `.map()` and `.gen()` execution. For `.agg()`, checkpoints are only created on successful completion — if an aggregation fails, it restarts from scratch.

## What Invalidates Checkpoints

Checkpoints are tied to the chain's operations. Any change produces a different hash and triggers recomputation:

- Changing filter conditions, UDF code, parameters, or output types
- Adding, removing, or reordering operations in the chain
- Changing the source data (e.g. reading from a new dataset version)

**Exception:** If a UDF failed mid-execution and you fix only the code (without changing the output type), DataChain continues from partial results instead of restarting. If you change the output type, partial results are discarded and the UDF reruns from scratch.

## When Checkpoints Are Used

Checkpoints work automatically when running Python scripts:

```bash
python my_script.py              # checkpoints enabled
datachain job run my_script.py   # checkpoints enabled (Studio)
```

In Studio UI, you can choose between **Run from scratch** (ignores checkpoints) and **Continue from last checkpoint** when triggering a job.

Checkpoints are **not** used in:

- Interactive sessions (Python REPL, Jupyter notebooks)
- Module execution (`python -m mymodule`)

To force a fresh run ignoring existing checkpoints:

```bash
DATACHAIN_IGNORE_CHECKPOINTS=1 python my_script.py
```

## Limitations

- **Script path matters:** DataChain links runs by the script's absolute path. Moving the script to a different path breaks checkpoint linking.
- **Threading/multiprocessing:** Checkpoints are automatically disabled when Python threading or multiprocessing is detected. DataChain's built-in `parallel` setting for UDFs is not affected.
- **Unhashable callables:** Built-in functions (`len`, `str`), C extensions, and `Mock` objects cannot be reliably hashed. Use regular `def` functions or lambdas for UDFs.
