# Checkpoints

Checkpoints let DataChain skip work that was already done in a previous run and recover from failures. When you re-run a script, DataChain detects which datasets and UDF results were already created and reuses them instead of recomputing. If a UDF fails mid-execution, you can fix the bug and re-run — only the remaining rows are processed.

## Example

Save this as `process.py`:

```python
import datachain as dc


def process(file) -> str:
    # Bug: crashes on certain files
    if "cat" in file.path:
        raise ValueError("can't handle cats")
    return file.path.upper()


(
    dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True)
    .map(result=process)
    .save("processed_files")
)
```

**First run** (`python process.py`): Processes some files, then crashes on a cat image.

**Fix the bug** — edit `process.py`:

```python
import datachain as dc


def process(file) -> str:
    return file.path.upper()


(
    dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True)
    .map(result=process)
    .save("processed_files")
)
```

**Second run** (`python process.py`): Skips already-processed rows and continues with the fixed code. No progress is lost.

**Third run** (`python process.py`): DataChain detects that `processed_files` was already created with the same chain. It skips everything:

```
Checkpoint found for dataset 'processed_files', skipping creation
```

## What Invalidates Checkpoints

Checkpoints are tied to the chain's operations. Any change produces a different hash and triggers recomputation:

- Changing filter conditions, parameters, or output types
- Adding, removing, or reordering operations in the chain
- Changing the source data (e.g. reading from a new dataset version)
- Modifying UDF code after a successful completion

**Exception:** If a UDF failed mid-execution and you fix the code (without changing the output type), DataChain continues from partial results instead of restarting. If you change the output type, partial results are discarded and the UDF reruns from scratch.

## How to Use

Checkpoints work automatically when running Python scripts:

```bash
python my_script.py              # checkpoints enabled
datachain job run my_script.py   # checkpoints enabled (Studio)
```

In Studio UI, you can choose between **Run** (ignores checkpoints) and **Resume** (continues from last checkpoint) when triggering a job.

![Run and Resume buttons in Studio](images/run_resume_cta.png)

Checkpoints are **not** used in:

- Interactive sessions (Python REPL, Jupyter notebooks)
- Module execution (`python -m mymodule`)

To force a fresh run ignoring existing checkpoints:

```bash
DATACHAIN_IGNORE_CHECKPOINTS=1 python my_script.py
```

## More Examples

### Changing one chain doesn't affect others

Save this as `multi_chain.py`:

```python
import datachain as dc

# Chain 1 — filter files by size
(
    dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True)
    .filter(dc.C("file.size") > 15000)
    .save("large_files")
)

# Chain 2 — get all file paths
(
    dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True)
    .map(path=lambda file: file.path, output=str)
    .save("all_paths")
)
```

Run it twice — both chains are reused on the second run. Now change the filter in chain 1 (e.g. `> 20000`) and run again. Chain 1 is recomputed, but chain 2 is reused — its chain is untouched.

### Generator recovery

Save this as `gen_tags.py`:

```python
import datachain as dc
from collections.abc import Iterator


def extract_parts(file) -> Iterator[str]:
    # yields each part of the file path separately
    for part in file.path.split("/"):
        if part:
            yield part


(
    dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True)
    .gen(part=extract_parts)
    .save("path_parts")
)
```

Generators (`.gen()`) that yield multiple rows per input also support partial recovery. If `extract_parts` crashes mid-execution, the next run continues from where it stopped — already-processed inputs are skipped, and incomplete inputs are re-processed.

### What recomputes and what doesn't

**Recomputes** — changing a filter:

```python
# Run 1
dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True).filter(
    dc.C("file.size") > 10000
).save("filtered")

# Run 2 — changed threshold
dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True).filter(
    dc.C("file.size") > 20000
).save("filtered")
# → "filtered" recomputes
```

**Recomputes** — changing UDF code after successful completion:

```python
# Run 1 — succeeds
def score(file) -> float:
    return file.size

# Run 2 — changed logic
def score(file) -> float:
    return file.size / 1024
# → UDF recomputes from scratch
```

**Does NOT recompute** — fixing a bug after a crash:

```python
# Run 1 — crashes mid-UDF
def score(file) -> float:
    if "cat" in file.path:
        raise ValueError("bug")
    return file.size

# Run 2 — bug fixed, same output type
def score(file) -> float:
    return file.size
# → UDF continues from where it stopped, already-processed rows are skipped
```

**Does NOT recompute** — changing an unrelated chain:

```python
# Chain 1 — changed
dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True).filter(
    dc.C("file.size") > 20000  # was 10000
).save("large_files")

# Chain 2 — untouched
dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True).map(
    path=lambda file: file.path, output=str
).save("all_paths")
# → "all_paths" is reused, only "large_files" recomputes
```

## Limitations

- **Script path matters:** DataChain links runs by the script's absolute path. Moving the script breaks checkpoint linking.
- **Threading/multiprocessing:** Checkpoints are automatically disabled when Python threading or multiprocessing is detected. DataChain's built-in `parallel` setting for UDFs is not affected.
- **Unhashable callables:** Built-in functions (`len`, `str`), C extensions, and `Mock` objects produce a different hash on each run, so checkpoints using these as UDFs will always recompute. Use regular `def` functions or lambdas instead.
