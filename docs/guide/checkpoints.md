# Checkpoints

Checkpoints allow DataChain to automatically skip re-creating datasets that were successfully saved in previous script runs. When a script fails or is interrupted, you can re-run it and DataChain will resume from where it left off, reusing datasets that were already created.

Checkpoints are available for both local script runs and Studio executions.

## How Checkpoints Work

### Local Script Runs

When you run a Python script locally (e.g., `python my_script.py`), DataChain automatically:

1. **Creates a job** for the script execution, using the script's absolute path as the job name
2. **Tracks previous runs** by finding the last job with the same script name
3. **Calculates hashes** for each dataset save operation based on the DataChain operations chain
4. **Creates checkpoints** after each successful `.save()` call, storing the hash
5. **Checks for existing checkpoints** on subsequent runs - if a matching checkpoint exists from the previous run, DataChain skips the save and reuses the existing dataset

This means that if your script creates multiple datasets and fails partway through, the next run will skip recreating the datasets that were already successfully saved.

### Studio Runs

#### Using `datachain job run` CLI

When you run `datachain job run my_script.py`, DataChain automatically:

1. **Links jobs** by finding previous runs of the same script (by absolute path) that were also executed in Studio
2. **Passes checkpoint context** to Studio, enabling checkpoint reuse across runs

This means running the same script multiple times via `datachain job run` will automatically benefit from checkpoints without any additional configuration.

#### Using Studio UI

When triggering jobs through the Studio interface:

1. **Job execution** is triggered using the Run button in the Studio interface
2. **Checkpoint control** is explicit - you choose between:
   - **Run from scratch**: Ignores any existing checkpoints and recreates all datasets
   - **Continue from last checkpoint**: Resumes from the last successful checkpoint, skipping already-completed stages
3. **Job linking between runs** is handled automatically by the system
4. **Checkpoint behavior** during execution is the same as local runs: datasets are saved at each `.save()` call and can be reused on retry


## Example

Consider this script that processes data in multiple stages:

```python
import datachain as dc

# Stage 1: Load and filter data
filtered = (
    dc.read_csv("s3://mybucket/data.csv")
    .filter(dc.C("score") > 0.5)
    .save("filtered_data")
)

# Stage 2: Transform data
transformed = (
    filtered
    .map(value=lambda x: x * 2, output=float)
    .save("transformed_data")
)

# Stage 3: Aggregate results
result = (
    transformed
    .agg(
        total=lambda values: sum(values),
        partition_by="category",
    )
    .save("final_results")
)
```

**First run:** The script executes all three stages and creates three datasets: `filtered_data`, `transformed_data`, and `final_results`. If the script fails during Stage 3, only `filtered_data` and `transformed_data` are saved.

**Second run:** DataChain detects that `filtered_data` and `transformed_data` were already created in the previous run with matching hashes. It skips recreating them and proceeds directly to Stage 3, creating only `final_results`.

## When Checkpoints Are Used

Checkpoints are automatically used when:

- Running a Python script locally (e.g., `python my_script.py`)
- The script has been run before
- A dataset with the same name is being saved
- The chain hash matches a checkpoint from the previous run

Checkpoints are **not** used when:

- Running code interactively (Python REPL, Jupyter notebooks)
- Running code as a module (e.g., `python -m mymodule`)
- The `DATACHAIN_IGNORE_CHECKPOINTS` environment variable is set (see below)

## Resetting Checkpoints

To ignore existing checkpoints and run your script from scratch, set the `DATACHAIN_IGNORE_CHECKPOINTS` environment variable:

```bash
DATACHAIN_IGNORE_CHECKPOINTS=1 python my_script.py
```

This forces DataChain to recreate all datasets, regardless of existing checkpoints.

## How Job Names Are Determined

DataChain uses different strategies for naming jobs depending on how the code is executed:

### Script Execution (Checkpoints Enabled)

When running `python my_script.py`, DataChain uses the **absolute path** to the script as the job name:

```
/home/user/projects/my_script.py
```

This allows DataChain to link runs of the same script together, enabling checkpoint lookup across runs.

### Interactive or Module Execution (Checkpoints Disabled)

When running code interactively or as a module, DataChain uses a **unique UUID** as the job name:

```
a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

This prevents unrelated executions from being linked together, but also means checkpoints cannot be used.

## How Checkpoint Hashes Are Calculated

For each `.save()` operation, DataChain calculates a hash based on:

1. The hash of the previous checkpoint in the current job (if any)
2. The hash of the current DataChain operations chain

This creates a chain of hashes that uniquely identifies each stage of data processing. On subsequent runs, DataChain matches these hashes against checkpoints from the previous run and skips recreating datasets where the hashes match.

### Hash Invalidation

**Checkpoints are automatically invalidated when you modify the chain.** Any change to the DataChain operations will result in a different hash, causing DataChain to skip the checkpoint and recompute the dataset.

Changes that invalidate checkpoints include:

- **Modifying filter conditions:** `.filter(dc.C("score") > 0.5)` → `.filter(dc.C("score") > 0.8)`
- **Changing map/gen/agg functions:** Any modification to UDF logic
- **Altering function parameters:** Changes to column names, output types, or other parameters
- **Adding or removing operations:** Inserting new `.filter()`, `.map()`, or other steps
- **Reordering operations:** Changing the sequence of transformations

### Example

```python
# First run - creates three checkpoints
dc.read_csv("data.csv").save("stage1")  # Hash = H1

dc.read_dataset("stage1").filter(dc.C("x") > 5).save("stage2")  # Hash = H2 = hash(H1 + pipeline_hash)

dc.read_dataset("stage2").select("name", "value").save("stage3")  # Hash = H3 = hash(H2 + pipeline_hash)
```

**Second run (no changes):**
- All three hashes match → all three datasets are reused → no computation

**Second run (modified filter):**
```python
dc.read_csv("data.csv").save("stage1")  # Hash = H1 matches ✓ → reused

dc.read_dataset("stage1").filter(dc.C("x") > 10).save("stage2")  # Hash ≠ H2 ✗ → recomputed

dc.read_dataset("stage2").select("name", "value").save("stage3")  # Hash ≠ H3 ✗ → recomputed
```

Because the filter changed, `stage2` has a different hash and must be recomputed. Since `stage3` depends on `stage2`, its hash also changes (because it includes H2 in the calculation), so it must be recomputed as well.

**Key insight:** Modifying any step in the chain invalidates that checkpoint and all subsequent checkpoints, because the hash chain is broken.

## Dataset Persistence

Starting with the checkpoints feature, datasets created during script execution persist even if the script fails or is interrupted. This is essential for checkpoint functionality, as it allows subsequent runs to reuse successfully created datasets.

If you need to clean up datasets from failed runs, you can use:

```python
import datachain as dc

# Remove a specific dataset
dc.delete_dataset("dataset_name")

# List all datasets to see what's available
for ds in dc.datasets():
    print(ds.name)
```

## UDF-Level Checkpoints

In addition to dataset-level checkpointing via `.save()`, DataChain automatically creates checkpoints for individual UDFs (`.map()`, `.gen()`, `.agg()`) during execution.

**Two levels of checkpointing:**
- **Dataset checkpoints** (via `.save()`): When you explicitly save a dataset, it's persisted and can be used in other scripts. If you re-run the same chain with unchanged code, DataChain skips recreation and reuses the saved dataset.
- **UDF checkpoints** (automatic): Each UDF execution is automatically checkpointed. If a UDF completes successfully, it's skipped entirely on re-run (if code unchanged). If a UDF fails mid-execution, only the unprocessed rows are recomputed on re-run.

**Key differences:**
- `.save()` creates a named dataset that persists even if your script fails later, and can be used in other scripts
- UDF checkpoints are automatic and internal - they optimize execution within a single script by skipping or resuming UDFs

For `.map()` and `.gen()`, **DataChain saves processed rows continuously during UDF execution**. This means:
- If a UDF **completes successfully**, a checkpoint is created and the entire UDF is skipped on re-run (unless code changes)
- If a UDF **fails mid-execution**, the next run continues from where it left off, skipping already-processed rows - even if you've modified the UDF code to fix a bug

**Note:** For `.agg()`, checkpoints are created when the aggregation completes successfully, but partial results are not tracked. If an aggregation fails partway through, it will restart from scratch on the next run.

### How It Works

When executing `.map()` or `.gen()`, DataChain:

1. **Saves processed rows incrementally** as the UDF processes your dataset
2. **Creates a checkpoint** when the UDF completes successfully
3. **Allows you to fix bugs and continue** - if the UDF fails, you can modify the code and re-run, skipping already-processed rows
4. **Invalidates the checkpoint if you change the UDF after successful completion** - completed UDFs are recomputed from scratch if the code changes

For `.agg()`, checkpoints are only created upon successful completion, without incremental progress tracking.

### Example: Fixing a Bug Mid-Execution

```python

def process_image(file: File) -> int:
    # Bug: this will fail on some images
    img = Image.open(file.get_local_path())
    return img.size[0]

(
    dc.read_dataset("images")
    .map(width=process_image)
    .save("image_dimensions")
)
```

**First run:** Script processes 50% of images successfully, then fails on a corrupted image.

**After fixing the bug:**

```python
from datachain import File

def process_image(file: File) -> int:
    # Fixed: handle corrupted images gracefully
    try:
        img = Image.open(file.get_local_path())
        return img.size[0]
    except Exception:
        return 0
```

**Second run:** DataChain automatically skips the 50% of images that were already processed successfully, and continues processing the remaining images using the fixed code. You don't lose any progress from the first run.

### When UDF Checkpoints Are Invalidated

DataChain distinguishes between two types of UDF changes:

#### 1. Code-Only Changes (Bug Fixes) - Continues from Partial Results

When you fix a bug in your UDF code **without changing the output type**, DataChain allows you to continue from where the UDF failed. This is the key benefit of UDF-level checkpoints - you don't lose progress when fixing bugs.

**Example: Bug fix without output change**
```python
# First run - fails partway through
def process(num: int) -> int:
    if num > 100:
        raise Exception("Bug!")  # Oops, a bug!
    return num * 10

# Second run - continues from where it failed
def process(num: int) -> int:
    return num * 10  # Bug fixed! ✓ Continues from partial results
```

In this case, DataChain will skip already-processed rows and continue processing the remaining rows with your fixed code.

#### 2. Output Schema Changes - Forces Re-run from Scratch

When you change the **output type** of your UDF, DataChain automatically detects this and reruns the entire UDF from scratch. This prevents schema mismatches that would cause errors or corrupt data.

**Example: Output change**
```python
# First run - fails partway through
def process(num: int) -> int:
    if num > 100:
        raise Exception("Bug!")
    return num * 10

# Second run - output type changed
def process(num: int) -> str:
    return f"value_{num * 10}"  # Output type changed! ✗ Reruns from scratch
```

In this case, DataChain detects that the output type changed from `int` to `str` and discards partial results to avoid schema incompatibility. All rows will be reprocessed with the new output.

#### Changes That Invalidate In-Progress UDF Checkpoints

Partial results are automatically discarded when you change:

- **Output type** - Changes to the `output` parameter or return type annotations
- **Operations before the UDF** - Any changes to the data processing chain before the UDF

#### Changes That Invalidate Completed UDF Checkpoints

Once a UDF completes successfully, its checkpoint is tied to the UDF function code. If you modify the function and re-run the script, DataChain will detect the change and recompute the entire UDF from scratch.

Changes that invalidate completed UDF checkpoints:

- **Modifying the UDF function logic** - Any code changes inside the function
- **Changing function parameters or output types** - Changes to input/output specifications
- **Altering any operations before the UDF in the chain** - Changes to upstream data processing

**Key takeaway:** For in-progress (partial) UDFs, you can fix bugs freely as long as the output stays the same. For completed UDFs, any code change triggers a full recomputation.

## Limitations

When running locally:

- **Script-based:** Code must be run as a script (not interactively or as a module).
- **Same script path:** The script must be run from the same absolute path for linking to previous runs to work.
- **Threading/Multiprocessing:** Checkpoints are automatically disabled when Python threading or multiprocessing is detected to prevent race conditions. Any checkpoints created before threading starts remain valid for future runs. DataChain's built-in `parallel` setting for UDF execution is not affected by this limitation.

These limitations don't apply when running on Studio, where job linking between runs is handled automatically by the platform.

### UDF Hashing Limitations

DataChain computes checkpoint hashes by inspecting UDF code and metadata. Certain types of callables cannot be reliably hashed:

- **Built-in functions** (`len`, `str`, `int`, etc.): Cannot access bytecode, so a random hash is generated on each run. Checkpoints using these functions will not be reused.
- **C extensions**: Same limitation as built-ins - no accessible bytecode means a new hash each run.
- **Mock objects**: `Mock(side_effect=...)` cannot be reliably hashed because the side effect is not discoverable via inspection. Use regular functions instead.
- **Dynamically generated callables**: If a callable is created via `exec`/`eval` or its behavior depends on runtime state, the hash reflects only the method's code, not captured state.

To ensure checkpoints work correctly, use regular Python functions defined with `def` or lambda expressions for your UDFs.

## Future Plans

### Partial Result Tracking for Aggregations

Currently, `.agg()` creates checkpoints only upon successful completion, without tracking partial progress. Future versions will extend the same incremental progress tracking that `.map()` and `.gen()` have to aggregations, allowing them to resume from where they failed rather than restarting from scratch.
