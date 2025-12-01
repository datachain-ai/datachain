# Checkpoints

Checkpoints allow DataChain to automatically skip re-creating datasets that were successfully saved in previous script runs. When a script fails or is interrupted, you can re-run it and DataChain will resume from where it left off, reusing datasets that were already created.

Checkpoints are available for both local script runs and Studio executions.

## How Checkpoints Work

### Local Script Runs

When you run a Python script locally (e.g., `python my_script.py`), DataChain automatically:

1. **Creates a job** for the script execution, using the script's absolute path as the job name
2. **Tracks parent jobs** by finding the last job with the same script name
3. **Calculates hashes** for each dataset save operation based on the DataChain operations chain
4. **Creates checkpoints** after each successful `.save()` call, storing the hash
5. **Checks for existing checkpoints** on subsequent runs - if a matching checkpoint exists in the parent job, DataChain skips the save and reuses the existing dataset

This means that if your script creates multiple datasets and fails partway through, the next run will skip recreating the datasets that were already successfully saved.

### Studio Runs

When running jobs on Studio, the checkpoint workflow is managed through the UI:

1. **Job execution** is triggered using the Run button in the Studio interface
2. **Checkpoint control** is explicit - you choose between:
   - **Run from scratch**: Ignores any existing checkpoints and recreates all datasets
   - **Continue from last checkpoint**: Resumes from the last successful checkpoint, skipping already-completed stages
3. **Parent-child job linking** is handled automatically by the system - no need for script path matching or job name conventions
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

**Second run:** DataChain detects that `filtered_data` and `transformed_data` were already created in the parent job with matching hashes. It skips recreating them and proceeds directly to Stage 3, creating only `final_results`.

## When Checkpoints Are Used

Checkpoints are automatically used when:

- Running a Python script locally (e.g., `python my_script.py`)
- The script has been run before
- A dataset with the same name is being saved
- The chain hash matches a checkpoint from the parent job

Checkpoints are **not** used when:

- Running code interactively (Python REPL, Jupyter notebooks)
- Running code as a module (e.g., `python -m mymodule`)
- The `DATACHAIN_CHECKPOINTS_RESET` environment variable is set (see below)

## Resetting Checkpoints

To ignore existing checkpoints and run your script from scratch, set the `DATACHAIN_CHECKPOINTS_RESET` environment variable:

```bash
export DATACHAIN_CHECKPOINTS_RESET=1
python my_script.py
```

Or set it inline:

```bash
DATACHAIN_CHECKPOINTS_RESET=1 python my_script.py
```

This forces DataChain to recreate all datasets, regardless of existing checkpoints.

## How Job Names Are Determined

DataChain uses different strategies for naming jobs depending on how the code is executed:

### Script Execution (Checkpoints Enabled)

When running `python my_script.py`, DataChain uses the **absolute path** to the script as the job name:

```
/home/user/projects/my_script.py
```

This allows DataChain to link runs of the same script together as parent-child jobs, enabling checkpoint lookup.

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

This creates a chain of hashes that uniquely identifies each stage of data processing. On subsequent runs, DataChain matches these hashes against the parent job's checkpoints and skips recreating datasets where the hashes match.

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

## Limitations

- **Script-based:** Code must be run as a script (not interactively or as a module).
- **Hash-based matching:** Any change to the chain will create a different hash, preventing checkpoint reuse.
- **Same script path:** The script must be run from the same absolute path for parent job linking to work.

## Future Plans

### UDF-Level Checkpoints

Currently, checkpoints are created only when datasets are saved using `.save()`. This means that if a script fails during a long-running UDF operation (like `.map()`, `.gen()`, or `.agg()`), the entire UDF computation must be rerun on the next execution.

Future versions will support **UDF-level checkpoints**, creating checkpoints after each UDF step in the chain. This will provide much more granular recovery:

```python
# Future behavior with UDF-level checkpoints
result = (
    dc.read_csv("data.csv")
    .map(heavy_computation_1)  # Checkpoint created after this UDF
    .map(heavy_computation_2)  # Checkpoint created after this UDF
    .map(heavy_computation_3)  # Checkpoint created after this UDF
    .save("result")
)
```

If the script fails during `heavy_computation_3`, the next run will skip re-executing `heavy_computation_1` and `heavy_computation_2`, resuming only the work that wasn't completed.
