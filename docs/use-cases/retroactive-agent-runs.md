---
title: Retroactive Agent Runs
---

# Retroactive Agent Runs

## Without the harness

A new model ships on Friday. You want to re-score yesterday's 500,000 documents with it and compare against the old verdicts. The script you have re-reads the files from the bucket, re-decodes them, re-runs the embedding step the new judge needs, and re-scores everything. The base work you did yesterday is paid for again. With LLM API calls at fractions of a cent per call, a single re-run can cost thousands of dollars.

## With DataChain

The expensive base work is already a saved dataset, and saved datasets are immutable. The retroactive run does not change `dialog_verdicts`; it produces a new dataset derived from it, carrying every existing column plus one new column for the new judge. The file references live in the parent dataset, so no bytes are re-downloaded; only the new judge call pays per row. Checkpoints make even the new column resumable mid-run.

```python
import datachain as dc
from pydantic import BaseModel
from anthropic import Anthropic
import os

# yesterday's `dialog_verdicts` dataset is already saved with file references
# and the prior verdict column

class ClaudeVerdict(BaseModel):
    success: bool
    rationale: str

def judge_with_claude(file: dc.File) -> ClaudeVerdict:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=200,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "Was this dialog successful? Reply JSON: {success: bool, rationale: str}."},
            {"type": "text", "text": file.read()},
        ]}],
    )
    return ClaudeVerdict.model_validate_json(response.content[0].text)

(
    dc.read_dataset("dialog_verdicts")
    .settings(parallel=4, cache=True)
    .map(claude_verdict=judge_with_claude)
    .save("dialog_verdicts_claude")
)
```

`dialog_verdicts` is unchanged. `dialog_verdicts_claude` is a new derived dataset that carries the file references, the original `verdict` column, and the new `claude_verdict` column side by side. Lineage records that it depends on `dialog_verdicts`.

Both verdict columns live on the new dataset, so comparing them is one filter, not a join:

```python
import datachain as dc

(
    dc.read_dataset("dialog_verdicts_claude")
    .filter(dc.C("verdict.success") != dc.C("claude_verdict.success"))
    .select("file.path", "verdict.success", "claude_verdict.success", "claude_verdict.rationale")
    .show(20)
)
```

For an incremental run over only files added since the last save, `delta=True` skips processed rows automatically:

```python
import datachain as dc

(
    dc.read_storage("s3://acme-robots/runs/", anon=True, type="video", delta=True)
    .map(detections=detect_obstacles)
    .save("obstacle_detections")
)
```

## What this enables

- **The model upgrade gets cheap.** A new judge, a new embedding model, or a new prompt produces a new dataset derived from the prior one; only the new column pays per-row LLM cost. File reads and prior columns come for free from the parent.
- **Prior runs stay intact.** The original dataset is immutable. The retroactive run never overwrites it, so older experiments remain reproducible against the exact rows they were run on.
- **Comparisons are first-class.** Old and new columns sit side by side on the derived dataset; the diff is a filter, not a join across two ad-hoc tables.

## See also

- [Checkpoints](../guide/checkpoints.md): mid-run resume
- [Delta processing](../guide/delta.md): incremental updates over new files
- [Dataset DB](../concepts/dataset-db.md): why prior work is recoverable instead of recomputed
