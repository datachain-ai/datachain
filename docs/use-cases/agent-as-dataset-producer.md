---
title: Agent as a Dataset Producer
---

# Agent as a Dataset Producer

## Without the harness

An agent runs an LLM judge over 50,000 dialog files and prints the verdicts in the notebook. The notebook gets shared in Slack, the kernel restarts, the cells age out, and three weeks later a teammate opens a new notebook to do the same scoring with the same prompt because the prior verdicts are not findable. The work happened; the conclusion did not stick.

## With DataChain

The agent's output is a versioned typed dataset, not a notebook artifact. The Pydantic model declares the shape of the conclusion. The `.save("name")` call records it into the Dataset DB with full lineage: the source dataset, the prompt, the model, the script, and the time. The Knowledge Base compiles the dataset into a page the next agent reads as context before generating any code.

```python
import os
import datachain as dc
from pydantic import BaseModel
from mistralai import Mistral

class Verdict(BaseModel):
    success: bool
    rationale: str

PROMPT = "Was this dialog successful? Reply JSON: {success: bool, rationale: str}."

def judge(file: dc.File) -> Verdict:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    response = client.chat.complete(
        model="open-mixtral-8x22b",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": file.read()},
        ],
        response_format={"type": "json_object"},
    )
    return Verdict.model_validate_json(response.choices[0].message.content)

(
    dc.read_storage("gs://datachain-demo/chatbot-KiT/", type="text", anon=True)
    .settings(parallel=4, cache=True)
    .map(verdict=judge)
    .save("dialog_verdicts")
)
```

The next session, in any agent or any notebook on the same project, reads the verdicts as a queryable typed dataset:

```python
import datachain as dc

verdicts = dc.read_dataset("dialog_verdicts")
print(verdicts.filter(dc.C("verdict.success") == False).count())
verdicts.select("file.path", "verdict.rationale").show(5)
```

The agent that wrote `dialog_verdicts` did not need to coordinate with the agent that reads it. The dataset name is the contract; the Pydantic schema is the interface.

## What this enables

- **Typed contracts between sessions.** A teammate or a different agent reads `verdict.success` as a typed boolean column; no parsing, no schema guessing, no broken pipelines on a renamed key.
- **Provenance for free.** Every saved dataset records the script, the parent datasets, the author, and the timestamp. The exact model version and prompt that produced a verdict are recoverable a year later.
- **Discovery instead of duplication.** The Knowledge Base lists `dialog_verdicts` with its schema and a preview, so the next agent finds the prior work before regenerating it.

## See also

- [Datasets](../concepts/datasets.md): why each save is a unit of reasoning, not just storage
- [Dataset DB](../concepts/dataset-db.md): provenance and immutability
- [Files and types](../concepts/files-and-types.md): Pydantic models as the schema layer
