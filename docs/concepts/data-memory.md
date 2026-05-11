---
title: Data Memory
---

# Data Memory

Multi-session work over data is impossible without persistent typed memory. Data Memory is the layer that makes it possible: the accumulated record of everything a team has done with its data, deposited automatically as a consequence of running queries, queryable by the next operation as a settled premise. Without it, every session re-derives from raw bytes; with it, agents and people build new conclusions on top of prior conclusions.

## Why Memory Matters

Every query, exploration, and labeling session produces knowledge. Without persistence infrastructure, that knowledge evaporates when the script finishes. The next person, the next project, the next agent starts from raw data and a blank script. People become the memory, and people do not scale.

Memory changes this. Every query records its results, schemas, lineage, and context as a structural consequence of running, not a separate maintenance step. The hundredth query runs in an environment qualitatively richer than the tenth: more features extracted, more connections traced, more context for the next person or agent.

## Composed of Datasets

Memory is not a formless accumulation; it is a collection of named, versioned, typed [datasets](datasets.md), each one a discrete deposit. Each dataset is a unit of reasoning, not just storage: a materialized conclusion that the next query treats as a premise. The system does not remember raw events or intermediate state; it remembers datasets. This atomic structure is what makes memory queryable, shareable, and compoundable.

```python
import datachain as dc

# Each save() deposits into memory
(
    dc.read_storage("s3://bucket/images/", type="image")
    .map(emb=compute_embedding)
    .save("image_embeddings")
)

# The next query builds on it
ds = dc.read_dataset("image_embeddings")
```

## Formed Through Doing, Not Curation

Memory is not a catalog that someone maintains. It is what happens when every operation records its results as a side effect. Systems that depend on voluntary human entry collapse within months: representations drift, adoption falls, teams revert to social workarounds. Memory stays current because it *is* the operational reality.

## Queryable at Warehouse Speed

Filter, join, group_by, order_by, and similarity search are properties of Data Memory itself, not a separate query engine sitting next to it. They run on a columnar SQL backend (SQLite locally, ClickHouse in Studio) at sub-second latency over millions of per-file records, two orders of magnitude below the cost of re-running the producing pipeline.

```python
import datachain as dc

(
    dc.read_storage("gs://datachain-demo/")
    .filter(dc.C("file.size") > 0)
    .group_by(
        count=dc.func.count(),
        total=dc.func.sum(dc.C("file.size")),
        partition_by=dc.func.path.file_ext(dc.C("file.path")),
    )
    .order_by("total", descending=True)
    .show()
)
```

No Python runtime spins up. No rows are deserialized. The query runs at warehouse speed.

The boundary with the [Compute Engine](compute-engine.md) is total. Every operation is either heavy Python compute over raw files (LLM calls, model inference, multimodal extraction), or a Data Memory operation over typed records already saved. The user writes one chain; the system decides which side handles what.

## Compounding Requires Fast Recall

Compounding only works when recall is cheaper than recreation. If recall is slower than recreation, the team silently re-runs the work, and memory degrades into archive. Warehouse-speed recall makes building on prior work always the path of least resistance.

## Dataset Registry

Data Memory is the queryable system of record for every dataset and its versions. The registry indexes names, versions, provenance, and links every dataset to the records it holds. Agents reach the registry without filesystem traversal: a single query returns what already exists, with full lineage attached.

```python
import datachain as dc

# Browse all datasets
for info in dc.datasets().collect("dataset"):
    print(f"{info.name} v{info.version}")

# Inspect a specific dataset
ds = dc.read_dataset("image_embeddings")
ds.print_schema()
print(ds.name, ds.version)
```

## Provenance

Memory that cannot explain how it was produced is memory that gets rebuilt from scratch. Every `.save()` automatically records four things alongside its results:

1. **Dependencies**: parent datasets with versions, storage URIs for every input
2. **Source code**: the full script, stored verbatim
3. **Author**: the person or service account that ran the script
4. **Creation time**

None of this requires manual declaration. It is captured from code and execution as a consequence of the operation. Each deposit is verifiable without asking the person who created it.

DataChain also supports attaching metrics and parameters alongside provenance:

```python
import datachain as dc

results = (
    dc.read_dataset("training_data")
    .map(prediction=run_model)
    .save("predictions")
)

dc.metrics.set("accuracy", 0.95)
dc.metrics.set("f1_score", 0.91)

learning_rate = dc.param("learning_rate", 0.001)
```

Metrics are recorded in the [dataset registry](#dataset-registry) alongside provenance. Parameters are captured in lineage so the exact configuration that produced a dataset is always recoverable.

## Discoverable by Humans and Agents

Memory that exists but cannot be found does not compound. Three people build three versions of the same dataset because nobody can see what already exists. An agent hallucinates a column instead of finding the one already computed. The [knowledge base](knowledge-base.md) turns "the dataset exists" into "the next person found it and built on it."

## Operational, Not Declarative

Data Memory works by running, not by describing. Every chain creates new datasets; every save records full lineage; every read returns typed records the next chain builds on. Existing context-layer products (Snowflake Cortex, Databricks Metric Views, dbt MetricFlow, Cube, AtScale) come at the same goal from the declarative side. They ship YAML semantic models and governed views over warehouse tables that say what the data means and how to read it. The two modes work together. Declarative models give agents stable definitions and governed access. The operational side lets agents produce, recall, and combine new datasets through Python, ML, and multimodal compute. A full data context layer for agent work needs both, and Data Memory is what brings the operational side.

## Shared, Not Per-Session

Conversation memory tools (Letta, Mem0, Hindsight, supermemory) keep per-session agent state: chat history, user preferences, summaries of recent actions. That state belongs to the agent and lives only inside its conversations. Data Memory sits in a different layer. It belongs to the team. It exists before any conversation starts and persists after every session ends. The agent reads from it rather than rewriting it during a single turn. The two compose; one does not replace the other.

## Out of Scope: Institutional Knowledge

Data Memory is derived from data, code, and execution. The organizational context in Slack, Notion, and meeting notes requires different tooling and is deliberately out of scope.
