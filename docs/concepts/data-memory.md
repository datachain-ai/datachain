---
title: Data Memory
---

# Data Memory

Data Memory is the accumulated record of everything a team has done with its data, deposited automatically as a structural consequence of the work itself.

## Why Memory Matters

Every pipeline, exploration, and labeling session produces knowledge. Without persistence infrastructure, that knowledge evaporates when the script finishes. The next person, the next project, the next agent starts from raw data and a blank script. People become the memory -- and people do not scale.

Memory changes this. Every operation records its results, schemas, lineage, and context as a side effect of running. The hundredth pipeline runs in an environment qualitatively richer than the tenth -- more features extracted, more connections traced, more context for the next person or agent.

## Composed of Datasets

Memory is not a formless accumulation -- it is a collection of named, versioned, typed [datasets](datasets.md). Each one is a discrete deposit. The system does not remember raw events or intermediate state; it remembers datasets. This atomic structure is what makes memory queryable, shareable, and compoundable.

```python
import datachain as dc

# Each save() deposits into memory
(
    dc.read_storage("s3://bucket/images/", type="image")
    .map(emb=compute_embedding)
    .save("image_embeddings")
)

# The next pipeline builds on it
ds = dc.read_dataset("image_embeddings")
```

## Formed Through Doing, Not Curation

Memory is not a catalog that someone maintains. It is what happens when every operation records its results as a side effect. Systems that depend on voluntary human entry collapse within months -- representations drift, adoption falls, teams revert to social workarounds. Memory stays current because it *is* the operational reality.

## Compounding Requires Fast Recall

Compounding only works when recall is cheaper than recreation. If retrieval is slower than re-running the pipeline, the team silently re-runs it and memory degrades into archive. The [Memory Engine](execution-model.md) -- a columnar SQL backend -- makes building on prior work always the path of least resistance.

## Trustworthy by Construction

Memory that cannot explain how it was produced is memory that gets rebuilt from scratch. The [provenance store](provenance.md) automatically records dependencies, source code, author, and creation time alongside every `.save()`. Each deposit is verifiable without asking the person who created it.

## Discoverable by Humans and Agents

Memory that exists but cannot be found does not compound. Three people build three versions of the same dataset because nobody can see what already exists. An agent hallucinates a column instead of finding the one already computed. The [knowledge base](knowledge-base.md) turns "the dataset exists" into "the next person found it and built on it."
