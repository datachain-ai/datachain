---
title: Welcome to DataChain
---
# <a class="main-header-link" href="/" ><img style="display: inline-block;" src="/assets/datachain.svg" alt="DataChain"> <span style="display: inline-block;"> DataChain</span></a>

<style>
.md-content .md-typeset h1 { font-weight: bold; display: flex; align-items: center; justify-content: center; gap: 5px; }
.md-content .md-typeset h1 .main-header-link { display: flex; align-items: center; justify-content: center; gap: 8px;
 }
</style>

<p align="center">
  <a href="https://pypi.org/project/datachain/" target="_blank">
    <img src="https://img.shields.io/pypi/v/datachain.svg" alt="PyPI">
  </a>
  <a href="https://pypi.org/project/datachain/" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/datachain" alt="Python Version">
  </a>
  <a href="https://codecov.io/gh/datachain-ai/datachain" target="_blank">
    <img src="https://codecov.io/gh/datachain-ai/datachain/graph/badge.svg?token=byliXGGyGB" alt="Codecov">
  </a>
  <a href="https://github.com/datachain-ai/datachain/actions/workflows/tests.yml" target="_blank">
    <img src="https://github.com/datachain-ai/datachain/actions/workflows/tests.yml/badge.svg" alt="Tests">
  </a>
</p>

DataChain is the data memory layer for AI. Every pipeline, every exploration, every labeling session produces knowledge -- and that knowledge evaporates when the script finishes. DataChain changes this: every `.save()` deposits a versioned, typed, lineage-tracked dataset that the next person or agent starts from.

## See it in action

Task: find dogs in S3 similar to a reference image, filtered by breed, mask availability, and image dimensions.

```bash
datachain cp --anon s3://dc-readme/fiona.jpg .
```

```prompt
Find dogs in s3://dc-readme/oxford-pets-micro/ similar to fiona.jpg:
  - Pull breed metadata and mask files from annotations/
  - Exclude images without mask
  - Exclude Cocker Spaniels
  - Only include images wider than 400px
```

```
┌──────┬───────────────────────────────────┬────────────────────────────┬──────────┐
│ Rank │               Image               │           Breed            │ Distance │
├──────┼───────────────────────────────────┼────────────────────────────┼──────────┤
│    1 │ shiba_inu_52.jpg                  │ shiba_inu                  │    0.244 │
├──────┼───────────────────────────────────┼────────────────────────────┼──────────┤
│    2 │ shiba_inu_53.jpg                  │ shiba_inu                  │    0.323 │
├──────┼───────────────────────────────────┼────────────────────────────┼──────────┤
│    3 │ great_pyrenees_17.jpg             │ great_pyrenees             │    0.325 │
└──────┴───────────────────────────────────┴────────────────────────────┴──────────┘
```

The agent decomposed this into embedding, metadata, and filtering steps -- each saved as a named dataset. Next time, it starts from what's already built.

```
dc-knowledge
├── buckets
│   └── s3
│       └── dc_readme.md
├── datasets
│   ├── oxford_micro_dog_breeds.md
│   ├── oxford_micro_dog_embeddings.md
│   └── similar_to_fiona.md
└── index.md
```

![Visualize data knowledge base](assets/readme_obsidian.gif)

## Or write pipelines directly

```python
import datachain as dc

(
    dc.read_storage("s3://bucket/images/", type="image")
    .settings(parallel=8, cache=True)
    .map(emb=compute_embedding)
    .save("image_embeddings")   # versioned, named, typed
)

# Later: anyone (or any agent) can build on it
ds = dc.read_dataset("image_embeddings")
```

## Next Steps

- **[Quick Start](quick-start.md)** -- install DataChain and run your first pipeline
- **[Core Concepts](getting-started/core-concepts.md)** -- understand Data Memory, Datasets, and the dual engine
- **[Guides](guide/index.md)** -- in-depth coverage of every capability
- **[Best Practices](guide/best-practices.md)** -- rules for writing correct DataChain code
