---
title: "Quick Start: Agents"
---

# Quick Start: Agents

## Installation

=== "pip"

    ```bash
    pip install datachain
    ```

=== "uv"

    ```bash
    uv add datachain
    ```

## Install the Skill

```bash
datachain skill install --target claude     # also: --target cursor, --target codex
```

The skill gives agents data awareness: what datasets exist, their schemas, which fields can be joined, and the meaning of columns inferred from the code that produced them.

## Your First Agent Task

Copy a reference image:

```bash
datachain cp --anon s3://dc-readme/fiona.jpg .
```

Enter prompt in Claude Code, Cursor, or Codex:

```prompt
Find dogs in s3://dc-readme/oxford-pets-micro/ similar to fiona.jpg:
  - Pull breed metadata and mask files from annotations/
  - Exclude images without mask
  - Exclude Cocker Spaniels
  - Only include images wider than 400px
```

Result:

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

The agent decomposed this into embedding, metadata, and filtering steps -- each saved as a named **dataset**. Next time, it starts from what's already built.

## Knowledge Base

The datasets are registered in a knowledge base in `dc-knowledge/`, optimized for both agents and humans:

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

Browse as markdown files, or open in Obsidian:

![Visualize data knowledge base](../assets/readme_obsidian.gif)

Run the skill prompt again to update the knowledge base after creating new datasets. See the [Knowledge Base guide](../guide/knowledge-base.md) for details.

## Next Steps

- [Core Concepts](core-concepts.md) -- understand Data Memory, Datasets, and the dual engine
- [Knowledge Base guide](../guide/knowledge-base.md) -- skill installation, generation, browsing
- [Guides](../guide/index.md) -- in-depth coverage of every capability
