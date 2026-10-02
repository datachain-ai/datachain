---
name: datachain-core
description: Use ONLY for abstract DataChain SDK questions — API usage, method signatures, or code patterns — when no specific dataset or bucket is referenced. If the request mentions creating, saving, listing, exploring datasets or buckets, use datachain-knowledge instead.
---

Read `{skill_dir}/SDK.md` in full before answering DataChain SDK questions or generating DataChain Python code. It holds the SDK rules: API usage, UDF signatures, settings, delta semantics, materialization patterns, saving, exporting. The last section below holds the steps that need a local checkout and the `dc-knowledge/` knowledge base.

## Scope of this skill

`SDK.md` owns how DataChain code is written — API usage, UDF signatures, the shape of a saved dataset, saving and exporting. It is self-sufficient on its own.

The **datachain-knowledge** skill owns the knowledge base at `dc-knowledge/`: what datasets already exist, what each one holds, how long a run will take, and keeping that record current. When it is loaded, it drives the session and calls the rules here to write the code.

## Before writing any pipeline code

1. If `dc-knowledge/index.md` exists, read it **first**.
2. When the user's task overlaps with an existing dataset, read its `.md` under `dc-knowledge/datasets/` for schema, code patterns, and lineage.
3. **Bucket access: anonymous or authenticated?** Check `dc-knowledge/buckets/` for a `.md` file with `anon: true/false` in frontmatter. If none, run `datachain bucket status <uri>` to detect. If `denied` or `not found`, stop and ask the user.

Never create or modify files under `dc-knowledge/` — that directory is owned by the `knowledge` skill.
