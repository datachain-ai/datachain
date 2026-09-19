---
name: datachain-core
description: Use ONLY for abstract DataChain SDK questions — API usage, method signatures, or code patterns — when no specific dataset or bucket is referenced. If the request mentions creating, saving, listing, exploring datasets or buckets, use datachain-knowledge instead.
---

Read `{skill_dir}/SDK.md` in full before answering DataChain SDK questions or generating DataChain Python code. It holds the SDK rules: API usage, UDF signatures, settings, delta semantics, materialization patterns, saving, exporting. The last section below holds the steps that need a local checkout and the `dc-knowledge/` knowledge base.

## Scope of this skill

**This skill does not own methodology.** Decisions about *which* datasets to build, what scope, what shape (Container / Asset / Sense / Task), what fields to save, and when to dialogue with the user about layer choices — those are the CAST methodology, which lives in the **datachain-knowledge** skill at `{knowledge_skill_dir}/CAST.md`.

When knowledge is loaded, it is the orchestrator: it plans the layers (CAST §4), invokes the rules in `SDK.md` to write the code, then runs the KB pipeline. When knowledge is *not* loaded (raw SDK use, no `dc-knowledge/` directory), `SDK.md` is self-sufficient — CAST doctrine simply does not apply.

If you find yourself reasoning about "should I build a Sense layer here?" or "should this be scoped to the bucket or the directory?" from inside this skill, stop — those questions belong upstream. Ask the user to load the knowledge skill, or fall through to a direct solve.

## Before writing any pipeline code

1. If `dc-knowledge/index.md` exists, read it **first**.
2. When the user's task overlaps with an existing dataset, read its `.md` under `dc-knowledge/datasets/` for schema, code patterns, and lineage.
3. **Bucket access: anonymous or authenticated?** Check `dc-knowledge/buckets/` for a `.md` file with `anon: true/false` in frontmatter. If none, run `datachain bucket status <uri>` to detect. If `denied` or `not found`, stop and ask the user.

Never create or modify files under `dc-knowledge/` — that directory is owned by the `knowledge` skill.
