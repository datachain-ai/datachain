---
title: Data Harness
---

# DataChain as a data harness

Claude Code, Cursor, and Codex are harnesses for code: they give the LLM repo context, dedicated tools, and persistent memory across sessions. The code side compounds because of this; the data side restarts every time. Today an agent calls a data tool and the session collapses: outputs exceed the context budget, schemas get guessed, embeddings recompute from scratch, and the next session starts from zero.

The data harness pairs with the code harness on every dimension. Its working unit is the typed dataset (what the file in a repo is for code), its persistent store is the Dataset DB, its tools are pipeline operations, and its surface for agents is a compiled knowledge layer they read before generating code. Both halves, code-side and data-side, feed the same agent in the same session.

<p align="center">
  <img src="../assets/harness.svg" alt="DataChain mirrors the code harness, for data" width="700" />
</p>

## Mapping

| Code harness | Data harness |
|---|---|
| Working unit: typed file in a repo | Working unit: typed dataset in the Dataset DB |
| Raw inputs: source files, libraries | Raw inputs: files in object storage, tables in databases |
| Persistent context: git history, `~/.claude/`, `CLAUDE.md` | Persistent context: Dataset DB + Knowledge Base (`dc-knowledge/`) |
| Tools: Read, Edit, Bash | Tools: DataChain operations (`read_storage`, `map`, `save`) |
| Output: new commit on the same repo | Output: new dataset version in `.datachain/db` |

A working data harness keeps recall economics populated: summaries first (~$0.01), datasets next (~$1), raw-data path only when nothing higher answers (~$100).

## Extension, not replacement

The data harness is not a new agent. Code harnesses already cover the code side; the data half gives the same harnesses their data side. Same session, same tool call, expanded context. Teams keep building their own skills and MCPs on top, and the extension moves data work off the slow end without changing the workflow they already adopted.
