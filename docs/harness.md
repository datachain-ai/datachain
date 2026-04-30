---
title: Data Harness
---

# DataChain as a data harness

Claude Code, Cursor, and Codex are harnesses for code: they give the LLM repo context, dedicated tools, and persistent memory across sessions. That is what makes agents good at code instead of guessing at it.

DataChain is the same shape for data. The agent's repo is replaced by your storage and databases; the harness's notes file is replaced by the Knowledge Base; the operational memory of git history is replaced by Data Memory. Both halves of the harness, code-side and data-side, feed the same agent.

<p align="center">
  <img src="../assets/harness.svg" alt="DataChain mirrors the code harness, for data" width="700" />
</p>

## Mapping

| Code harness | Data harness |
|---|---|
| Repo files | Files in object storage, tables in databases |
| `~/.claude/` notes, `CLAUDE.md` | Knowledge Base (`dc-knowledge/`) |
| git history, type signatures | Data Memory: typed, versioned datasets with lineage |
| Tools (Read, Edit, Bash) | DataChain operations (`read_storage`, `map`, `save`) |
| Test results in the working directory | Saved datasets in `.datachain/db` |
