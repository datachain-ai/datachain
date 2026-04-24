---
title: Guides
---

# Guides

In-depth coverage of DataChain capabilities. Start with **Get Data In** and **Transform** for the core workflow, then explore deeper topics as needed.

## Get Data In

- [Reading Data](reading-data.md) -- storage files, structured formats, SQL databases, in-memory sources, metadata merging
- [Remote Storage](remotes.md) -- S3, GCS, Azure configuration, credentials, and access patterns

## Transform

- [Data Engine Operations](operations.md) -- filter, merge, group_by, mutate, and other SQL-speed operations
- [Python Operations](python-engine.md) -- map, gen, agg, setup, and class-based lifecycle
- [Function Library](functions.md) -- dc.func.* for distance, aggregate, window, path, string, and conditional functions
- [Vector Search](vector-search.md) -- embedding computation, similarity search, drift detection

## Get Data Out

- [Exporting Data](exporting-data.md) -- pandas, Parquet, CSV, JSON, PyTorch DataLoader, train/test split, storage, SQL databases

## Datasets

- [Working with Datasets](datasets.md) -- creating, versioning, namespaces, comparing, management, metrics

## Agents

- [Knowledge Base](knowledge-base.md) -- skill installation, `dc-knowledge/` generation, agent workflow, browsing

## Scale and Recover

- [Scaling and Performance](scaling.md) -- parallel, distributed, async prefetch, caching
- [Delta Processing](delta.md) -- incremental processing of new and changed files
- [Checkpoints](checkpoints.md) -- automatic resume from failures
- [Multi-Stage Pipelines](multi-stage-pipelines.md) -- stage boundaries, comparative evaluation, cost tracking

## Reference

- [Best Practices](best-practices.md) -- rules for writing correct, idiomatic DataChain code
- [Error Handling and Retries](retry.md) -- handling processing errors
- [Data Processing Overview](processing.md) -- overview of processing features
- [Environment Variables](env.md) -- configuration options
- [Namespaces](namespaces.md) -- namespace and project details
- [Local DB Migrations](db_migrations.md) -- handling upgrades
