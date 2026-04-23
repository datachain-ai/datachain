---
title: Guides
---

# Guides

In-depth coverage of DataChain capabilities, from reading data through scaling to production.

## Data I/O

- [Reading Data](reading-data.md) -- storage files, structured formats, databases, in-memory sources, metadata merging
- [Exporting Data](exporting-data.md) -- pandas, Parquet, CSV, JSON, PyTorch, storage, databases

## Processing

- [Data Engine Operations](operations.md) -- filter, merge, group_by, mutate, and other SQL-speed operations
- [Python Operations (UDFs)](udfs.md) -- map, gen, agg, setup, and class-based lifecycle
- [Function Library](functions.md) -- dc.func.* for distance, aggregate, window, path, string, and conditional functions

## Datasets

- [Working with Datasets](datasets.md) -- creating, versioning, namespaces, comparing, management, metrics

## Advanced

- [Scaling and Performance](scaling.md) -- parallel, distributed, checkpoints, delta updates, caching
- [Multi-Stage Pipelines](multi-stage-pipelines.md) -- stage boundaries, comparative evaluation, cost tracking
- [Vector Search](vector-search.md) -- embedding computation, similarity search, drift detection
- [Best Practices](best-practices.md) -- rules for writing correct, idiomatic DataChain code

## Environment and Configuration

- [Data Processing Overview](processing.md) -- overview of processing features
- [Delta Processing](delta.md) -- incremental processing details
- [Error Handling and Retries](retry.md) -- handling processing errors
- [Checkpoints](checkpoints.md) -- automatic resume from failures
- [Environment Variables](env.md) -- configuration options
- [Namespaces](namespaces.md) -- namespace and project details
- [Local DB Migrations](db_migrations.md) -- handling upgrades
