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

<p align="center">
<em>🔨 Wrangle unstructured AI data at scale</em>
</p>


DataChain is a Python-based AI-data warehouse for transforming and
analyzing unstructured data like images, audio, videos, text and PDFs.
It integrates with external storage (e.g. S3, GCP, Azure, HuggingFace) to process data
efficiently without data duplication and manages metadata in an internal
database for easy and efficient querying.

## Use Cases

1.  **ETL.** Pythonic framework for describing and running unstructured
    data transformations and enrichments, applying models to data,
    including LLMs.
2.  **Analytics.** DataChain dataset is a table that combines all the
    information about data objects in one place + it provides
    dataframe-like API and vectorized engine to do analytics on these
    tables at scale.
3.  **Versioning.** DataChain doesn't store, require moving or copying
    data (unlike DVC). Perfect use case is a bucket with thousands or
    millions of images, videos, audio, PDFs.

## Key Features

📂 **Multimodal Dataset Versioning.**

:   -   Version unstructured data without moving or creating data
        copies, by supporting references to S3, GCP, Azure, and local
        file systems.
    -   Multimodal data support: images, video, text, PDFs, JSONs, CSVs,
        parquet, etc.
    -   Unite files and metadata together into persistent, versioned,
        columnar datasets.

🐍 **Python-friendly.**

:   -   Operate on Python objects and object fields: float scores,
        strings, matrixes, LLM response objects.
    -   Run Python code in a high-scale, terabytes size datasets, with
        built-in parallelization and memory-efficient computing --- no
        SQL or Spark required.

🧠 **Data Enrichment and Processing.**

:   -   Generate metadata using local AI models and LLM APIs.
    -   Filter, join, and group datasets by metadata. Search by vector
        embeddings.
    -   High-performance vectorized operations on Python objects: sum,
        count, avg, etc.
    -   Pass datasets to Pytorch and Tensorflow, or export them back
        into storage.


## Documentation Guide

The following pages provide detailed documentation on DataChain's features, architecture, and usage patterns. You'll learn how to effectively use DataChain for managing and processing unstructured data at scale.

- [🏃🏼‍♂️ Quick Start](quick-start.md): Get up and running with DataChain in no time.
- [🎯 Examples](examples.md): Explore practical examples and use cases.
- [📚 Tutorials](tutorials.md): Learn how to use DataChain for specific tasks.
- [📚 User Guide](guide/index.md): Deeper dive into DataChain technical aspects and supported workflows.
- [🐍 API Reference](references/index.md): Dive into the technical details and API reference.
- [🤝 Contributing](contributing.md): Learn how to contribute to DataChain.


<!-- Open source and Studio -->

## Open Source and Studio

DataChain is available as an open source project and Studio as a proprietary solution for teams.

- [DataChain Studio](https://studio.datachain.ai/):
    - **Centralized dataset registry** to manage data, code and dependencies in one place.
    - **Data Lineage** for data sources as well as derivative dataset.
    - **UI for Multimodal Data** like images, videos, and PDFs.
    - **Scalable Compute** to handle large datasets (100M+ files) and in-house AI model inference.
    - **Access control** including SSO and team based collaboration.
- [DataChain Open Source](https://github.com/datachain-ai/datachain):
    - Python-based AI-data warehouse for transforming and analyzing unstructured data like images, audio, videos, text and PDFs.
