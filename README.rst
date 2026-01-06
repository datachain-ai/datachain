================
|logo| DataChain
================

|PyPI| |Python Version| |Codecov| |Tests| |DeepWiki|

.. |logo| image:: docs/assets/datachain.svg
   :height: 24
.. |PyPI| image:: https://img.shields.io/pypi/v/datachain.svg
   :target: https://pypi.org/project/datachain/
   :alt: PyPI
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/datachain
   :target: https://pypi.org/project/datachain
   :alt: Python Version
.. |Codecov| image:: https://codecov.io/gh/datachain-ai/datachain/graph/badge.svg?token=byliXGGyGB
   :target: https://codecov.io/gh/datachain-ai/datachain
   :alt: Codecov
.. |Tests| image:: https://github.com/datachain-ai/datachain/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/datachain-ai/datachain/actions/workflows/tests.yml
   :alt: Tests
.. |DeepWiki| image:: https://deepwiki.com/badge.svg
   :target: https://deepwiki.com/datachain-ai/datachain
   :alt: DeepWiki

DataChain is a Python-based AI-data warehouse for transforming and analyzing unstructured
data like images, audio, videos, text and PDFs. It integrates with external storage
(e.g. S3) to process data efficiently without data duplication and manages metadata
in an internal database for easy and efficient querying.


Use Cases
=========

1. **ETL.** Pythonic framework for describing and running unstructured data transformations
   and enrichments, applying models to data, including LLMs.
2. **Analytics.** DataChain dataset is a table that combines all the information about data
   objects in one place + it provides dataframe-like API and vectorized engine to do analytics
   on these tables at scale.
3. **Versioning.** DataChain doesn't store, require moving or copying data.
   Perfect use case is a bucket with thousands or millions of images, videos, audio, PDFs.
4. **Incremental Processing.** DataChain's delta and retry features allow for efficient
   processing workflows:

   - **Delta Processing**: Process only new or changed files/records
   - **Retry Processing**: Automatically reprocess records with errors or missing results
   - **Combined Approach**: Process new data and fix errors in a single pipeline

Getting Started
===============

Visit `Quick Start <https://docs.datachain.ai/quick-start>`_ and `Docs <https://docs.datachain.ai/>`_
to get started with `DataChain` and learn more.

.. code:: bash

        pip install datachain


Example: Download Subset of Files Based on Metadata
---------------------------------------------------

Sometimes users only need to download a specific subset of files from cloud storage,
rather than the entire dataset.
For example, you could use a JSON file's metadata to download just cat images with
high confidence scores.


.. code:: py

    import datachain as dc

    meta = dc.read_json("gs://datachain-demo/dogs-and-cats/*json", column="meta", anon=True)
    images = dc.read_storage("gs://datachain-demo/dogs-and-cats/*jpg", anon=True)

    images_id = images.map(id=lambda file: file.path.split('.')[-2])
    annotated = images_id.merge(meta, on="id", right_on="meta.id")

    likely_cats = annotated.filter((dc.Column("meta.inference.confidence") > 0.93) \
                                   & (dc.Column("meta.inference.class_") == "cat"))
    likely_cats.to_storage("high-confidence-cats/", signal="file")


Example: Incremental Processing with Error Handling
---------------------------------------------------

This example shows how to use both delta and retry processing for efficient handling of large
datasets that evolve over time and may occasionally have processing errors.

.. code:: py

    import datachain as dc

    def process_file(file: dc.File) -> tuple[str, str, str]:
        """Analyze a file, may occasionally fail."""
        try:
            # Your processing logic here
            content = file.read_text()
            result = content.upper()
            return content, result, ""  # No error
        except Exception as e:
            # Return an error that will trigger reprocessing next time
            return "", "", str(e)  # Error field will trigger retry

    # Process files efficiently with delta and retry
    # Run it many times, keep adding files, to see delta and retry in action
    chain = (
        dc.read_storage(
            "data/",
            update=True,
            delta=True,              # Process only new/changed files
            delta_on="file.path",    # Identify files by path
            delta_retry="error",     # Process files with error again
        )
        .map(process_file, output=("content", "result", "error"))
        .save("processed-data")
    )


Example: LLM based text-file evaluation
---------------------------------------

In this example, we evaluate chatbot conversations stored in text files
using LLM based evaluation.

.. code:: shell

    $ pip install mistralai # Requires version >=1.0.0
    $ export MISTRAL_API_KEY=_your_key_

Python code:

.. code:: py

    import os
    from mistralai import Mistral
    import datachain as dc

    PROMPT = "Was this dialog successful? Answer in a single word: Success or Failure."

    def eval_dialogue(file: dc.File) -> bool:
         client = Mistral(api_key = os.environ["MISTRAL_API_KEY"])
         response = client.chat.complete(
             model="open-mixtral-8x22b",
             messages=[{"role": "system", "content": PROMPT},
                       {"role": "user", "content": file.read()}])
         result = response.choices[0].message.content
         return result.lower().startswith("success")

    chain = (
       dc.read_storage("gs://datachain-demo/chatbot-KiT/", column="file", anon=True)
       .settings(parallel=4, cache=True)
       .map(is_success=eval_dialogue)
       .save("mistral_files")
    )

    successful_chain = chain.filter(dc.Column("is_success") == True)
    successful_chain.to_storage("./output_mistral")

    print(f"{successful_chain.count()} files were exported")



With the instruction above, the Mistral model considers 31/50 files to hold the successful dialogues:

.. code:: shell

    $ ls output_mistral/datachain-demo/chatbot-KiT/
    1.txt  15.txt 18.txt 2.txt  22.txt 25.txt 28.txt 33.txt 37.txt 4.txt  41.txt ...
    $ ls output_mistral/datachain-demo/chatbot-KiT/ | wc -l
    31


Key Features
============

📂 **Multimodal Dataset Versioning.**
   - Version unstructured data without moving or creating data copies, by supporting
     references to S3, GCP, Azure, and local file systems.
   - Multimodal data support: images, video, text, PDFs, JSONs, CSVs, parquet, etc.
   - Unite files and metadata together into persistent, versioned, columnar datasets.

🐍 **Python-friendly.**
   - Operate on Python objects and object fields: float scores, strings, matrixes,
     LLM response objects.
   - Run Python code in a high-scale, terabytes size datasets, with built-in
     parallelization and memory-efficient computing — no SQL or Spark required.

🧠 **Data Enrichment and Processing.**
   - Generate metadata using local AI models and LLM APIs.
   - Filter, join, and group datasets by metadata. Search by vector embeddings.
   - High-performance vectorized operations on Python objects: sum, count, avg, etc.
   - Pass datasets to Pytorch and Tensorflow, or export them back into storage.


Contributing
============

Contributions are very welcome. To learn more, see the `Contributor Guide`_.


Community and Support
=====================

* `Docs <https://docs.datachain.ai/>`_
* `File an issue`_ if you encounter any problems
* `Discord Chat <https://dvc.org/chat>`_
* `Email <mailto:support@dvc.org>`_
* `Twitter <https://twitter.com/DVCorg>`_


DataChain Studio Platform
=========================

`DataChain Studio`_ is a proprietary solution for teams that offers:

- **Centralized dataset registry** to manage data, code and
  dependencies in one place.
- **Data Lineage** for data sources as well as derivative dataset.
- **UI for Multimodal Data** like images, videos, and PDFs.
- **Scalable Compute** to handle large datasets (100M+ files) and in-house
  AI model inference.
- **Access control** including SSO and team based collaboration.

.. _PyPI: https://pypi.org/
.. _file an issue: https://github.com/datachain-ai/datachain/issues
.. github-only
.. _Contributor Guide: https://docs.datachain.ai/contributing
.. _Pydantic: https://github.com/pydantic/pydantic
.. _publicly available: https://radar.kit.edu/radar/en/dataset/FdJmclKpjHzLfExE.ExpBot%2B-%2BA%2Bdataset%2Bof%2B79%2Bdialogs%2Bwith%2Ban%2Bexperimental%2Bcustomer%2Bservice%2Bchatbot
.. _SQLite: https://www.sqlite.org/
.. _Getting Started: https://docs.datachain.ai/
.. _DataChain Studio: https://studio.datachain.ai/
