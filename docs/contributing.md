---
title: Contributing
---

# Contributor Guide

Thank you for your interest in improving this project. This project is
open-source under the [Apache 2.0
license](https://opensource.org/licenses/Apache-2.0) and welcomes
contributions in the form of bug reports, feature requests, and pull
requests.

Here is a list of important resources for contributors:

-   [Source Code](https://github.com/datachain-ai/datachain)
-   [Documentation](https://docs.dvc.ai/datachain)
-   [Issue Tracker](https://github.com/datachain-ai/datachain/issues)
-   [Code of Conduct](https://github.com/datachain-ai/datachain?tab=coc-ov-file)

## How to report a bug

Report bugs on the [Issue
Tracker](https://github.com/datachain-ai/datachain/issues).

When filing an issue, make sure to answer these questions:

-   Which operating system and Python version are you using?
-   Which version of this project are you using?
-   What did you do?
-   What did you expect to see?
-   What did you see instead?

The best way to get your bug fixed is to provide a test case, and/or
steps to reproduce the issue.

## How to request a feature

Request features on the [Issue
Tracker](https://github.com/datachain-ai/datachain/issues).

## How to set up your development environment

You need Python 3.8+ and the following tools:

-   [Nox](https://nox.thea.codes/)

Install the package with development requirements:

``` console
$ pip install nox
```

## How to test the project

Run the full test suite:

``` console
$ nox
```

List the available Nox sessions:

``` console
$ nox --list-sessions
```

You can also run a specific Nox session. For example, invoke the unit
test suite like this:

``` console
$ nox --session=tests
```

Unit tests are located in the `tests` directory, and are written using
the [pytest](https://pytest.readthedocs.io/) testing framework.

## Build documentation

If you've made any changes to the documentation (including changes to
function signatures, class definitions, or docstrings that will appear
in the API documentation), make sure it builds successfully.

``` console
$ nox -s docs
```

In order to run this locally with hot reload on changes:

``` console
$ mkdocs serve
```

## How to submit changes

Open a [pull request](https://github.com/datachain-ai/datachain/pulls) to
submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

-   The Nox test suite must pass without errors and warnings.
-   Include unit tests. This project maintains 100% code coverage.
-   If your changes add functionality, update the documentation
    accordingly.

Feel free to submit early, though---we can always iterate on this.

To run linting and code formatting checks, you can invoke a `lint` session in nox:

``` console
$ nox -s lint
```

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate
your approach.

## Deprecation Policy

When deprecating an existing API, parameter, command, or data format:

1. **Use the Unified Helper**:
   Emit warnings using `datachain.warnings.warn_deprecated(what, *, instead=None, removal=None, once=False, category=None)`:
   ```python
   from datachain.warnings import warn_deprecated

   warn_deprecated("DataChain.print_schema()", instead="print(chain.schema)")
   ```

2. **Warning Categories**:
   - **`FutureWarning` (default)**: Used for all user-facing deprecations (public APIs, dataset storage formats, CLI flags). `FutureWarning` is visible by default in Python so end users are notified in their scripts and notebooks.
   - **`DeprecationWarning`**: Pass explicitly (`category=DeprecationWarning`) for internal or developer-facing deprecations. `DeprecationWarning` is ignored by default outside the `__main__` module.

3. **Message Format**:
   All deprecation messages follow a standardized single-line format:
   - `"X is deprecated; use Y instead."`
   - `"X is deprecated and will be removed in Z; use Y instead."`
   - `"X is deprecated and will be removed in Z."`
   - `"X is deprecated."`

4. **Hot Paths**:
   For performance-sensitive paths executed in tight loops (such as row readers or deserialization routines), pass `once=True` to emit the warning only once per process.

5. **Removal Timeline**:
   Deprecated features should remain functional and emit warnings for at least two minor releases (or one major release) before actual removal.
