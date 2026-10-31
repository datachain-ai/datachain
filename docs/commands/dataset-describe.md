# dataset describe

Inspect recorded local dataset state without scanning its rows or accessing object storage:

```sh
datachain dataset ls --local
datachain dataset describe local.local.labels
datachain dataset describe labels@1.0.0 --json
```

The default version is the latest complete version. `--version VERSION` selects another
complete version. Output includes schema, row count, creation time, available versions,
ancestor dependencies and their saved row counts, direct dependents, recorded storage
sources and `query_script`. Dependency counts describe the exact recorded versions,
without scanning their tables.
The Python equivalent is `catalog.describe_dataset(name, version=None)`.

`external_state: "not_checked"` means source files and models have not been checked for
changes. A completed dataset is not a freshness guarantee. Storage accessed inside a UDF
is visible only if recorded explicitly or in the saved code. The saved script does not
bundle imported modules, environment variables or model weights. Interactive runs may
have no captured script; a script can be supplied to `save(query_script=...)` explicitly.
