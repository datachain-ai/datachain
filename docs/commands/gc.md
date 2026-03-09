# gc

Garbage collect temporary tables, failed dataset versions, and outdated checkpoints.

## Synopsis

```usage
usage: datachain gc [-h] [-v] [-q] [--checkpoint-ttl CHECKPOINT_TTL]
```

## Description

This command cleans up internal DataChain storage by removing:

- **Temporary tables** created during query execution that were not properly cleaned up (e.g., due to crashes or interrupted operations).
- **Failed dataset versions** that were left in an incomplete or failed state.
- **Outdated checkpoints** and their associated UDF tables that have exceeded the time-to-live (TTL) threshold. See [Checkpoints](../guide/checkpoints.md) for more details.

## Options

* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.
* `--checkpoint-ttl` - Time-to-live for checkpoints in seconds. Checkpoints older than this value are considered outdated and eligible for cleanup. Defaults to 4 hours (14400 seconds).

## Examples

1. Run garbage collection:
```bash
datachain gc
```

2. Run garbage collection with a custom checkpoint TTL of 1 hour:
```bash
datachain gc --checkpoint-ttl 3600
```

Example output:
```
Collecting temporary tables...
  Removed 3 temporary tables.
Collecting failed dataset versions...
  No failed dataset versions to clean up.
Collecting outdated checkpoints...
  Removed 5 outdated checkpoints.
```
