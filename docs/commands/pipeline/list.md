# pipeline list

List pipelines in Studio

## Synopsis

```usage
usage: datachain pipeline list [-h] [-v] [-q]
                            [-t TEAM] [-s STATUS]
                            [-l LIMIT] [-S SEARCH]
```

## Description

This command fetches the list of the pipelines in Studio filtered by status or search term if provided.

## Options

*  `-s STATUS, --status STATUS` - Status of the pipelines to list. Possible values are: `'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'PAUSED', 'CANCELED'`
*  `-l LIMIT, --limit LIMIT` - Limit the number of pipelines to list
*  `-S SEARCH, --search SEARCH` - Search for pipelines by name or the dataset created from.
* `-t TEAM, --team TEAM` - Team to list pipelines for (default: from config)
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

#### Command
1. List recent 20 pipelines:
```bash
datachain pipeline list
```

2. List recent 50 pipelines
```bash
datachain pipeline list --limit 50
```

3. List failed pipelines only
```bash
datachain pipeline list --status failed
```

4. Search for a pipeline by dataset name
```bash
datachain pipeline list --search final_result
```

5. Search for a pipeline by name
```bash
datachain pipeline list --search rathe-kyat
```


#### Sample output

```
+------------+-----------+------------------------------------------+------------+---------------------+
| Name       | Status    | Target                                   | Progress   | Created At          |
+============+===========+==========================================+============+=====================+
| rathe-kyat | COMPLETED | @amritghimire.default.final_result@1.0.9 | 16/16      | 2025-12-08T17:06:54 |
+------------+-----------+------------------------------------------+------------+---------------------+
| surgy-dook | COMPLETED | @amritghimire.default.final_result@1.0.9 | 16/16      | 2025-12-08T16:52:41 |
+------------+-----------+------------------------------------------+------------+---------------------+
| dowie-curs | COMPLETED | @amritghimire.default.final_result@1.0.9 | 16/16      | 2025-12-08T16:52:24 |
+------------+-----------+------------------------------------------+------------+---------------------+
| gaunt-torc | COMPLETED | @amritghimire.default.final_result@1.0.9 | 16/16      | 2025-12-08T16:01:39 |
+------------+-----------+------------------------------------------+------------+---------------------+
| hardy-toea | COMPLETED | @amritghimire.default.final_result@1.0.9 | 16/16      | 2025-12-08T10:26:31 |
+------------+-----------+------------------------------------------+------------+---------------------+
| tamer-jest | COMPLETED | @amritghimire.default.final_result@1.0.9 | 16/16      | 2025-12-08T08:11:33 |
+------------+-----------+------------------------------------------+------------+---------------------+
| white-loft | COMPLETED | @amritghimire.default.final_result@1.0.9 | 16/16      | 2025-12-04T13:25:13 |
+------------+-----------+------------------------------------------+------------+---------------------+
```

## Notes
* To see more details of failed pipelines, use `datachain pipeline status` with the name from the list.
