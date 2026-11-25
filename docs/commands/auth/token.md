# auth token

Display the current authentication token for Studio.

## Synopsis

```usage
usage: datachain auth token [-h] [-v] [-q] [--team TEAM]
```

## Description

This command displays the token DataChain uses for the default team, or the team given with `--team`. See [`datachain auth login`](login.md) for how the token is picked.

## Options

* `--team TEAM` - Show the token used for this team instead of the default team.
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

1. Display the current token:
```bash
datachain auth token
```

2. Display the token used for another team:
```bash
datachain auth token --team customer-team
```
