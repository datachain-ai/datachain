# auth list

List the saved Studio authentication tokens.

## Synopsis

```usage
usage: datachain auth list [-h] [-v] [-q]
```

## Description

This command shows the current Studio and default team, and lists the tokens saved for this project and for you: the Studio each token is for, the teams it is scoped to, its name, when it expires and the file it is saved in. The token used for the current Studio and team is marked with `*`. Token values are not printed; use [`datachain auth token`](token.md) for that.

## Options

* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

1. List saved tokens:
```bash
datachain auth list
```
