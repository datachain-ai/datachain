# auth logout

Revoke and remove saved Studio authentication tokens.

## Synopsis

```usage
usage: datachain auth logout [-h] [-v] [-q] [--team TEAM | --all]
                             [-H HOSTNAME]
```

## Description

This command logs out of Studio: it revokes a saved token, so Studio no longer accepts it, and removes it from the credentials file. If Studio no longer accepts the token, for example because it was already revoked, the token is only removed from the file.

Each `datachain auth login` has a matching logout:

* `datachain auth logout` removes the token saved by `datachain auth login`, without `--team`.
* `datachain auth logout --team TEAM` removes the token saved by `datachain auth login --team TEAM`. If that token is also used for other teams, it is removed for them too.
* `datachain auth logout --all` removes every token for the Studio.

Your default team is cleared with the last token for the Studio set in your config.

With a self-hosted Studio, logout removes the tokens for the instance in use, and `-H` picks another one.

## Options

* `--team TEAM` - Remove the token for this team instead of the one saved without `--team`.
* `--all` - Remove every token for the Studio, saved with or without `--team`.
* `-H HOSTNAME`, `--hostname HOSTNAME` - The self-hosted Studio instance to log out from. Defaults to the one in use.
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

1. Remove your token:
```bash
datachain auth logout
```

2. Remove the token for one team:
```bash
datachain auth logout --team customer-team
```

3. Remove all your tokens:
```bash
datachain auth logout --all
```
