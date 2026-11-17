# auth logout

Revoke and remove saved Studio authentication tokens.

## Synopsis

```usage
usage: datachain auth logout [-h] [-v] [-q] [--local] [--team TEAM]
                             [-H HOSTNAME]
```

## Description

This command logs out of Studio: it revokes the saved tokens, so Studio no longer accepts them, and removes them from the credentials file. If Studio no longer accepts a token, for example because it was already revoked, the token is only removed from the file.

By default, it removes your tokens for the current Studio. Use `--local` to remove the project's tokens instead. With `--team`, only the token for that team is removed. If that token is used for more teams, it is removed for all of them.

## Options

* `--local` - Remove the project's tokens instead of your global tokens.
* `--team TEAM` - Remove only the token for this team.
* `-H HOSTNAME`, `--hostname HOSTNAME` - The Studio to log out from. Defaults to the current Studio.
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

1. Remove your tokens for the current Studio:
```bash
datachain auth logout
```

2. Remove the project's tokens:
```bash
datachain auth logout --local
```

3. Remove only the token used for one team:
```bash
datachain auth logout --team customer-team
```
