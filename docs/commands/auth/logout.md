# auth logout

Revoke and remove saved Studio authentication tokens.

## Synopsis

```usage
usage: datachain auth logout [-h] [-v] [-q] [--local] [--team TEAM]
                             [-H HOSTNAME]
```

## Description

This command revokes the tokens saved for the current Studio on Studio and removes them. By default, it removes your global tokens; use `--local` to remove the project's tokens instead. With `--team`, only the token scoped to that team is removed. If that token is scoped to more teams, you are logged out of all of them.

## Options

* `--local` - Remove the project's tokens instead of your global tokens.
* `--team TEAM` - Remove only the token scoped to this team.
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
