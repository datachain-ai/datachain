# auth logout

Revoke and remove saved Studio authentication tokens.

## Synopsis

```usage
usage: datachain auth logout [-h] [-v] [-q] [--local] [--team TEAM | --all]
                             [-H HOSTNAME]
```

## Description

This command logs out of Studio: it revokes a saved token, so Studio no longer accepts it, and removes it from the credentials file. If Studio no longer accepts the token, for example because it was already revoked, the token is only removed from the file.

Each `datachain auth login` has a matching logout:

* `datachain auth logout` removes the token for all teams, saved by `datachain auth login`.
* `datachain auth logout --team TEAM` removes the token for that team, saved by `datachain auth login --team TEAM`. If that token is also used for other teams, it is removed for them too.
* `datachain auth logout --all` removes every token for the Studio.

By default, it removes your tokens for the current Studio. Use `--local` to remove the project's tokens instead, and `-H` for another Studio. When the last token for the Studio is removed from a file, the default team set at the same level is cleared too.

## Options

* `--local` - Remove the project's tokens instead of your global tokens.
* `--team TEAM` - Remove the token for this team instead of the token for all teams.
* `--all` - Remove every token for the Studio: the token for all teams and the token for each team.
* `-H HOSTNAME`, `--hostname HOSTNAME` - The Studio to log out from. Defaults to the current Studio.
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

1. Remove your token for all teams:
```bash
datachain auth logout
```

2. Remove the project's token for all teams:
```bash
datachain auth logout --local
```

3. Remove the token for one team:
```bash
datachain auth logout --team customer-team
```

4. Remove every token for the current Studio:
```bash
datachain auth logout --all
```
