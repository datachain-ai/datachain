# auth list

List the saved Studio authentication tokens.

## Synopsis

```usage
usage: datachain auth list [-h] [-v] [-q]
```

## Description

This command lists your saved Studio tokens and shows the default team. For each token, it shows:

* `Teams` - The teams the token is for, saved with `--team`, or `*` for a token saved without `--team`: the fallback when no token is saved for a team.
* `Name` - The token name, as shown in your Studio profile.
* `Expires` - The date the token expires.

If `DATACHAIN_STUDIO_TOKEN` is set, the output says so: that token is used for every team instead of the saved tokens. It is never sent to a Studio URL that only a project's config sets; the output says that too. See [`datachain auth login`](login.md) for how DataChain picks a token.

Token values are not shown. Run [`datachain auth token`](token.md) to print one.

With a self-hosted Studio, it also shows the Studio URL in use and, for each token, the Studio instance that issued it.

## Options

* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

1. List saved tokens:
```bash
datachain auth list
```

```
Default team: acme

Teams       Name        Expires
----------  ----------  ----------
*           laptop      2027-11-16
customer-x  customer-x  2027-02-14
```
