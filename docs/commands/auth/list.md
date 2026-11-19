# auth list

List the saved Studio authentication tokens.

## Synopsis

```usage
usage: datachain auth list [-h] [-v] [-q]
```

## Description

This command lists the Studio tokens saved for you and for this project, and shows the default team. For each token, it shows:

* `Teams` - The teams the token is for, saved with `--team`, or `*` for a token saved without `--team`: the fallback when no token is saved for a team.
* `Name` - The token name, as shown in your Studio profile.
* `Expires` - The date the token expires.
* `Saved in` - The file the token is saved in.

If `DATACHAIN_STUDIO_TOKEN` is set, the output says so: that token is used for every team, and the saved tokens are not. See [`datachain auth login`](login.md) for how DataChain picks a token.

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

Teams       Name        Expires     Saved in
----------  ----------  ----------  -------------------------------
*           laptop      2027-11-16  ~/.config/datachain/credentials
customer-x  customer-x  2027-02-14  ~/.config/datachain/credentials
```
