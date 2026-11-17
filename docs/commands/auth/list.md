# auth list

List the saved Studio authentication tokens.

## Synopsis

```usage
usage: datachain auth list [-h] [-v] [-q]
```

## Description

This command shows the Studio URL and the default team DataChain uses, and the Studio tokens saved for this project and for you. For each token, it shows:

* `Studio` - The Studio instance that issued the token.
* `Teams` - The teams the token is used for, or `all teams`.
* `Name` - The token name, as shown in your Studio profile.
* `Expires` - The date the token expires.
* `Saved in` - The file the token is saved in.

The token used for the default team is marked with `*`. See [`datachain auth login`](login.md) for how DataChain picks it.

Token values are not shown. Run [`datachain auth token`](token.md) to print one.

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
Studio URL: https://studio.datachain.ai
Default team: acme

    Studio                       Teams       Name        Expires     Saved in
--  ---------------------------  ----------  ----------  ----------  ---------------------------------
*   https://studio.datachain.ai  all teams   laptop      2027-11-16  ~/.config/datachain/credentials
    https://studio.datachain.ai  customer-x  customer-x  2027-02-14  ~/.config/datachain/credentials
```
