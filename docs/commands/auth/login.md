# auth login

Authenticate DataChain with Studio and save a client access token.

## Synopsis

```usage
usage: datachain auth login [-h] [-v] [-q] [-H HOSTNAME] [-s SCOPES] [-n NAME]
                            [--no-open] [--team TEAM]
                            [--expires-in EXPIRES_IN]
```

## Description

By default, this command authenticates DataChain with Studio using default scopes and assigns a random name as the token name. DataChain uses the token for later Studio commands.

The token is saved in a `credentials` file in your DataChain config directory, for example `~/.config/datachain/credentials` on Linux. Only you can read the file. On macOS and Linux, if other users can read it, DataChain ignores it; on Windows, it relies on your user folder's permissions. Tokens are never saved in a project, so they can't be committed with it.

### Tokens for specific teams

A command runs for a team: the one given with `--team`, or the default team (see [`datachain auth team`](team.md)). DataChain uses the first of these tokens:

1. The `DATACHAIN_STUDIO_TOKEN` environment variable. It is only sent to the Studio set by `DATACHAIN_STUDIO_URL`, your config or the system config, or the default Studio, never to one that only a project's config sets.
2. Your token for this team (saved with `--team`).
3. Your token saved without `--team`.

A token saved with `--team` is used only for those teams. A token saved without `--team` is the fallback: DataChain uses it when no token is saved for the command's team. Which teams Studio accepts it for depends on what you chose in Studio when you logged in. You can keep both, for example a token for a team from another account, or a token with fewer permissions for one team.

To work with a team in one project, make it the project's default team with `datachain auth team TEAM --local`. Commands run there then use your token for that team.

Run [`datachain auth list`](list.md) to see the saved tokens. If the token picked has expired, the command fails and names the token to replace; DataChain does not switch to another token, which could belong to another account.

If you log in with exactly one `--team` and no default team is set, that team becomes your default team. A project's team doesn't count, and a login to a Studio set only by a project's config or by `DATACHAIN_STUDIO_URL` doesn't set one. To keep the default team off, run [`datachain auth team ""`](team.md).

### Self-hosted Studio

With `-H`, DataChain logs in to that Studio instance and uses it for later commands. A token is only sent to the instance it was issued by.

## Options

* `-H HOSTNAME`, `--hostname HOSTNAME` - The hostname of a self-hosted Studio instance to authenticate with.
* `-s SCOPES`, `--scopes SCOPES` - Authentication token scopes. Allowed scopes: `EXPERIMENTS`, `DATASETS`, `MODELS`. Defaults to all available scopes.
* `-n NAME`, `--name NAME` - The name of the authentication token. It will be used to identify the token shown in Studio profile. Defaults to a random name.
* `--no-open` - Use code-based authentication without browser. You will be presented with a user code to enter in the browser. DataChain will also use this if it cannot launch the browser on your behalf.
* `--team TEAM` - Create a token for this team, used only for it. Can be specified multiple times for several teams.
* `--expires-in EXPIRES_IN` - Token expiration in days. Defaults to 365 days.
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

1. Basic authentication with default settings:
```bash
datachain auth login
```

2. Authenticate with specific scopes:
```bash
datachain auth login --scopes EXPERIMENTS,DATASETS
```

3. Authenticate with a custom token name:
```bash
datachain auth login --name my-token
```

4. Authenticate using code-based flow:
```bash
datachain auth login --no-open
```

5. Use a team's token in this project:
```bash
datachain auth login --team data-team
datachain auth team data-team --local
```

6. Log in with a token for one team:
```bash
datachain auth login --team data-team
```

7. Log in with a token for several teams:
```bash
datachain auth login --team data-team --team ml-team
```

8. Authenticate with custom expiration:
```bash
datachain auth login --expires-in 90
```

9. Keep a second token for one team next to your main token:
```bash
datachain auth login
datachain auth login --team customer-team
datachain dataset ls --studio --team customer-team  # uses the customer-team token
```
