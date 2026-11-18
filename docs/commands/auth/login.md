# auth login

Authenticate DataChain with Studio and save a client access token.

## Synopsis

```usage
usage: datachain auth login [-h] [-v] [-q] [-H HOSTNAME] [-s SCOPES] [-n NAME]
                            [--no-open] [--local] [--team TEAM]
                            [--expires-in EXPIRES_IN]
```

## Description

By default, this command authenticates DataChain with Studio using default scopes and assigns a random name as the token name. DataChain uses the token for later Studio commands.

The token is saved in a `credentials` file in your DataChain config directory, for example `~/.config/datachain/credentials` on Linux. With `--local`, it is saved in the project's `.datachain/credentials` instead, and `.datachain/.gitignore` is updated so the file is not committed to Git. Only you can read the file. If other users can read it, DataChain ignores it.

### Tokens for specific teams

A command runs for a team: the one given with `--team`, or the default team (see [`datachain auth team`](team.md)). DataChain uses the first token from this list that is saved for that team:

1. The `DATACHAIN_STUDIO_TOKEN` environment variable.
2. The project's token for this team (saved with `--local --team`).
3. The project's token saved without `--team` (with `--local`).
4. Your token for this team (saved with `--team`).
5. Your token saved without `--team`.

A token saved with `--team` is used only for those teams. A token saved without `--team` is the fallback: DataChain uses it when no token is saved for the command's team. Which teams Studio accepts it for depends on what you chose in Studio when you logged in. You can keep both, for example a token for a team from another account, or a token with fewer permissions for one team.

Run [`datachain auth list`](list.md) to see the saved tokens and which one is used. If that token has expired, the command fails and names the token to replace; DataChain does not switch to another token, which could belong to another account.

If you log in with exactly one `--team` and no default team is set yet, that team becomes the default team.

### Self-hosted Studio

With `-H`, DataChain logs in to that Studio instance and uses it for later commands. A token is only sent to the instance it was issued by.

## Options

* `-H HOSTNAME`, `--hostname HOSTNAME` - The hostname of a self-hosted Studio instance to authenticate with.
* `-s SCOPES`, `--scopes SCOPES` - Authentication token scopes. Allowed scopes: `EXPERIMENTS`, `DATASETS`, `MODELS`. Defaults to all available scopes.
* `-n NAME`, `--name NAME` - The name of the authentication token. It will be used to identify the token shown in Studio profile. Defaults to a random name.
* `--no-open` - Use code-based authentication without browser. You will be presented with a user code to enter in the browser. DataChain will also use this if it cannot launch the browser on your behalf.
* `--local` - Save the token in the project's `.datachain/credentials`. The project's tokens are used before yours.
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

5. Save a token for this project only, for the team it works with:
```bash
datachain auth login --local --team data-team
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
datachain dataset ls --team customer-team  # uses the customer-team token
```
