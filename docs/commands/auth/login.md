# auth login

Authenticate DataChain with Studio and save a client access token.

## Synopsis

```usage
usage: datachain auth login [-h] [-v] [-q] [-H HOSTNAME] [-s SCOPES] [-n NAME]
                            [--no-open] [--local] [--team TEAM]
                            [--expires-in EXPIRES_IN]
```

## Description

By default, this command authenticates DataChain with Studio using default scopes and assigns a random name as the token name. The authentication token will be used for subsequent Studio operations.

The token is saved in a `credentials` file next to the global DataChain config, or in the project's `.datachain/credentials` with `--local`. The file is created readable by you only (`0600`), and is kept out of the config file, so the config holds no secrets. A project's `.datachain/credentials` is also listed in `.datachain/.gitignore`.

You can keep several tokens. Each one is saved for the Studio it was issued by and is never sent to another one. For every command, DataChain picks the token by the Studio and team the command uses:

1. The `DATACHAIN_STUDIO_TOKEN` environment variable, if set.
2. The project's tokens, then your own: a token scoped to the team (`--team`) first, then a token for all teams.

So one login covers all your teams, and a token logged in with `--team` is used for those teams only, for example a token from another account or a token with fewer permissions.

If you log in with exactly one `--team` and no default team is set yet, that team becomes the default. If `-H` names a Studio other than the current one, it becomes the current Studio.

Tokens saved by older DataChain versions in the config file still work. `datachain auth login` moves such a token to the credentials file.

## Options

* `-H HOSTNAME`, `--hostname HOSTNAME` - The hostname of the Studio instance to authenticate with.
* `-s SCOPES`, `--scopes SCOPES` - Authentication token scopes. Allowed scopes: `EXPERIMENTS`, `DATASETS`, `MODELS`. Defaults to all available scopes.
* `-n NAME`, `--name NAME` - The name of the authentication token. It will be used to identify the token shown in Studio profile. Defaults to a random name.
* `--no-open` - Use code-based authentication without browser. You will be presented with a user code to enter in the browser. DataChain will also use this if it cannot launch the browser on your behalf.
* `--local` - Save the token in the project's `.datachain/credentials` instead of your global credentials. A project's tokens are used before your own.
* `--team TEAM` - Scope token to specific teams, and use it for these teams only. Can be specified multiple times to scope to multiple teams.
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

5. Save a token for this project only, scoped to the team it works with:
```bash
datachain auth login --local --team data-team
```

6. Authenticate with team scoping:
```bash
datachain auth login --team data-team
```

7. Authenticate with multiple teams:
```bash
datachain auth login --team data-team --team ml-team
```

8. Authenticate with custom expiration:
```bash
datachain auth login --expires-in 90
```

9. Authenticate with team scoping and custom expiration:
```bash
datachain auth login --team data-team --expires-in 30 --name short-term-token
```

10. Keep a second token for one team next to your main token:
```bash
datachain auth login
datachain auth login --team customer-team
datachain dataset ls --team customer-team  # uses the customer-team token
```
