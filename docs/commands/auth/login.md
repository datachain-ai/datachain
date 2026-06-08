# auth login

Authenticate DataChain with Studio to save a client access token to DataChain configuration.

## Synopsis

```usage
usage: datachain auth login [-h] [-v] [-q] [-H HOSTNAME] [-s SCOPES] [-n NAME]
                            [--no-open] [--local] [--team TEAM]
                            [--expires-in EXPIRES_IN]
```

## Description

By default, this command authenticates DataChain with Studio using default scopes and assigns a random name as the token name. The authentication token will be used for subsequent Studio operations.

## Options

* `-H HOSTNAME`, `--hostname HOSTNAME` - The hostname of the Studio instance to authenticate with.
* `-s SCOPES`, `--scopes SCOPES` - Authentication token scopes. Allowed scopes: `EXPERIMENTS`, `DATASETS`, `MODELS`. Defaults to all available scopes.
* `-n NAME`, `--name NAME` - The name of the authentication token. It will be used to identify the token shown in Studio profile. Defaults to a random name.
* `--no-open` - Use code-based authentication without browser. You will be presented with a user code to enter in the browser. DataChain will also use this if it cannot launch the browser on your behalf.
* `--local` - Save the token in the local project config instead of the global configuration.
* `--team TEAM` - Scope token to specific teams. Can be specified multiple times to scope to multiple teams.
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

5. Save token locally for the project:
```bash
datachain auth login --local
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
