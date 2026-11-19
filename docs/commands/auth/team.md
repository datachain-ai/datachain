# auth team

Set the default team for Studio operations.

## Synopsis

```usage
usage: datachain auth team [-h] [-v] [-q] [--local] [team_name]
```

## Description

This command sets or shows the default team for Studio operations. By default, the team setting is saved globally, but you can use the `--local` option to set it for the current project.
If team_name is not passed, the current team in use is shown to the user.

An empty name (`""`) turns the default team off: commands then need `--team` or `DATACHAIN_STUDIO_TEAM`, and `datachain auth login --team` does not set a default team. In a project, `--local ""` turns off your global default team for that project.

## Arguments

* `team_name` - Optional, name of the team to set as default, or `""` to turn the default team off.

## Options

* `--local` - Set team locally for the current project
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

1. Set default team for all projects:
```bash
datachain auth team my-team
```

2. Set default team locally for the current project:
```bash
datachain auth team --local my-team
```

3. Print the current default in use
```bash
datachain auth team
```

4. Turn off the default team, so every command names its team:
```bash
datachain auth team ""
```
