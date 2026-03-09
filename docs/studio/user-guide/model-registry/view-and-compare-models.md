# View and compare models

You can find all your models in the [models dashboard](#models-dashboard). Each
model has separate [model details pages](#model-details-page) for all its model
versions. Also, all models from a given Git repository are included as
[`model` columns in the experiment tables](#model-columns-in-the-projects-experiment-table)
of those projects that connect to this Git repository.

## Models dashboard:

The models in your model registry are organized in a central dashboard that
facilitates search and discovery.


You can sort the models in the dashboard by several criteria, including model
framework, repository, etc.

DataChain Studio consolidates the stages of all the models in the registry, and
provides a way to filter models by stages.

You can take a look at the [models dashboard] in Iterative's public (read only)
model registry.

## Model details page:

You can open the details of any model in the registry by clicking on the name of
the model in the models dashboard.


A model details page is divided into the following sections:

- Top section: This contains
  - the model name,
  - a link to the model's Git repository,
  - latest registered version of the model,
  - a button to
    [register a new version](register-version.md),
    and
  - information about how many projects in DataChain Studio have been created from the
    model's Git repository.
- Left section: The left section contains information that is specific to a
  particular registered version of the model. It has a version picker, which you
  can use to switch between different registered versions of the model. For the
  selected version, the left section shows
  - buttons for common actions such as opening the corresponding experiment,
    deregistering the model version, and
    [assigning a stage to the version](assign-stage.md),
  - all assigned stages,
  - version description and labels,
  - path to the model,
  - metrics, params and plots.
- Right section: The right section contains information that is applicable
  across all the versions of the model. In particular, it displays
  - the assigned stages for the different versions, and
  - the history of all version registration and stage assignment actions.

You can find an example of a [model detail page] in Iterative's public (read
only) model registry.

## Model columns in the project's experiment table:

The models will also appear as `model` columns in the experiment tables of those
projects that have been created from the Git repository to which the model
belongs.


## Comparing model versions:

To compare model versions, select relevant commits in the project's experiment
table and click `Compare` and/or `Plots` buttons:


This way you can compare both registered model versions and unregistered
experimental iterations and make a decision to register a new version out of the
latter.

[models dashboard]: https://studio.datachain.ai/team/Iterative/models
[model detail page]:
  https://studio.datachain.ai/team/Iterative/models/PTzV-9EJgmZ6TGspXtwKqw==/lightgbm-model/v2.0.1
