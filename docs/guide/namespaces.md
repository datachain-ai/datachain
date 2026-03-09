# Organizing Datasets with Namespace and Project

DataChain allows you to organize datasets using namespaces and projects. These provide an additional structure for managing data across different workflows, use cases, or organizational structures.

A dataset in DataChain is organized as:

```
<namespace>.<project>.<dataset>
```

For example:

```
dev.analytics.metrics
```

## Default Namespace and Project

If no namespace or project is specified, DataChain uses defaults depending on whether you're using **Studio** or the **CLI**.

### Studio

- **Namespace:** `@<username>` (e.g. `@jondoe`)
- **Project:** `default`
- Saving without namespace/project:

```python
dc.read_values(scores=[1.2, 3.4, 2.5]).save("metrics")
# Saved as @jondoe.default.metrics
```

### CLI

- **Namespace:** `local`
- **Project:** `local`
- Saving without namespace/project:

```python
dc.read_values(scores=[2.0, 2.2, 2.8]).save("metrics")
# Saved as local.local.metrics
```

In the CLI, you cannot create or use any namespaces or projects other than the default `local.local`.

## Creating a Project (Studio only)

In Studio, you can explicitly create a project and namespace using:

```python
import datachain as dc

dc.create_project("dev", "analytics")
```

This creates the `dev` namespace (if it doesn't exist) and a project called `analytics` inside it.

**Note:** Creating custom namespaces and projects is only supported in **Studio**. In the **CLI**, only the default `local` namespace and `local` project are available.

## Saving a Dataset Using a Fully Qualified Name

You can implicitly create and use namespaces and projects by saving a dataset using a fully qualified name:

```python
dc.read_values(scores=[1.2, 3.4, 2.5]).save("dev.analytics.metrics")
```

In Studio, this automatically creates the namespace and project if they don’t already exist.

In CLI, only `local.local.<dataset>` is supported. Using any other namespace or project will result in an error.

## Using `.settings()` to Set Namespace and Project

You can also set the namespace and project using `.settings()`:

```python
dc.read_values(scores=[1.2, 3.4, 2.5])
  .settings(namespace="dev", project="analytics")
  .save("metrics")
```

This is equivalent to saving to `dev.analytics.metrics`.

In CLI, `.settings()` is only supported when both `namespace` and `project` are set to `"local"`.

## Setting Namespace and Project via Environment Variables

In addition to using `.settings()`, you can configure the namespace and project using environment variables:

- `DATACHAIN_NAMESPACE` sets the namespace.
- `DATACHAIN_PROJECT` sets the project name, or both the namespace and project using the format `namespace.project`.

### Examples

```
# Set namespace only
export DATACHAIN_NAMESPACE=dev

# Set project only
export DATACHAIN_PROJECT=analytics

# Set both namespace and project
export DATACHAIN_PROJECT=dev.analytics
```

##  How Namespace and Project Are Resolved

When determining which namespace and project to use, Datachain applies the following precedence:

1. **Fully qualified dataset name**
   If the dataset name includes both the namespace and project, these values take highest precedence.
   ```python
   dc.read_dataset("dev.analytics.metrics")

2. **Explicit settings in code**
   Values provided via `.settings()` or passed directly to `read_dataset()` or similar methods.
   ```python
   dc.settings(namespace="dev", project="analytics")
   dc.read_dataset("metrics", namespace="dev", project="analytics")
   ```
3. **Environment variables**
   Namespace and project set using environment variables:
   ```console
   export DATACHAIN_PROJECT=dev.analytics
   ```
4. **Defaults**
If none of the above are provided, Datachain falls back to the default namespace and project.

## Reading a Dataset from a Project

To read a dataset from a specific namespace and project:

```python
ds = dc.read_dataset("dev.analytics.metrics")
```

In CLI, this only works for datasets saved in the default `local.local` project.


## Example (Studio)

```python
import datachain as dc

dc.create_project("prod", "analytics")

dc.read_csv("gs://bucket/metrics.csv") \
  .save("prod.analytics.metrics")

ds = dc.read_dataset("prod.analytics.metrics")
ds.show()
```

## Example (CLI – default only)

```python
import datachain as dc

dc.read_values(scores=[0.8, 1.5, 2.1]).save("metrics")

ds = dc.read_dataset("local.local.metrics")
ds.show()
```

## Removing Namespaces and Projects

Use `delete_namespace` to remove an empty namespace or an empty project within a namespace. Delete will fail if the target is not empty.

### Signature

```python
def delete_namespace(name: str, session: Session | None) -> None:
```

- **`<namespace>`** — deletes the namespace (must contain no projects or datasets).
- **`<namespace>.<project>`** — deletes the project (must contain no datasets).

### Examples

```python
import datachain as dc

dc.delete_namespace("dev.my-project")  # delete project
dc.delete_namespace("dev")             # delete namespace
```
