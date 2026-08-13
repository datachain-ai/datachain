# Security & Permissions

Access inside a team comes from two things: each collaborator's **team role**,
and the **fine-grained permissions** that grant `read` or `write` on individual
namespaces, projects, and datasets.

## Roles

Every collaborator has one of three team roles. The role controls team-level
capabilities, but, except for admins, it does **not** by itself grant access to
any namespace, project, or dataset. That access comes from
[fine-grained permissions](#fine-grained-permissions).

- **`Admins`** - Have full access to everything in the team. Admins can see and
  modify every namespace, project, and dataset regardless of permission rules,
  invite and remove collaborators, manage groups and rules, manage team settings,
  configure cloud credentials, and manage billing.
- **`Editors`** - Can create resources, run jobs, and work with queries and
  experiment projects. On datasets, they can edit or delete a dataset and its
  files where they have a `write` [grant](#fine-grained-permissions) on it (or on
  the namespace/project that contains it). They cannot modify team settings,
  manage collaborators, or manage permissions.
- **`Viewers`** - Have read-only access. They can explore jobs and experiments,
  and any dataset they have a `read` [grant](#fine-grained-permissions) on. They
  cannot create or modify resources, and cannot change team settings.

Fine-grained grants only gate datasets and the namespaces and projects that
contain them (see [How access is resolved](#how-access-is-resolved)), an Editor
or Viewer with no matching grant sees no datasets. Jobs, queries, experiments,
and storage are governed by the team role itself, not by grants.

DataChain Studio does not have the concept of an `Owner` role. The user who
creates the team has the `Admin` role. The privileges of such an admin is the
same as that of any other collaborator who has been assigned the `Admin` role.

!!! note

    If your Git account does not have write access on the Git repository connected
    to a project, you cannot push changes (e.g., new experiments) to the repository
    even if the project belongs to a team where you are an `Editor` or `Admin`.

### How access is resolved

For namespaces, projects, and datasets, DataChain Studio resolves access like
this:

- **Admins bypass everything.** A team admin can read and write every resource,
  no matter what rules exist.
- **Access is grant-only.** For non-admins, access to a resource comes only from
  an explicit permission rule (targeting you or a group you belong to) or from
  ownership. If nothing grants you access, you have none, there is no fallback to
  your team role.
- **Grants are additive; the highest one wins.** If several rules apply to you
  (directly or through groups), you get the highest permission among them. A
  `read` grant never cancels a `write` grant.
- **Rules cascade downward.** A rule on a namespace also covers every project and
  dataset inside it; a rule on a project covers every dataset inside it.
- **Creating a namespace grants you `write` on it.** The creator of a namespace
  automatically gets `write` on that namespace (and, by cascade, on everything
  inside it). Creating a project or dataset inside an existing namespace does not
  add a new grant on its own.

!!! note

    Existing teams keep working after this feature rolls out. Each team starts
    with two ready-made groups, **All members (read)** and **All members
    (write)**, that grant every member access to all existing namespaces. Admins
    can edit or remove these groups to start scoping access down.

**Example.** Suppose a dataset lives at `prod.analytics.metrics`. A rule that
grants the `ml-team` group `read` on the `prod` namespace lets every member of
`ml-team` read `prod.analytics.metrics` (and everything else under `prod`). If
one of those members also has a personal `write` rule on
`prod.analytics.metrics`, they get `write` on it, the higher grant wins.

These rules are enforced everywhere datasets are accessed, the web dashboard,
the API, and running DataChain jobs (for example, `dc.read_dataset(...)` runs
under the submitting user's grants), not just in the UI.

Fine-grained grants govern **datasets** (and, by cascade, the namespaces and
projects that contain them); the datasets table below shows what a `read` versus
`write` [grant](#fine-grained-permissions) allows. **Jobs, queries, experiments,
and storage** are governed by the **team role** instead, their tables show what
each role can do. Admins can perform every action regardless of grant or role.

### Privileges for datasets

| Feature                     | Read | Write |
| --------------------------- | ---- | ----- |
| List datasets               | Yes  | Yes   |
| View dataset information    | Yes  | Yes   |
| View dataset rows           | Yes  | Yes   |
| View dataset versions       | Yes  | Yes   |
| Export datasets             | Yes  | Yes   |
| Preview files               | Yes  | Yes   |
| Create datasets             | No   | Yes   |
| Edit dataset metadata       | No   | Yes   |
| Delete datasets             | No   | Yes   |
| Upload files                | No   | Yes   |
| Move files in storage       | No   | Yes   |
| Delete files                | No   | Yes   |
| Reindex storage             | No   | Yes   |
| Create dataset from storage | No   | Yes   |

### Privileges for jobs

| Feature              | Viewer | Editor | Admin |
| -------------------- | ------ | ------ | ----- |
| List jobs            | Yes    | Yes    | Yes   |
| View job details     | Yes    | Yes    | Yes   |
| View job logs        | Yes    | Yes    | Yes   |
| List clusters        | Yes    | Yes    | Yes   |
| Create jobs          | No     | Yes    | Yes   |
| Cancel running jobs  | No     | Yes    | Yes   |
| Update job status    | No     | Yes    | Yes   |

### Privileges for queries

| Feature                 | Viewer | Editor | Admin |
| ----------------------- | ------ | ------ | ----- |
| List queries            | Yes    | Yes    | Yes   |
| View query details      | Yes    | Yes    | Yes   |
| Create queries          | No     | Yes    | Yes   |
| Update queries          | No     | Yes    | Yes   |
| Duplicate queries       | No     | Yes    | Yes   |
| Delete queries          | No     | Yes    | Yes   |

### Privileges for experiments

| Feature                                       | Viewer | Editor | Admin |
| --------------------------------------------- | ------ | ------ | ----- |
| Open a team's project                         | Yes    | Yes    | Yes   |
| View experiments and metrics                  | Yes    | Yes    | Yes   |
| Apply filters                                 | Yes    | Yes    | Yes   |
| Show / hide columns                           | Yes    | Yes    | Yes   |
| Save filters and column settings              | No     | Yes    | Yes   |
| Add a new project                             | No     | Yes    | Yes   |
| Edit project settings                         | No     | Yes    | Yes   |
| Delete a project                              | No     | Yes    | Yes   |
| Share a project                               | No     | Yes    | Yes   |

### Privileges for storage and activity logs

| Feature                  | Viewer | Editor | Admin |
| ------------------------ | ------ | ------ | ----- |
| List storage files       | Yes    | Yes    | Yes   |
| View activity logs       | Yes    | Yes    | Yes   |
| Create activity logs     | No     | Yes    | Yes   |
| Get presigned URLs       | No     | Yes    | Yes   |

### Privileges to manage the team

These team-management capabilities are governed by the team role itself (not by
permission rules), and are reserved for admins:

| Feature                            | Viewer | Editor | Admin |
| ---------------------------------- | ------ | ------ | ----- |
| Manage team settings               | No     | No     | Yes   |
| Manage team collaborators          | No     | No     | Yes   |
| Manage groups and permission rules | No     | No     | Yes   |
| Configure cloud credentials        | No     | No     | Yes   |
| Manage GitLab server connections   | No     | No     | Yes   |
| Configure Single Sign-on (SSO)     | No     | No     | Yes   |
| Manage team plan and billing       | No     | No     | Yes   |
| Delete a team                      | No     | No     | Yes   |

## Fine-grained permissions

Team roles are coarse, an Editor can write across everything they're granted, a
Viewer can read it. **Permissions** let admins be specific: sort collaborators
into **groups** and write **rules** that grant `read` or `write` on individual
namespaces, projects, and datasets.

You'll find Permissions under **Collaborators & Permissions** in the
[team settings](settings.md) page, below the Collaborators list. This area is
**admin-only**, other collaborators see "Only team admins can manage
permissions."

!!! note

    Non-admins only reach namespaces, projects, and datasets through permissions.
    Put people in groups, then add rules that grant read or write.

![Permissions](../../../assets/permissions/permissions_overview_v1.png)

### Resources: namespaces, projects, and datasets

Rules target a resource in DataChain's hierarchy. A dataset is addressed as:

```
<namespace>.<project>.<dataset>
```

for example `prod.analytics.metrics`. A rule can target any level of this
hierarchy, and it **cascades downward**: a rule on the `prod` namespace covers
every project and dataset inside it, and a rule on `prod.analytics` covers every
dataset in that project. For more on how datasets are organized, see
[Organizing Datasets with Namespace and Project](../../../guide/namespaces.md).

### Groups

Groups bundle people so one rule can grant access to many at once.

To create a group, open the **Groups** section and click **New group**. Give it a
name (for example, `ML Platform`) and an optional description. You can tick **Add
all N team members** to seed the group with everyone currently on the team, this
is a one-time snapshot, so new teammates aren't added automatically.

![Add Group](../../../assets/permissions/permissions_groups_v1.png)

To manage an existing group, use its edit button. From there you can rename it,
edit its description, and add or remove members. Removing a member revokes any
access they had through that group.
![Edit Group](../../../assets/permissions/permissions_group_members_v1.png)

Deleting a group asks for confirmation and warns that it *"Deletes this group and
every rule granted through it. Members lose that access but keep their team
role."* Deletion can't be undone.

### Rules

A rule grants a user or a group `read` or `write` on a specific resource. The
highest matching grant wins, and a namespace grant covers everything inside it
(see [How access is resolved](#how-access-is-resolved)).

To add a rule, open the **Rules** section and click **New rule**:

1. **Pick the resource.** Choose the resource type (namespace, project, or
   dataset), then select it through the cascading pickers, for a dataset you
   pick its namespace, then its project, then the dataset.
2. **Assign it.** Choose whether the rule targets a **group** or a **user**, then
   pick which one.
3. **Choose the permission**, **Read** (view only) or **Write** (read &
   modify).

![Add Rules](../../../assets/permissions/permissions_new_rule_v1.png)

Before you save, the dialog spells out exactly what the rule grants, including
the cascade (for example, "…including every project and dataset inside it"). If a
rule for that user or group on that resource already exists, the dialog offers to
update it to the new permission instead of erroring. Use **Create & add another**
to add several rules without re-walking the resource picker.

Existing rules can be browsed three ways, **All rules**, **By resource**, or
**By user / group** (the default), and filtered by resource type or search. You
can change a rule's permission inline from its row, or delete it.

![Rules](../../../assets/permissions/permissions_rules_v1.png)

## Next Steps

- [Manage collaborators](settings.md#edit-collaborators) and the rest of your
  [team settings](settings.md)
- [Configure Single Sign-on](../authentication/single-sign-on.md) so members
  authenticate through your identity provider
- Read up on
  [namespaces and projects](../../../guide/namespaces.md), the hierarchy that
  rules cascade through
