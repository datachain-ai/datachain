# Teams

DataChain Studio enables collaborative work through teams, allowing you to share
projects, datasets, and jobs with team members. You can create teams with one or
more team members, also called collaborators, and assign different roles to
control access permissions. The projects that you create in your team's page
will be accessible to all members of the team.

In this page, you will learn about:

- [How to create a team](#create-a-team)
- [How to invite collaborators (team members)](#invite-collaborators)
- [The privileges (access permissions) of different roles](#roles)
- [How to manage connections to self-hosted GitLab servers](#manage-connections-to-self-hosted-gitlab-servers)
- [How to configure Single Sign-on (SSO)](#configure-single-sign-on-sso)
- [How to upgrade to an Enterprise plan](#get-enterprise)

## Create a team

Click on the drop down next to `Personal`. All the teams that you have created
so far will be listed within `Teams` in the drop down menu. If you have not
created any team so far, this list will be empty.

To create a new team, click on `Create a team`.
![](https://static.iterative.ai/img/studio/team_create_v3.png)

You will be asked to enter the URL namespace for your team. Enter a unique name.
The URL for your team will be formed using this name.
![](https://static.iterative.ai/img/studio/team_enter_name_v3.png)

Then, click the `Create team` button on the top right corner.

## Invite collaborators

To add collaborators, enter their email addresses. Each collaborator can be
assigned the [Admin, Edit, or View role](#roles). An email invite will be sent
to each invitee. Then, click on `Send invites and close`.

![](https://static.iterative.ai/img/studio/team_roles_v3.png)

You can also click on `Skip and close` to skip adding collaborators while
creating the team, and
[add them later by accessing team settings](#edit-collaborators).

## Roles

Team members or collaborators can have the following roles:

- **`Viewers`** (Read permission) - Have read-only access to datasets, jobs,
  queries, and projects. They can view and explore data but cannot make any
  changes or create new resources.
- **`Editors`** (Write permission) - Can create and edit datasets, jobs,
  queries, and projects. They can upload files, run jobs, and manage team
  resources but cannot modify team settings or manage collaborators.
- **`Admins`** (Admin permission) - Have full access to all team resources and
  settings. They can add (invite) and remove collaborators, manage team
  settings, configure cloud credentials, and perform all operations available to
  Editors and Viewers.

DataChain Studio does not have the concept of an `Owner` role. The user who
creates the team has the `Admin` role. The privileges of such an admin is the
same as that of any other collaborator who has been assigned the `Admin` role.

!!! note

    If your Git account does not have write access on the Git repository connected
    to a project, you cannot push changes (e.g., new experiments) to the repository
    even if the project belongs to a team where you are an `Editor` or `Admin`.


### Privileges for datasets

| Feature                     | Viewer | Editor | Admin |
| --------------------------- | ------ | ------ | ----- |
| List datasets               | Yes    | Yes    | Yes   |
| View dataset information    | Yes    | Yes    | Yes   |
| View dataset rows           | Yes    | Yes    | Yes   |
| View dataset versions       | Yes    | Yes    | Yes   |
| Export datasets             | Yes    | Yes    | Yes   |
| Preview files               | Yes    | Yes    | Yes   |
| Create datasets             | No     | Yes    | Yes   |
| Edit dataset metadata       | No     | Yes    | Yes   |
| Delete datasets             | No     | Yes    | Yes   |
| Upload files                | No     | Yes    | Yes   |
| Move files in storage       | No     | Yes    | Yes   |
| Delete files                | No     | Yes    | Yes   |
| Reindex storage             | No     | Yes    | Yes   |
| Create dataset from storage | No     | Yes    | Yes   |

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

| Feature                            | Viewer | Editor | Admin |
| ---------------------------------- | ------ | ------ | ----- |
| Manage team settings               | No     | No     | Yes   |
| Manage team collaborators          | No     | No     | Yes   |
| Configure cloud credentials        | No     | No     | Yes   |
| Manage GitLab server connections   | No     | No     | Yes   |
| Configure Single Sign-on (SSO)     | No     | No     | Yes   |
| Manage team plan and billing       | No     | No     | Yes   |
| Delete a team                      | No     | No     | Yes   |

## Manage your team and its resources

Once you have created the team, the team's workspace opens up.

![](https://static.iterative.ai/img/studio/team_page_v6.png)

In this workspace, you can manage the team's:
- [Datasets](#datasets)
- [Jobs](#jobs)
- [Projects (Experiments)](#projects-dvc-experiments)
- [Settings](#settings)

## Datasets

The datasets dashboard displays all datasets created by team members. Access
permissions are controlled by team roles:
- **Viewers** can explore and export datasets
- **Editors** can create, edit, and delete datasets
- **Admins** have full control over all datasets

To create a new dataset, you can upload files, connect to cloud storage, or
create datasets from DataChain queries.

## Jobs

The jobs dashboard shows all DataChain jobs running on the team's compute
clusters. Team members can:
- **Viewers** can view job status and logs
- **Editors** can create, run, and cancel jobs
- **Admins** have full control over all jobs

## Projects (Experiments)

This is the projects dashboard for DVC (acquired by lakeFS) experiment
tracking. All projects on this
dashboard are accessible to all team members based on their roles.

To add a project to this dashboard, click on `Add a project`. The process for
adding a project is the same as that for adding personal projects
([instructions](./experiments/create-a-project.md)).

## Settings

In the [team settings](#settings) page, you can change the team name, add
credentials for the data remotes, and delete the team. Note that these settings
are applicable to the team and are thus different from
[project settings](./experiments/configure-a-project.md).

Additionally, you can also
[manage connections to self-hosted GitLab servers](#manage-connections-to-self-hosted-gitlab-servers),
[configure sso](#configure-single-sign-on-sso),
[edit collaborators](#edit-collaborators).

### Manage connections to self-hosted GitLab servers

If your team’s Git repositories are on a self-hosted GitLab server, you can go
to the `GitLab connections` section of the team settings page to set up a
connection to this server. Once you set up the connection, all your team members
can connect to the Git repositories on this server. For more details, refer to
[Custom GitLab Server Connection](./git-connections/custom-gitlab-server.md).

### Configure Single Sign-on (SSO)

Single Sign-on (SSO) allows your team members to authenticate to DataChain
Studio using your organization's identity Provider (IdP) such as Okta, LDAP,
Microsoft AD, etc.

Details on how to configure SSO for your team can be found
[here](./authentication/single-sign-on.md).

Once the SSO configuration is complete, users can login to DataChain Studio
using their team's login page at
`http://studio.datachain.ai/api/teams/<TEAM_NAME>/sso`. They can also login
directly from their Okta dashboards by clicking on the DataChain Studio
integration icon.

If a user does not have a pre-assigned role when they sign in to a team, they
will be auto-assigned the [`Viewer` role](#roles).

### Edit collaborators

To manage the collaborators (team members) of your team, go to the
`Collaborators` section of the team settings page. Here you can invite new team
members as well as remove or change the [roles](#roles) of existing team
members.

The number of collaborators in your team depends on your team plan. By default,
all teams are on the Free plan, and can have 2 collaborators. To add more
collaborators, [upgrade to the Enterprise plan](#get-enterprise).

All collaborators and pending invites get counted in the subscription. Suppose
you have subscribed for a 10 member team. If you have 5 members who have
accepted your team invite and 3 pending invites, then you will have 2 remaining
seats. This means that you can invite 2 more collaborators. At this point, if
you remove any one team member or pending invite, that seat becomes available
and so you will have 3 remaining seats.

## Get Enterprise

**To upgrade to the Enterprise plan**, [schedule a call] with our in-house
experts. They will try to understand your needs and suggest a suitable plan and
pricing.

[schedule a call]: https://calendly.com/gtm-2/studio-introduction
