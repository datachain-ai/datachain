# Team Settings

In the team settings page, you can change the team name, add credentials for the
data remotes, and delete the team. Note that these settings are applicable to the
team and are thus different from
[project settings](../experiments/configure-a-project.md).

Additionally, you can also
[manage connections to self-hosted GitLab servers](#manage-connections-to-self-hosted-gitlab-servers),
[configure sso](#configure-single-sign-on-sso),
[edit collaborators](#edit-collaborators), and
[set up permissions](permissions.md#permissions).

## Manage connections to self-hosted GitLab servers

If your team's Git repositories are on a self-hosted GitLab server, you can go
to the `GitLab connections` section of the team settings page to set up a
connection to this server. Once you set up the connection, all your team members
can connect to the Git repositories on this server. For more details, refer to
[Custom GitLab Server Connection](../git-connections/custom-gitlab-server.md).

## Configure Single Sign-on (SSO)

Single Sign-on (SSO) allows your team members to authenticate to DataChain
Studio using your organization's identity Provider (IdP) such as Okta, LDAP,
Microsoft AD, etc. See
[Single Sign-on](../authentication/single-sign-on.md) for how to configure it.

Once the SSO configuration is complete, users can login to DataChain Studio
using their team's login page at
`http://studio.datachain.ai/api/teams/<TEAM_NAME>/sso`. They can also login
directly from their Okta dashboards by clicking on the DataChain Studio
integration icon.

If a user does not have a pre-assigned role when they sign in to a team, they
will be auto-assigned the [`Viewer` role](permissions.md#roles).

## Edit collaborators

To manage the collaborators (team members) of your team, go to the
`Collaborators & Permissions` section of the team settings page. Here you can
invite new team members as well as remove or change the
[roles](permissions.md#roles) of existing team members. A role does not grant
access to any dataset on its own, use
[permissions](permissions.md#permissions) for that.

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

## Next Steps

- Scope access to datasets with
  [groups and rules](permissions.md#permissions)
- Connect a [self-hosted GitLab server](../git-connections/custom-gitlab-server.md)
  or the [GitHub App](../git-connections/github-app.md)
