# Account Management

To open your account settings, click on your user icon on the top right corner of DataChain Studio, and go to your `Settings`. You can view and update the following settings:

- [General settings](#general-settings)
    - [Profile details](#profile-details) update your name and profile picture
    - [Account details](#account-details) manage your username, password, email addresses, and delete your account
- [Git connections](#git-connections) with GitHub, GitLab and Bitbucket
- [Cloud credentials](#cloud-credentials) for data remotes
- [Teams](#teams) that you own
- [Tokens](#tokens)
    - [Client access tokens](#client-access-tokens) for dataset and job operations

!!! note
    This does not include managing your team plan (Free or Enterprise). Team plans are defined for each team separately. [Get Enterprise](team-collaboration.md#get-enterprise).

## General settings

In your settings page, the general tab includes your profile and account settings.

### Profile details

Here, you can update your first name, last name and profile picture.

### Account details

In the account section, your username is displayed. Here, you can also update your username, password and email addresses.

!!! note
    If you signed up with a GitHub, GitLab or Bitbucket account, these details are fetched from your connected Git hosting account.

#### Managing email addresses

You can add multiple email addresses to a single DataChain Studio account. You can login to the account with any of your verified email addresses as long as you have set up a password for your account. This is true even if you signed up using your GitHub, GitLab, or Bitbucket.

One of your email addresses must be designated as primary. This is the address to which DataChain Studio will send all your account notification emails.

You can change your primary email address by clicking on the `Primary` button next to the email address which you want to designate as primary.

You can delete your non-primary email addresses.

#### Delete account

If you delete your account, all the projects you own and the links that you have shared will be permanently deleted. So, click on `Delete my account` only if you are absolutely sure that you do not need those projects or links anymore.

!!! note
    Deleting your account in DataChain Studio does not delete your Git repositories.

## Git Connections

In this section, you can:

- Connect to GitHub.com, GitLab.com or Bitbucket.org.

  When you connect to a Git hosting provider, you will be prompted to grant DataChain Studio access to your account.

  To connect to your GitHub repositories, you must install the DataChain Studio GitHub app. Refer to the section on [GitHub app installation](git-connections/github-app.md) for more details.

  Note that if you signed up to use DataChain Studio using your GitHub, GitLab or Bitbucket account, integration with that Git account will have been created during sign up.

  Also, note that **connections to self-hosted GitLab servers** are not managed in this section. If you want to connect to a self-hosted GitLab server, you should create a team and [set up the GitLab server connection](git-connections/custom-gitlab-server.md) in the team settings.

- Disconnect from your GitHub, GitLab, or Bitbucket accounts.
- Configure your GitHub account connection. That is, install the DataChain Studio GitHub app on additional organizations or repositories, or even remove the app from organizations or repositories where you no longer need it.

## Cloud credentials

In this section, you can view, add and update credentials for cloud resources. These credentials are used to fetch project data from data remotes and cloud storage.

To add new credentials, click `Add credentials` and select the cloud provider. Depending on the provider, you will be asked for more details.

The credentials must have the required permissions for accessing your data storage.

Finally, click `Save credentials`.

!!! tip
    DataChain Studio also supports [OpenID Connect authentication](authentication/openid-connect.md) for some cloud providers.

## Teams

In this section, you can view all the teams you are member of.

Click on `select` to switch to the team's dashboard. Or, click on `manage` to go to the team settings page and manage the team.

To create a new team, click on `Create a team` and enter the team name. You can invite members to the team by entering their email addresses. Find more details in the [team collaboration guide](team-collaboration.md#create-a-team).

## Tokens

### Client access tokens

In this tokens section of your settings page, you can generate new client access tokens with specific scopes as well as delete existing access tokens. These tokens can be used to give limited permissions to a client without granting full access to your Studio account. You can restrict the access token to a certain team or allow it to access all teams as well.

The available scopes are:

- `Dataset operations` - Used for managing datasets and related operations.
- `Job related operations` - Used for managing data processing jobs and related operations.
- `Admin operations` - Used for team management and project creation.
- `Storage related operations` - Used for managing storage and related operations.
