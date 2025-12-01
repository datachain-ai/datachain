# GitHub App

Learn how to install and configure the DataChain Studio GitHub App for seamless integration with your GitHub repositories.

## Overview

The DataChain Studio GitHub App provides secure, fine-grained access to your GitHub repositories, enabling:

- **Repository Access**: Connect public and private repositories
- **Webhook Integration**: Automatic job triggering on code changes
- **Security**: OAuth-based authentication with granular permissions
- **Team Collaboration**: Shared access across team members

## Installation

### Install for Personal Account

1. Navigate to [DataChain Studio GitHub App](https://github.com/apps/datachain-studio)
2. Click "Install" or "Configure"
3. Choose "Only select repositories" or "All repositories"
4. Select the repositories you want to connect
5. Review and approve permissions
6. Complete installation

### Install for Organization

1. Go to your organization's settings on GitHub
2. Navigate to "Third-party access" → "GitHub Apps"
3. Search for "DataChain Studio" or use the installation link
4. Configure repository access and permissions
5. Complete installation for the organization

## Configuration

### Repository Selection

Choose which repositories to connect:

- **All repositories**: Grants access to all current and future repositories
- **Selected repositories**: Choose specific repositories to connect
- **Recommended**: Start with selected repositories for better security

### Permissions

The DataChain Studio GitHub App requests these permissions:

#### Repository Permissions
- **Contents**: Read repository files and commit history
- **Metadata**: Read repository information and settings
- **Pull requests**: Read PR information for job triggering
- **Commit statuses**: Update commit status based on job results

#### Organization Permissions
- **Members**: Read organization membership (for team features)
- **Plan**: Read organization plan information

## Usage

### Creating Datasets

Once installed, you can create datasets from GitHub repositories:

1. Go to DataChain Studio
2. Click "Create Dataset"
3. Select your GitHub organization
4. Choose the repository
5. Configure dataset settings
6. Create the dataset

### Webhook Integration

The GitHub App automatically configures webhooks for:

- **Push events**: Trigger jobs on new commits
- **Pull requests**: Run validation jobs on PRs
- **Releases**: Deploy or process data on releases

## Troubleshooting

### App Not Visible

If you don't see the GitHub App or repositories:

1. **Check Installation**: Verify the app is installed on the correct account/organization
2. **Repository Access**: Ensure the app has access to the desired repositories
3. **Permissions**: Verify you have admin access to the organization
4. **Cache**: Try logging out and back into DataChain Studio

### Permission Issues

If you encounter permission errors:

1. **Review Permissions**: Check that all required permissions are granted
2. **Reinstall**: Try uninstalling and reinstalling the app
3. **Organization Approval**: Some organizations require admin approval for new apps

### Webhook Issues

If webhooks aren't triggering jobs:

1. **Check Webhook Settings**: Verify webhooks are configured in repository settings
2. **Event Types**: Ensure the correct event types are enabled
3. **Repository Access**: Confirm the app has access to the repository
4. **Network**: Check that GitHub can reach DataChain Studio servers

## Security

### Best Practices

1. **Least Privilege**: Only grant access to repositories that need DataChain integration
2. **Regular Reviews**: Periodically review and audit app permissions
3. **Organization Policies**: Follow your organization's security policies
4. **Access Monitoring**: Monitor app access logs and usage

### Permissions Audit

Regularly audit GitHub App permissions:

1. Go to your GitHub settings
2. Navigate to "Applications" → "Authorized GitHub Apps"
3. Review DataChain Studio permissions
4. Update or revoke access as needed

## Next Steps

- Learn about [custom GitLab server](custom-gitlab-server.md) integration
- Explore [team collaboration](../team-collaboration.md) features
- Set up [automated workflows](../../../guide/processing.md)
- Configure [webhooks](../../webhooks.md) for notifications
