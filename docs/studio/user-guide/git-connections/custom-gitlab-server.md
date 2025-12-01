# Custom GitLab Server

Learn how to connect DataChain Studio to your self-hosted GitLab server for enterprise deployments.

## Overview

DataChain Studio supports integration with self-hosted GitLab servers, enabling:

- **Enterprise Integration**: Connect to corporate GitLab instances
- **Custom Domains**: Work with internal GitLab servers
- **Advanced Security**: Leverage enterprise security features
- **On-premises Data**: Keep code and data within your network

## Prerequisites

Before connecting to a custom GitLab server:

- **GitLab Server**: Running GitLab CE or EE (version 12.0+)
- **Network Access**: DataChain Studio must be able to reach your GitLab server
- **Admin Access**: GitLab administrator privileges for OAuth app creation
- **SSL Certificate**: Valid SSL certificate for HTTPS (recommended)

## Configuration Steps

### 1. Create OAuth Application in GitLab

1. Log in to your GitLab server as an administrator
2. Navigate to Admin Area → Applications
3. Click "New Application"
4. Configure the application:
   - **Name**: DataChain Studio
   - **Redirect URI**: `https://studio.datachain.ai/api/auth/gitlab/callback`
   - **Scopes**: Select required scopes:
     - `read_user`: Read user information
     - `read_repository`: Access repositories
     - `read_api`: API access

5. Click "Save application"
6. Copy the **Application ID** and **Secret**

### 2. Configure DataChain Studio

1. Log in to DataChain Studio
2. Go to Account Settings → Git Connections
3. Click "Add GitLab Server"
4. Enter server details:
   - **Server URL**: Your GitLab server URL (e.g., `https://gitlab.company.com`)
   - **Application ID**: From step 1
   - **Application Secret**: From step 1
   - **Server Name**: Friendly name for identification

5. Click "Save Configuration"
6. Test the connection

### 3. Team Configuration

For team-based access:

1. Create or select a team in DataChain Studio
2. Go to Team Settings → Git Connections
3. Add the custom GitLab server configuration
4. Configure team-specific access permissions

## OAuth Scopes

Required OAuth scopes for different features:

### Basic Integration
- `read_user`: Read user profile information
- `read_repository`: Access repository contents

### Advanced Features
- `read_api`: API access for webhooks and automation
- `write_repository`: Update commit statuses (optional)

### Webhook Integration
- `read_api`: Required for webhook configuration
- `admin`: May be required for some webhook operations

## Network Configuration

### Firewall Rules

Ensure proper network access:

#### Outbound (DataChain Studio → GitLab)
- **HTTPS (443)**: For API and OAuth communication
- **SSH (22)**: For Git operations (if using SSH)

#### Inbound (GitLab → DataChain Studio)
- **HTTPS (443)**: For webhook callbacks
- **Custom Port**: If using custom webhook endpoints

### SSL/TLS Configuration

For secure communication:

1. **Valid Certificate**: Use a valid SSL certificate for your GitLab server
2. **Certificate Chain**: Ensure complete certificate chain is configured
3. **TLS Version**: Use TLS 1.2 or higher
4. **Cipher Suites**: Configure secure cipher suites

## Webhook Configuration

### Automatic Configuration

DataChain Studio can automatically configure webhooks:

1. Ensure OAuth app has sufficient permissions
2. Grant `read_api` scope
3. DataChain Studio will create webhooks automatically

### Manual Configuration

If automatic configuration fails:

1. Go to your repository settings in GitLab
2. Navigate to Settings → Webhooks
3. Add webhook with:
   - **URL**: `https://studio.datachain.ai/api/webhooks/gitlab`
   - **Secret Token**: (optional but recommended)
   - **Trigger Events**: Push events, Merge requests
   - **SSL Verification**: Enable (recommended)

## Troubleshooting

### Connection Issues

#### SSL Certificate Errors
- Verify certificate validity and chain
- Check certificate matches server hostname
- Ensure DataChain Studio trusts the certificate

#### Network Connectivity
- Test connectivity from DataChain Studio to GitLab
- Check firewall rules and network policies
- Verify DNS resolution

#### OAuth Errors
- Verify Application ID and Secret
- Check redirect URI configuration
- Ensure OAuth app is enabled

### Repository Access Issues

#### Permission Denied
- Verify user has access to repositories
- Check OAuth scopes are sufficient
- Ensure repositories are not archived or disabled

#### Webhook Failures
- Check webhook URL is accessible
- Verify webhook secret configuration
- Test webhook manually from GitLab

### Performance Issues

#### Slow Repository Loading
- Check network latency between systems
- Verify GitLab server performance
- Consider repository size and complexity

#### Timeout Errors
- Increase timeout settings if possible
- Check for network bottlenecks
- Monitor GitLab server resource usage

## Security Considerations

### OAuth Security
- **Secret Protection**: Secure storage of OAuth credentials
- **Scope Limitation**: Grant minimum required scopes
- **Regular Rotation**: Rotate OAuth secrets periodically

### Network Security
- **VPN Access**: Consider VPN for additional security
- **IP Restrictions**: Limit access to specific IP ranges
- **Audit Logging**: Enable comprehensive audit logging

### Data Protection
- **Data Classification**: Classify repository data appropriately
- **Access Controls**: Implement proper access controls
- **Compliance**: Ensure compliance with data protection regulations

## Enterprise Features

### Single Sign-On (SSO)
- Integrate with corporate identity providers
- Leverage existing authentication systems
- Centralized user management

### Advanced Permissions
- Role-based access control
- Group-based permissions
- Project-level access controls

### Audit and Compliance
- Comprehensive audit logging
- Compliance reporting
- Security monitoring

## Next Steps

- Configure [team collaboration](../team-collaboration.md)
- Set up [automated workflows](../../../guide/processing.md)
- Explore [webhook integration](../../webhooks.md)
- Learn about [GitHub integration](github-app.md) as an alternative
