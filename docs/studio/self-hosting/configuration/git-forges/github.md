# GitHub Configuration

This guide covers how to configure DataChain Studio to integrate with GitHub.com or GitHub Enterprise Server.

## Overview

DataChain Studio integrates with GitHub using GitHub Apps, providing:

- **Secure Authentication**: OAuth-based user authentication
- **Repository Access**: Fine-grained access to repositories
- **Webhook Integration**: Automatic job triggering on Git events
- **Team Synchronization**: GitHub organization and team mapping

## Prerequisites

- GitHub organization with admin access
- DataChain Studio deployment ready for configuration
- Valid domain name for DataChain Studio instance

## GitHub App Setup

### 1. Create GitHub App

1. Navigate to your GitHub organization settings
2. Go to **Settings** → **Developer settings** → **GitHub Apps**
3. Click **New GitHub App**

### 2. Configure Basic Information

Fill in the application details:

- **GitHub App name**: `DataChain Studio`
- **Description**: `DataChain Studio integration for data processing workflows`
- **Homepage URL**: `https://studio.yourcompany.com`
- **User authorization callback URL**: `https://studio.yourcompany.com/auth/github/callback`
- **Setup URL**: `https://studio.yourcompany.com/setup/github`
- **Webhook URL**: `https://studio.yourcompany.com/api/webhooks/github`
- **Webhook secret**: Generate a secure random string (save this for later)

### 3. Configure Permissions

Set the following repository permissions:

#### Repository Permissions
- **Contents**: Read (for accessing repository files)
- **Metadata**: Read (for repository information)
- **Pull requests**: Read (for PR information)
- **Commit statuses**: Write (for updating commit status)
- **Issues**: Read (optional, for issue tracking)

#### Organization Permissions
- **Members**: Read (for team synchronization)
- **Plan**: Read (for organization information)

### 4. Subscribe to Events

Enable the following webhook events:

- **Push** - For triggering jobs on code changes
- **Pull request** - For PR-based workflows
- **Release** - For release-based deployments
- **Repository** - For repository changes
- **Installation** - For app installation changes

### 5. Generate Private Key

1. After creating the app, scroll down to **Private keys**
2. Click **Generate a private key**
3. Download the `.pem` file (you'll need this for configuration)

### 6. Install the App

1. Go to **Install App** tab
2. Install the app on your organization
3. Choose repositories (all or selected)
4. Complete the installation

## DataChain Studio Configuration

### Basic Configuration

Add the following to your `values.yaml` file:

```yaml
global:
  git:
    github:
      enabled: true
      appId: "123456"  # Your GitHub App ID
      privateKey: |
        -----BEGIN RSA PRIVATE KEY-----
        MIIEpAIBAAKCAQEA1234567890abcdef...
        ... (your GitHub App private key) ...
        -----END RSA PRIVATE KEY-----
      webhookSecret: "your-webhook-secret"

      # Optional: GitHub Enterprise Server URL
      # url: "https://github.enterprise.com"
```

### Advanced Configuration

For more advanced setups:

```yaml
global:
  git:
    github:
      enabled: true
      appId: "123456"
      privateKey: |
        -----BEGIN RSA PRIVATE KEY-----
        ... (private key content) ...
        -----END RSA PRIVATE KEY-----
      webhookSecret: "your-webhook-secret"

      # GitHub Enterprise Server configuration
      url: "https://github.enterprise.com"
      apiUrl: "https://github.enterprise.com/api/v3"

      # SSL configuration for GitHub Enterprise
      ssl:
        verify: true
        caCertificate: |
          -----BEGIN CERTIFICATE-----
          ... (GitHub Enterprise CA certificate) ...
          -----END CERTIFICATE-----

      # Webhook configuration
      webhooks:
        events:
          - push
          - pull_request
          - release
          - repository

        # Custom webhook settings
        deliveryTimeout: 30s
        retryAttempts: 3

      # Rate limiting
      rateLimit:
        requestsPerHour: 5000
        burstSize: 100

      # Repository access control
      repositories:
        # Allow specific repositories
        allowList:
          - "org/important-repo"
          - "org/data-*"

        # Block specific repositories
        blockList:
          - "org/sensitive-repo"

      # Organization filtering
      organizations:
        allowList:
          - "your-org"
          - "partner-org"
        blockList:
          - "external-org"
```

### Secret Management

For Kubernetes deployments, store sensitive data in secrets:

```bash
# Create secret for GitHub App private key
kubectl create secret generic github-app-key \
  --namespace datachain-studio \
  --from-file=private-key=/path/to/github-app.pem

# Create secret for webhook secret
kubectl create secret generic github-webhook \
  --namespace datachain-studio \
  --from-literal=secret=your-webhook-secret
```

Then reference in your configuration:

```yaml
global:
  git:
    github:
      enabled: true
      appId: "123456"
      privateKeySecret:
        name: github-app-key
        key: private-key
      webhookSecretSecret:
        name: github-webhook
        key: secret
```

## GitHub Enterprise Server

For GitHub Enterprise Server deployments:

```yaml
global:
  git:
    github:
      enabled: true
      appId: "your-app-id"
      url: "https://github.enterprise.com"
      apiUrl: "https://github.enterprise.com/api/v3"

      # Upload URL for Enterprise Server
      uploadUrl: "https://github.enterprise.com/api/uploads"

      privateKey: |
        -----BEGIN RSA PRIVATE KEY-----
        ... private key ...
        -----END RSA PRIVATE KEY-----

      # Custom CA certificate for Enterprise Server
      ssl:
        verify: true
        caCertificate: |
          -----BEGIN CERTIFICATE-----
          ... Enterprise Server CA certificate ...
          -----END CERTIFICATE-----
```

## Webhook Configuration

### Automatic Webhook Setup

DataChain Studio can automatically configure webhooks:

```yaml
global:
  git:
    github:
      webhooks:
        autoSetup: true
        events:
          - push
          - pull_request
          - release

        # Webhook delivery settings
        contentType: "application/json"
        insecureSSL: false  # Set to true only for testing
        active: true
```

### Manual Webhook Setup

If automatic setup doesn't work, configure webhooks manually:

1. Go to repository **Settings** → **Webhooks**
2. Click **Add webhook**
3. Configure:
   - **Payload URL**: `https://studio.yourcompany.com/api/webhooks/github`
   - **Content type**: `application/json`
   - **Secret**: Your webhook secret
   - **Events**: Select individual events or "Send me everything"
   - **Active**: ✓ Checked

## User Authentication

Configure GitHub OAuth for user authentication:

```yaml
global:
  auth:
    github:
      enabled: true
      clientId: "your-oauth-app-client-id"
      clientSecret: "your-oauth-app-client-secret"

      # OAuth scopes
      scopes:
        - user:email
        - read:org
        - repo

      # Team synchronization
      teamSync:
        enabled: true
        organizationWhitelist:
          - "your-org"
```

## Permissions and Access Control

### Repository-Level Permissions

Configure fine-grained repository access:

```yaml
global:
  git:
    github:
      permissions:
        # Default repository permissions
        default:
          contents: read
          metadata: read
          pull_requests: read

        # Custom permissions for specific repositories
        repositories:
          "org/critical-repo":
            contents: read
            metadata: read
            pull_requests: write
            commit_statuses: write
```

### Team Mapping

Map GitHub teams to DataChain Studio roles:

```yaml
global:
  teams:
    github:
      mapping:
        # GitHub team slug → Studio role
        "developers": "member"
        "data-engineers": "member"
        "admin-team": "admin"
        "read-only": "viewer"

      # Organization-wide settings
      defaultRole: "viewer"
      syncInterval: "1h"
```

## Monitoring and Debugging

### Health Checks

Monitor GitHub integration health:

```yaml
monitoring:
  github:
    enabled: true

    healthChecks:
      api: true
      webhooks: true
      rateLimit: true

    alerts:
      - name: "GitHub API Rate Limit"
        condition: "github_rate_limit_remaining < 100"
        severity: "warning"

      - name: "GitHub Webhook Failures"
        condition: "github_webhook_failure_rate > 5%"
        severity: "critical"
```

### Debug Configuration

Enable debug logging for GitHub integration:

```yaml
global:
  logging:
    level: DEBUG
    components:
      github: DEBUG
      webhooks: DEBUG
```

## Testing the Integration

### Test GitHub App Installation

```bash
# Check app installation status
curl -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://api.github.com/app/installations

# Test repository access
curl -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://api.github.com/installation/repositories
```

### Test Webhook Delivery

```bash
# Test webhook endpoint
curl -X POST https://studio.yourcompany.com/api/webhooks/github \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: ping" \
  -H "X-GitHub-Delivery: 12345-678-90" \
  -H "X-Hub-Signature-256: sha256=..." \
  -d '{"zen": "Testing webhook delivery"}'
```

### Validate Configuration

```bash
# Test GitHub API connectivity
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  curl -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/user

# Check webhook configuration
kubectl logs -f deployment/datachain-studio-backend -n datachain-studio | grep -i github
```

## Troubleshooting

### Common Issues

**App installation failures:**
- Verify app permissions are correct
- Check organization access settings
- Ensure webhook URL is accessible

**Authentication errors:**
- Validate GitHub App ID and private key
- Check private key format (PEM)
- Verify webhook secret matches

**Webhook delivery failures:**
- Check webhook URL accessibility
- Verify SSL certificate validity
- Review webhook event configuration

### Debug Commands

```bash
# Check GitHub App configuration
kubectl get configmap datachain-studio-config -n datachain-studio -o yaml

# View GitHub-related logs
kubectl logs -f deployment/datachain-studio-backend -n datachain-studio | grep -i github

# Test GitHub API connectivity
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  curl -v https://api.github.com
```

## Security Considerations

### Private Key Security

- Store private keys in Kubernetes secrets
- Rotate private keys regularly
- Limit access to private key files
- Use RBAC to control secret access

### Webhook Security

- Always use webhook secrets
- Validate webhook signatures
- Use HTTPS for webhook URLs
- Monitor webhook delivery logs

### Access Control

- Use principle of least privilege
- Regularly audit app permissions
- Monitor app installation changes
- Review repository access patterns

## Next Steps

- Configure [GitLab integration](gitlab.md) for additional Git forges
- Set up [SSL/TLS certificates](../ssl-tls.md) for secure communications
- Review [troubleshooting guide](../../troubleshooting/index.md) for common issues
- Configure [monitoring and alerting](../index.md#monitoring) for the integration
