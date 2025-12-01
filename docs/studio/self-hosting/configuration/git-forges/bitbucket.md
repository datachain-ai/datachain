# Bitbucket Configuration

This guide covers how to configure DataChain Studio to integrate with Bitbucket Cloud or Bitbucket Server (Data Center).

## Overview

DataChain Studio integrates with Bitbucket using OAuth consumers, providing:

- **Secure Authentication**: OAuth 1.0a/2.0 user authentication
- **Repository Access**: Access to Bitbucket repositories
- **Webhook Integration**: Automatic job triggering on Git events
- **Team Synchronization**: Bitbucket workspace and team mapping

## Prerequisites

- Bitbucket workspace with admin access (for Bitbucket Cloud)
- Bitbucket Server with admin access (for self-hosted)
- DataChain Studio deployment ready for configuration
- Valid domain name for DataChain Studio instance

## Bitbucket Cloud Setup

### 1. Create OAuth Consumer

1. Go to [Bitbucket Cloud](https://bitbucket.org)
2. Navigate to your workspace settings
3. Go to **Settings** → **OAuth consumers**
4. Click **Add consumer**

### 2. Configure OAuth Consumer

Fill in the consumer details:

- **Name**: `DataChain Studio`
- **Description**: `DataChain Studio integration for data processing workflows`
- **Callback URL**: `https://studio.yourcompany.com/auth/bitbucket/callback`
- **URL**: `https://studio.yourcompany.com`
- **Permissions**: Select the following:
  - **Account**: Read
  - **Team membership**: Read
  - **Repositories**: Read, Write (if needed)
  - **Pull requests**: Read, Write (if needed)
  - **Issues**: Read, Write (optional)
  - **Webhooks**: Read, Write

### 3. Save Consumer Credentials

After creating the consumer, save:
- **Key** (Client ID)
- **Secret** (Client Secret)

## Bitbucket Server Setup

### 1. Create Application Link

1. Log in to Bitbucket Server as an administrator
2. Go to **Administration** → **Application links**
3. Click **Create link**

### 2. Configure Application Link

- **Application URL**: `https://studio.yourcompany.com`
- **Application Name**: `DataChain Studio`
- **Application Type**: Generic Application
- **Service Provider Name**: `DataChain Studio`
- **Consumer Key**: Generate a unique key
- **Shared Secret**: Generate a secure secret
- **Request Token URL**: `https://studio.yourcompany.com/auth/bitbucket/request_token`
- **Access Token URL**: `https://studio.yourcompany.com/auth/bitbucket/access_token`
- **Authorize URL**: `https://studio.yourcompany.com/auth/bitbucket/authorize`

## DataChain Studio Configuration

### Bitbucket Cloud Configuration

Add the following to your `values.yaml` file:

```yaml
global:
  git:
    bitbucket:
      enabled: true
      type: "cloud"  # or "server" for Bitbucket Server
      clientId: "your-bitbucket-client-id"
      clientSecret: "your-bitbucket-client-secret"
      webhookSecret: "your-webhook-secret"
```

### Bitbucket Server Configuration

For Bitbucket Server deployments:

```yaml
global:
  git:
    bitbucket:
      enabled: true
      type: "server"
      url: "https://bitbucket.yourcompany.com"
      consumerKey: "your-consumer-key"
      consumerSecret: "your-consumer-secret"
      webhookSecret: "your-webhook-secret"

      # SSL configuration for Bitbucket Server
      ssl:
        verify: true
        caCertificate: |
          -----BEGIN CERTIFICATE-----
          ... (your Bitbucket Server CA certificate) ...
          -----END CERTIFICATE-----
```

### Advanced Configuration

For more complex setups:

```yaml
global:
  git:
    bitbucket:
      enabled: true
      type: "cloud"  # or "server"
      url: "https://bitbucket.yourcompany.com"  # Only for server
      clientId: "your-client-id"
      clientSecret: "your-client-secret"
      webhookSecret: "your-webhook-secret"

      # OAuth configuration
      oauth:
        version: "2.0"  # "1.0a" for older integrations
        scopes:
          - account
          - team
          - repository
          - pullrequest

        # Additional OAuth parameters
        redirectUri: "https://studio.yourcompany.com/auth/bitbucket/callback"

      # Webhook configuration
      webhooks:
        events:
          - repo:push
          - pullrequest:created
          - pullrequest:updated
          - pullrequest:approved
          - pullrequest:merged

        # Webhook delivery settings
        active: true

      # Rate limiting
      rateLimit:
        requestsPerHour: 1000
        burstSize: 50

      # Connection settings
      timeout:
        connect: 30s
        read: 60s
        write: 30s

      # Repository access control
      repositories:
        # Allow specific repositories
        allowList:
          - "workspace/important-repo"
          - "workspace/data-*"

        # Block specific repositories
        blockList:
          - "workspace/sensitive-repo"

      # Workspace filtering
      workspaces:
        allowList:
          - "your-workspace"
          - "partner-workspace"
        blockList:
          - "external-workspace"
```

### Secret Management

For Kubernetes deployments, store sensitive data in secrets:

```bash
# Create secret for Bitbucket OAuth credentials
kubectl create secret generic bitbucket-oauth \
  --namespace datachain-studio \
  --from-literal=client-id=your-client-id \
  --from-literal=client-secret=your-client-secret

# Create secret for webhook secret
kubectl create secret generic bitbucket-webhook \
  --namespace datachain-studio \
  --from-literal=secret=your-webhook-secret
```

Reference secrets in configuration:

```yaml
global:
  git:
    bitbucket:
      enabled: true
      type: "cloud"
      clientIdSecret:
        name: bitbucket-oauth
        key: client-id
      clientSecretSecret:
        name: bitbucket-oauth
        key: client-secret
      webhookSecretSecret:
        name: bitbucket-webhook
        key: secret
```

## Webhook Configuration

### Automatic Webhook Setup

DataChain Studio can automatically configure webhooks:

```yaml
global:
  git:
    bitbucket:
      webhooks:
        autoSetup: true
        events:
          - repo:push
          - pullrequest:created
          - pullrequest:updated
          - pullrequest:merged

        # Additional webhook settings
        active: true
        skipCertVerification: false  # Only for testing
```

### Manual Webhook Setup

If automatic setup doesn't work, configure webhooks manually:

#### Bitbucket Cloud:
1. Go to repository **Settings** → **Webhooks**
2. Click **Add webhook**
3. Configure:
   - **Title**: `DataChain Studio`
   - **URL**: `https://studio.yourcompany.com/api/webhooks/bitbucket`
   - **Status**: Active
   - **Triggers**: Select relevant events:
     - Repository push
     - Pull request created
     - Pull request updated
     - Pull request merged
   - **Skip certificate verification**: Unchecked (unless testing)

#### Bitbucket Server:
1. Go to repository **Settings** → **Hooks**
2. Enable **Web Post Hooks**
3. Configure:
   - **URL**: `https://studio.yourcompany.com/api/webhooks/bitbucket`
   - **Secret**: Your webhook secret

## User Authentication

Configure Bitbucket OAuth for user authentication:

```yaml
global:
  auth:
    bitbucket:
      enabled: true
      type: "cloud"  # or "server"
      clientId: "your-oauth-client-id"
      clientSecret: "your-oauth-client-secret"

      # OAuth scopes (for Cloud)
      scopes:
        - account
        - team
        - repository

      # Team synchronization
      teamSync:
        enabled: true
        workspaceWhitelist:
          - "your-workspace"
```

## Permissions and Access Control

### Repository-Level Permissions

Configure fine-grained repository access:

```yaml
global:
  git:
    bitbucket:
      permissions:
        # Default repository permissions
        default:
          repository: read
          pullrequest: read

        # Custom permissions for specific repositories
        repositories:
          "workspace/critical-repo":
            repository: read
            pullrequest: write
            issues: read
```

### Team Mapping

Map Bitbucket teams to DataChain Studio roles:

```yaml
global:
  teams:
    bitbucket:
      mapping:
        # Bitbucket team → Studio role
        "developers": "member"
        "data-engineers": "member"
        "administrators": "admin"
        "viewers": "viewer"

      # Workspace-wide settings
      defaultRole: "viewer"
      syncInterval: "1h"
```

## Bitbucket Pipelines Integration

### Pipeline Triggers

Configure pipeline triggers from DataChain Studio:

```yaml
global:
  git:
    bitbucket:
      pipelines:
        enabled: true

        # Pipeline trigger settings
        triggers:
          # Trigger on data changes
          dataChange:
            enabled: true
            branch: "main"
            variables:
              DATACHAIN_TRIGGER: "data_change"

          # Custom pipeline variables
          customVariables:
            DATACHAIN_STUDIO_URL: "https://studio.yourcompany.com"
            DATACHAIN_WEBHOOK_SECRET: "webhook-secret"
```

### Build Status Updates

Update Bitbucket commit status from DataChain Studio jobs:

```yaml
global:
  git:
    bitbucket:
      buildStatus:
        enabled: true

        # Status contexts
        contexts:
          dataProcessing: "datachain/processing"
          dataValidation: "datachain/validation"
          dataQuality: "datachain/quality"

        # Status details
        url: "https://studio.yourcompany.com/jobs/{job_id}"
        description: "DataChain data processing job"
```

## Monitoring and Debugging

### Health Checks

Monitor Bitbucket integration health:

```yaml
monitoring:
  bitbucket:
    enabled: true

    healthChecks:
      api: true
      webhooks: true
      oauth: true

    metrics:
      - apiCalls
      - responseTime
      - errorRate
      - webhookDelivery

    alerts:
      - name: "Bitbucket API Errors"
        condition: "bitbucket_api_error_rate > 5%"
        duration: "5m"
        severity: "warning"

      - name: "Bitbucket Webhook Failures"
        condition: "bitbucket_webhook_failure_rate > 10%"
        duration: "5m"
        severity: "critical"
```

### Debug Configuration

Enable debug logging for Bitbucket integration:

```yaml
global:
  logging:
    level: DEBUG
    components:
      bitbucket: DEBUG
      webhooks: DEBUG
      oauth: DEBUG
```

## Testing the Integration

### Test Bitbucket API Access

#### Bitbucket Cloud:
```bash
# Test API connectivity
curl -H "Authorization: Bearer $BITBUCKET_TOKEN" \
  https://api.bitbucket.org/2.0/user

# Test repository access
curl -H "Authorization: Bearer $BITBUCKET_TOKEN" \
  https://api.bitbucket.org/2.0/repositories/workspace
```

#### Bitbucket Server:
```bash
# Test API connectivity
curl -H "Authorization: Bearer $BITBUCKET_TOKEN" \
  https://bitbucket.yourcompany.com/rest/api/1.0/projects

# Test user information
curl -H "Authorization: Bearer $BITBUCKET_TOKEN" \
  https://bitbucket.yourcompany.com/rest/api/1.0/users/username
```

### Test OAuth Flow

```bash
# Test OAuth authorization URL (Cloud)
curl "https://bitbucket.org/site/oauth2/authorize?client_id=YOUR_CLIENT_ID&redirect_uri=https://studio.yourcompany.com/auth/bitbucket/callback&response_type=code"
```

### Test Webhook Delivery

```bash
# Test webhook endpoint
curl -X POST https://studio.yourcompany.com/api/webhooks/bitbucket \
  -H "Content-Type: application/json" \
  -H "X-Event-Key: repo:push" \
  -H "X-Hook-UUID: webhook-uuid" \
  -d '{
    "push": {
      "changes": [{
        "new": {
          "name": "main",
          "target": {
            "hash": "abcdef123456"
          }
        }
      }]
    },
    "repository": {
      "name": "test-repo",
      "full_name": "workspace/test-repo"
    }
  }'
```

## Troubleshooting

### Common Issues

**OAuth authentication failures:**
- Verify client ID and secret are correct
- Check callback URL matches exactly
- Ensure required permissions are granted
- Verify OAuth version (1.0a vs 2.0)

**API connectivity issues:**
- Test Bitbucket API endpoint accessibility
- Check SSL certificate validity
- Verify network connectivity
- Review API rate limits

**Webhook delivery failures:**
- Confirm webhook URL is accessible
- Verify webhook secret matches
- Check SSL certificate validity
- Review webhook event configuration

### Debug Commands

```bash
# Check Bitbucket configuration
kubectl get configmap datachain-studio-config -n datachain-studio -o yaml | grep -A 20 bitbucket

# View Bitbucket-related logs
kubectl logs -f deployment/datachain-studio-backend -n datachain-studio | grep -i bitbucket

# Test Bitbucket API from container
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  curl -v https://api.bitbucket.org/2.0/user

# Test OAuth endpoint
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  curl -v https://bitbucket.org/site/oauth2/access_token
```

## Security Considerations

### OAuth Security

- Use confidential OAuth consumers
- Regularly rotate client secrets
- Limit OAuth scopes to minimum required
- Monitor OAuth token usage

### Webhook Security

- Always use webhook secrets
- Validate webhook signatures
- Use HTTPS for webhook URLs
- Monitor webhook delivery patterns

### Access Control

- Use principle of least privilege
- Regularly audit repository access
- Monitor API usage patterns
- Review team/workspace permissions

## Migration from Other Git Forges

When migrating from other Git forges to Bitbucket:

1. **Export existing configuration**
2. **Set up Bitbucket OAuth consumer**
3. **Configure DataChain Studio for Bitbucket**
4. **Migrate repository connections**
5. **Update webhook configurations**
6. **Test integration thoroughly**
7. **Update user authentication**

## Next Steps

- Configure [GitHub integration](github.md) for additional Git forges
- Set up [GitLab integration](gitlab.md) if needed
- Review [SSL/TLS configuration](../ssl-tls.md) for secure communications
- Check [troubleshooting guide](../../troubleshooting/index.md) for common issues
- Configure [monitoring and alerting](../index.md#monitoring) for the integration
