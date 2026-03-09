# GitLab Configuration

This guide covers how to configure DataChain Studio to integrate with GitLab.com or self-hosted GitLab instances.

## Overview

DataChain Studio integrates with GitLab using OAuth applications, providing:

- **Secure Authentication**: OAuth 2.0 user authentication
- **Repository Access**: Access to GitLab repositories and projects
- **Webhook Integration**: Automatic job triggering on Git events
- **Group Synchronization**: GitLab group and project mapping

## Prerequisites

- GitLab instance with admin access (GitLab.com or self-hosted)
- DataChain Studio deployment ready for configuration
- Valid domain name for DataChain Studio instance

## GitLab OAuth Application Setup

### 1. Create OAuth Application

#### For GitLab.com:
1. Go to [GitLab.com](https://gitlab.com)
2. Navigate to **Settings** → **Applications**
3. Click **New application**

#### For Self-hosted GitLab:
1. Log in to your GitLab instance as an administrator
2. Go to **Admin Area** → **Applications**
3. Click **New application**

### 2. Configure Application Settings

Fill in the application details:

- **Name**: `DataChain Studio`
- **Redirect URI**: `https://studio.yourcompany.com/auth/gitlab/callback`
- **Confidential**: ✓ Checked
- **Scopes**: Select the following:
  - `read_user` - Read user information
  - `read_repository` - Read repository contents
  - `read_api` - Read access to API
  - `write_repository` - Write access to repositories (optional)

### 3. Save Application Credentials

After creating the application, save:
- **Application ID** (Client ID)
- **Secret** (Client Secret)

### 4. Configure Webhooks (Optional)

For automatic webhook setup, ensure your GitLab user/admin has:
- Admin access to repositories/groups where webhooks will be created
- API access permissions

## DataChain Studio Configuration

### Basic Configuration

Add the following to your `values.yaml` file:

```yaml
global:
  git:
    gitlab:
      enabled: true
      url: "https://gitlab.com"  # Or your GitLab instance URL
      clientId: "your-gitlab-client-id"
      clientSecret: "your-gitlab-client-secret"
      webhookSecret: "your-webhook-secret"
```

### Self-hosted GitLab Configuration

For self-hosted GitLab instances:

```yaml
global:
  git:
    gitlab:
      enabled: true
      url: "https://gitlab.yourcompany.com"
      apiUrl: "https://gitlab.yourcompany.com/api/v4"
      clientId: "your-gitlab-client-id"
      clientSecret: "your-gitlab-client-secret"
      webhookSecret: "your-webhook-secret"

      # SSL configuration for self-hosted GitLab
      ssl:
        verify: true
        caCertificate: |
          -----BEGIN CERTIFICATE-----
          ... (your GitLab instance CA certificate) ...
          -----END CERTIFICATE-----
```

### Advanced Configuration

For more complex setups:

```yaml
global:
  git:
    gitlab:
      enabled: true
      url: "https://gitlab.yourcompany.com"
      apiUrl: "https://gitlab.yourcompany.com/api/v4"
      clientId: "your-gitlab-client-id"
      clientSecret: "your-gitlab-client-secret"
      webhookSecret: "your-webhook-secret"

      # OAuth configuration
      oauth:
        scopes:
          - read_user
          - read_repository
          - read_api

        # Additional OAuth parameters
        redirectUri: "https://studio.yourcompany.com/auth/gitlab/callback"

      # Webhook configuration
      webhooks:
        events:
          - push
          - merge_requests
          - tag_push
          - releases

        # Webhook delivery settings
        enableSSLVerification: true
        pushEventsBranchFilter: ""  # All branches

      # Rate limiting
      rateLimit:
        requestsPerMinute: 600
        burstSize: 100

      # Connection settings
      timeout:
        connect: 30s
        read: 60s
        write: 30s

      # Repository/project access control
      projects:
        # Allow specific projects
        allowList:
          - "group/important-project"
          - "group/data-*"

        # Block specific projects
        blockList:
          - "group/sensitive-project"

      # Group filtering
      groups:
        allowList:
          - "data-team"
          - "engineering"
        blockList:
          - "external-group"
```

### Secret Management

For Kubernetes deployments, store sensitive data in secrets:

```bash
# Create secret for GitLab OAuth credentials
kubectl create secret generic gitlab-oauth \
  --namespace datachain-studio \
  --from-literal=client-id=your-client-id \
  --from-literal=client-secret=your-client-secret

# Create secret for webhook secret
kubectl create secret generic gitlab-webhook \
  --namespace datachain-studio \
  --from-literal=secret=your-webhook-secret
```

Reference secrets in configuration:

```yaml
global:
  git:
    gitlab:
      enabled: true
      url: "https://gitlab.yourcompany.com"
      clientIdSecret:
        name: gitlab-oauth
        key: client-id
      clientSecretSecret:
        name: gitlab-oauth
        key: client-secret
      webhookSecretSecret:
        name: gitlab-webhook
        key: secret
```

## Webhook Configuration

### Automatic Webhook Setup

DataChain Studio can automatically configure webhooks:

```yaml
global:
  git:
    gitlab:
      webhooks:
        autoSetup: true
        events:
          - push_events
          - merge_requests_events
          - tag_push_events
          - releases_events

        # Additional webhook settings
        issues_events: false
        wiki_page_events: false
        deployment_events: false
        job_events: false
        pipeline_events: false

        # Security settings
        enable_ssl_verification: true
        push_events_branch_filter: ""
```

### Manual Webhook Setup

If automatic setup doesn't work, configure webhooks manually:

#### Project-level Webhooks:
1. Go to project **Settings** → **Webhooks**
2. Add webhook with:
   - **URL**: `https://studio.yourcompany.com/api/webhooks/gitlab`
   - **Secret Token**: Your webhook secret
   - **Trigger Events**:
     - ✓ Push events
     - ✓ Merge request events
     - ✓ Tag push events
     - ✓ Releases events
   - **SSL verification**: ✓ Enable SSL verification

#### Group-level Webhooks:
1. Go to group **Settings** → **Webhooks**
2. Configure the same settings as project-level webhooks

## User Authentication

Configure GitLab OAuth for user authentication:

```yaml
global:
  auth:
    gitlab:
      enabled: true
      url: "https://gitlab.yourcompany.com"
      clientId: "your-oauth-client-id"
      clientSecret: "your-oauth-client-secret"

      # OAuth scopes
      scopes:
        - read_user
        - read_repository
        - read_api

      # Group synchronization
      groupSync:
        enabled: true
        groupWhitelist:
          - "data-team"
          - "engineering"
```

## Permissions and Access Control

### Project-Level Permissions

Configure fine-grained project access:

```yaml
global:
  git:
    gitlab:
      permissions:
        # Default project permissions
        default:
          repository: read
          issues: read
          merge_requests: read

        # Custom permissions for specific projects
        projects:
          "group/critical-project":
            repository: read
            issues: write
            merge_requests: write
            deployments: read
```

### Group Mapping

Map GitLab groups to DataChain Studio roles:

```yaml
global:
  teams:
    gitlab:
      mapping:
        # GitLab group path → Studio role
        "data-engineers": "member"
        "senior-engineers": "admin"
        "analysts": "viewer"
        "contractors": "viewer"

      # Group-wide settings
      defaultRole: "viewer"
      syncInterval: "1h"

      # Nested group handling
      includeSubgroups: true
```

## GitLab CI/CD Integration

### Pipeline Triggers

Configure pipeline triggers from DataChain Studio:

```yaml
global:
  git:
    gitlab:
      ci:
        enabled: true

        # Pipeline trigger settings
        triggers:
          # Trigger on data changes
          dataChange:
            enabled: true
            branch: "main"
            variables:
              DATACHAIN_TRIGGER: "data_change"

          # Trigger on schedule
          scheduled:
            enabled: true
            cron: "0 2 * * *"
            variables:
              DATACHAIN_TRIGGER: "scheduled"

        # Job monitoring
        monitoring:
          enabled: true
          pollInterval: 30s
```

### Job Status Updates

Update GitLab commit status from DataChain Studio jobs:

```yaml
global:
  git:
    gitlab:
      commitStatus:
        enabled: true

        # Status contexts
        contexts:
          dataProcessing: "datachain/processing"
          dataValidation: "datachain/validation"
          dataQuality: "datachain/quality"

        # Status details
        targetUrl: "https://studio.yourcompany.com/jobs/{job_id}"
        description: "DataChain data processing job"
```

## Monitoring and Debugging

### Health Checks

Monitor GitLab integration health:

```yaml
monitoring:
  gitlab:
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
      - name: "GitLab API Errors"
        condition: "gitlab_api_error_rate > 5%"
        duration: "5m"
        severity: "warning"

      - name: "GitLab Webhook Failures"
        condition: "gitlab_webhook_failure_rate > 10%"
        duration: "5m"
        severity: "critical"
```

### Debug Configuration

Enable debug logging for GitLab integration:

```yaml
global:
  logging:
    level: DEBUG
    components:
      gitlab: DEBUG
      webhooks: DEBUG
      oauth: DEBUG
```

## Testing the Integration

### Test GitLab API Access

```bash
# Test API connectivity
curl -H "Authorization: Bearer $GITLAB_TOKEN" \
  https://gitlab.yourcompany.com/api/v4/user

# Test project access
curl -H "Authorization: Bearer $GITLAB_TOKEN" \
  https://gitlab.yourcompany.com/api/v4/projects
```

### Test OAuth Flow

```bash
# Test OAuth authorization URL
curl "https://gitlab.yourcompany.com/oauth/authorize?client_id=YOUR_CLIENT_ID&redirect_uri=https://studio.yourcompany.com/auth/gitlab/callback&response_type=code&scope=read_user+read_repository"
```

### Test Webhook Delivery

```bash
# Test webhook endpoint
curl -X POST https://studio.yourcompany.com/api/webhooks/gitlab \
  -H "Content-Type: application/json" \
  -H "X-Gitlab-Event: Push Hook" \
  -H "X-Gitlab-Token: your-webhook-secret" \
  -d '{
    "object_kind": "push",
    "ref": "refs/heads/main",
    "project": {
      "name": "test-project",
      "web_url": "https://gitlab.yourcompany.com/group/test-project"
    }
  }'
```

## Troubleshooting

### Common Issues

**OAuth authentication failures:**
- Verify client ID and secret are correct
- Check redirect URI matches exactly
- Ensure required scopes are granted
- Verify GitLab instance URL is correct

**API connectivity issues:**
- Test GitLab API endpoint accessibility
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
# Check GitLab configuration
kubectl get configmap datachain-studio-config -n datachain-studio -o yaml | grep -A 20 gitlab

# View GitLab-related logs
kubectl logs -f deployment/datachain-studio-backend -n datachain-studio | grep -i gitlab

# Test GitLab API from container
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  curl -v https://gitlab.yourcompany.com/api/v4/version

# Test OAuth endpoint
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  curl -v https://gitlab.yourcompany.com/oauth/token
```

## Security Considerations

### OAuth Security

- Use confidential OAuth applications
- Regularly rotate client secrets
- Limit OAuth scopes to minimum required
- Monitor OAuth token usage

### Webhook Security

- Always use webhook secrets
- Validate webhook signatures
- Use HTTPS for webhook URLs
- Monitor webhook delivery patterns

### Network Security

- Use TLS for all GitLab communications
- Validate SSL certificates
- Consider IP whitelisting
- Monitor API access patterns

## Migration from Other Git Forges

When migrating from other Git forges to GitLab:

1. **Export existing configuration**
2. **Set up GitLab OAuth application**
3. **Configure DataChain Studio for GitLab**
4. **Migrate repository connections**
5. **Update webhook configurations**
6. **Test integration thoroughly**
7. **Update user authentication**

## Next Steps

- Configure [GitHub integration](github.md) for additional Git forges
- Set up [Bitbucket integration](bitbucket.md) if needed
- Review [SSL/TLS configuration](../ssl-tls.md) for secure communications
- Check [troubleshooting guide](../../troubleshooting/index.md) for common issues
- Configure [monitoring and alerting](../index.md#monitoring) for the integration
