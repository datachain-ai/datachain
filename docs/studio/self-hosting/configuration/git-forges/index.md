# Git Forges Configuration

This section covers how to configure DataChain Studio to integrate with various Git hosting providers (forges) including GitHub, GitLab, and Bitbucket.

## Overview

DataChain Studio supports integration with multiple Git forges to enable:

- **Repository Access**: Connect to Git repositories for code and data
- **Authentication**: OAuth-based user authentication
- **Webhook Integration**: Automatic job triggering on Git events
- **Team Management**: Synchronize teams and permissions

## Supported Git Forges

- **[GitHub](github.md)** - GitHub.com and GitHub Enterprise Server
- **[GitLab](gitlab.md)** - GitLab.com and self-hosted GitLab instances
- **[Bitbucket](bitbucket.md)** - Bitbucket Cloud and Bitbucket Server

## General Configuration

All Git forge integrations share common configuration patterns:

### Basic Configuration Structure

```yaml
global:
  git:
    # GitHub configuration
    github:
      enabled: true
      appId: "your-app-id"
      privateKey: "your-private-key"
      webhookSecret: "your-webhook-secret"

    # GitLab configuration
    gitlab:
      enabled: true
      url: "https://gitlab.com"
      clientId: "your-client-id"
      clientSecret: "your-client-secret"
      webhookSecret: "your-webhook-secret"

    # Bitbucket configuration
    bitbucket:
      enabled: true
      clientId: "your-client-id"
      clientSecret: "your-client-secret"
      webhookSecret: "your-webhook-secret"
```

### Common Configuration Options

All Git forges support these common options:

```yaml
git:
  <forge-name>:
    enabled: true|false

    # Authentication settings
    clientId: "oauth-client-id"
    clientSecret: "oauth-client-secret"

    # Webhook configuration
    webhookSecret: "webhook-secret-key"
    webhookEvents:
      - push
      - pull_request
      - release

    # SSL/TLS settings
    ssl:
      verify: true
      caCertificate: |
        -----BEGIN CERTIFICATE-----
        ... custom CA certificate ...
        -----END CERTIFICATE-----

    # Rate limiting
    rateLimit:
      requestsPerHour: 5000
      burstSize: 100

    # Timeout settings
    timeout:
      connect: 30s
      read: 60s
      write: 30s
```

## Multi-Forge Configuration

DataChain Studio can be configured to work with multiple Git forges simultaneously:

```yaml
global:
  git:
    # Primary forge
    github:
      enabled: true
      appId: "123456"
      privateKey: |
        -----BEGIN RSA PRIVATE KEY-----
        ... GitHub App private key ...
        -----END RSA PRIVATE KEY-----

    # Secondary forge for internal repositories
    gitlab:
      enabled: true
      url: "https://gitlab.internal.company.com"
      clientId: "internal-gitlab-client-id"
      clientSecret: "internal-gitlab-secret"

    # Additional forge for specific teams
    bitbucket:
      enabled: true
      clientId: "bitbucket-client-id"
      clientSecret: "bitbucket-secret"
```

## Authentication Flow

### OAuth 2.0 Flow

All Git forges use OAuth 2.0 for authentication:

1. **User Authorization**: User authorizes DataChain Studio to access their Git forge account
2. **Code Exchange**: Studio exchanges authorization code for access token
3. **Token Storage**: Access tokens are securely stored and used for API calls
4. **Token Refresh**: Tokens are automatically refreshed when needed

### Configuration Requirements

Each forge requires specific OAuth application setup:

- **Redirect URIs**: Must include Studio's callback URLs
- **Scopes**: Appropriate permissions for repository and user access
- **Webhook URLs**: For receiving Git events

## Webhook Configuration

### Automatic Webhook Setup

DataChain Studio can automatically configure webhooks:

```yaml
git:
  <forge-name>:
    webhooks:
      autoSetup: true
      events:
        - push
        - pull_request
        - release

      # Custom webhook settings
      ssl:
        verify: true

      contentType: "application/json"
      secret: "webhook-secret-key"
```

### Manual Webhook Configuration

For manual webhook setup, configure each repository with:

- **Payload URL**: `https://studio.yourcompany.com/api/webhooks/<forge-name>`
- **Content Type**: `application/json`
- **Secret**: Your configured webhook secret
- **Events**: `push`, `pull_request`, `release`

## Security Configuration

### SSL/TLS Configuration

For self-hosted Git forges with custom certificates:

```yaml
git:
  gitlab:
    url: "https://gitlab.internal.company.com"
    ssl:
      verify: true
      caCertificate: |
        -----BEGIN CERTIFICATE-----
        ... your internal CA certificate ...
        -----END CERTIFICATE-----
```

### Access Control

Configure repository access patterns:

```yaml
git:
  <forge-name>:
    access:
      # Repository filtering
      repositories:
        allowed:
          - "org/allowed-repo"
          - "org/*-public"
        blocked:
          - "org/sensitive-repo"

      # User/organization filtering
      organizations:
        allowed:
          - "your-org"
          - "partner-org"
        blocked:
          - "external-org"
```

## Error Handling and Retry Logic

Configure resilient Git forge connections:

```yaml
git:
  <forge-name>:
    retry:
      enabled: true
      maxAttempts: 3
      initialDelay: 1s
      maxDelay: 30s
      exponentialBackoff: true

    circuitBreaker:
      enabled: true
      failureThreshold: 10
      recoveryTimeout: 60s
```

## Monitoring and Alerting

Monitor Git forge integrations:

```yaml
monitoring:
  gitForges:
    enabled: true

    healthChecks:
      enabled: true
      interval: 30s
      timeout: 10s

    metrics:
      - apiCalls
      - responseTime
      - errorRate
      - webhookDelivery

    alerts:
      - name: "Git Forge API Error Rate High"
        condition: "error_rate > 5%"
        duration: "5m"
        severity: "warning"
```

## Testing Configuration

### Connectivity Testing

Test Git forge connections:

```bash
# Test GitHub connection
curl -k https://studio.yourcompany.com/api/git/github/test

# Test GitLab connection
curl -k https://studio.yourcompany.com/api/git/gitlab/test

# Test webhook delivery
curl -X POST https://studio.yourcompany.com/api/webhooks/github \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: ping" \
  -d '{"zen": "Test webhook"}'
```

### Configuration Validation

Validate configuration before deployment:

```bash
# Validate Helm configuration
helm template datachain-studio ./chart \
  --values values.yaml \
  --dry-run

# Test OAuth flow
curl "https://github.com/login/oauth/authorize?client_id=YOUR_CLIENT_ID&redirect_uri=https://studio.yourcompany.com/auth/github/callback"
```

## Troubleshooting

### Common Issues

**OAuth authentication failures:**
- Verify client ID and secret
- Check redirect URI configuration
- Ensure proper scopes are granted

**Webhook delivery failures:**
- Verify webhook secret matches
- Check webhook URL accessibility
- Review webhook event configuration

**API rate limiting:**
- Monitor API usage
- Implement proper caching
- Configure rate limit settings

### Debug Commands

```bash
# Check Git forge connectivity
kubectl logs -f deployment/datachain-studio-backend -n datachain-studio | grep -i git

# Test OAuth flow
kubectl port-forward service/datachain-studio-frontend 8080:80 -n datachain-studio

# Verify webhook configuration
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- curl -I https://api.github.com
```

## Migration Between Forges

When migrating between Git forges:

1. **Export Configuration**: Back up existing Git forge settings
2. **Configure New Forge**: Set up authentication with new provider
3. **Update Repositories**: Migrate repository connections
4. **Test Integration**: Verify all functionality works
5. **Update Webhooks**: Reconfigure webhook endpoints
6. **Cleanup**: Remove old forge configuration

## Next Steps

Choose your Git forge for detailed configuration:

- **[GitHub Configuration](github.md)** - Set up GitHub.com or GitHub Enterprise
- **[GitLab Configuration](gitlab.md)** - Configure GitLab.com or self-hosted GitLab
- **[Bitbucket Configuration](bitbucket.md)** - Integrate with Bitbucket Cloud or Server

For additional configuration options:

- [SSL/TLS Configuration](../ssl-tls.md) for secure connections
- [CA Certificates](../ca-certificates.md) for custom certificate authorities
- [Troubleshooting Guide](../../troubleshooting/index.md) for common issues
