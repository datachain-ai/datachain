# Configuration

This section covers how to configure your self-hosted DataChain Studio instance for optimal performance, security, and integration with your infrastructure.

## Overview

DataChain Studio configuration involves several key areas:

- **[SSL/TLS Configuration](ssl-tls.md)** - Set up secure HTTPS connections
- **[CA Certificates](ca-certificates.md)** - Configure custom certificate authorities
- **[Git Forges](git-forges/index.md)** - Integrate with GitHub, GitLab, and Bitbucket

## Basic Configuration

### Environment Variables

DataChain Studio can be configured using environment variables:

```yaml
global:
  envVars:
    # Basic settings
    DATACHAIN_STUDIO_URL: "https://studio.yourcompany.com"
    DATACHAIN_STUDIO_SECRET_KEY: "your-secret-key"

    # Database settings
    DATABASE_URL: "postgresql://user:pass@host:5432/datachain_studio"
    REDIS_URL: "redis://host:6379"

    # Storage settings
    STORAGE_TYPE: "s3"
    S3_BUCKET: "your-studio-bucket"
    S3_REGION: "us-east-1"

    # Git integration
    GITHUB_APP_ID: "your-github-app-id"
    GITLAB_CLIENT_ID: "your-gitlab-client-id"
```

### Configuration File

For more complex configurations, use a YAML configuration file:

```yaml
# values.yaml
global:
  domain: studio.yourcompany.com

  # Security settings
  security:
    secretKey: "your-long-random-secret-key"
    sessionTimeout: 3600
    csrfProtection: true

  # Feature flags
  features:
    webhooks: true
    apiAccess: true
    teamCollaboration: true
    ssoIntegration: true

# Database configuration
database:
  type: postgresql
  host: postgres.yourcompany.com
  port: 5432
  name: datachain_studio
  user: studio_user
  password: secure-password
  sslMode: require

  # Connection pooling
  pool:
    minConnections: 5
    maxConnections: 20

# Cache configuration
cache:
  type: redis
  host: redis.yourcompany.com
  port: 6379
  password: redis-password
  database: 0

  # TTL settings
  ttl:
    sessions: 3600
    apiCache: 300
    dataCache: 1800

# Storage configuration
storage:
  type: s3
  config:
    bucket: datachain-studio-storage
    region: us-east-1
    accessKey: your-access-key
    secretKey: your-secret-key
    endpoint: s3.amazonaws.com

  # Alternative: Google Cloud Storage
  # type: gcs
  # config:
  #   bucket: datachain-studio-storage
  #   projectId: your-project-id
  #   keyFile: /path/to/service-account.json

# Logging configuration
logging:
  level: INFO
  format: json
  outputs:
    - console
    - file

  # Log rotation
  rotation:
    maxSize: 100MB
    maxAge: 30
    maxBackups: 10
```

## Advanced Configuration

### Performance Tuning

```yaml
# Performance settings
performance:
  # Worker processes
  workers:
    frontend: 4
    backend: 8
    jobProcessor: 2

  # Memory limits
  memory:
    frontend: "1Gi"
    backend: "2Gi"
    jobProcessor: "4Gi"

  # CPU limits
  cpu:
    frontend: "500m"
    backend: "1000m"
    jobProcessor: "2000m"

  # Caching
  cache:
    enabled: true
    size: "512Mi"
    evictionPolicy: "lru"
```

### Security Configuration {#security}

```yaml
# Security settings
security:
  # Authentication
  auth:
    methods:
      - local
      - oauth
      - saml

    # Password policy
    passwordPolicy:
      minLength: 8
      requireUppercase: true
      requireLowercase: true
      requireNumbers: true
      requireSpecialChars: true

    # Session management
    sessions:
      timeout: 3600
      renewalThreshold: 300
      maxConcurrent: 5

  # Network security
  network:
    allowedIPs:
      - "10.0.0.0/8"
      - "192.168.0.0/16"

    rateLimiting:
      enabled: true
      requestsPerMinute: 100
      burstSize: 20

  # Data encryption
  encryption:
    atRest:
      enabled: true
      algorithm: "AES-256-GCM"

    inTransit:
      enabled: true
      minTlsVersion: "1.2"
```

### Integration Configuration

```yaml
# External integrations
integrations:
  # Git forges
  git:
    github:
      enabled: true
      appId: "123456"
      privateKeyPath: "/etc/ssl/private/github.pem"
      webhookSecret: "github-webhook-secret"

    gitlab:
      enabled: true
      url: "https://gitlab.yourcompany.com"
      clientId: "gitlab-client-id"
      clientSecret: "gitlab-client-secret"
      webhookSecret: "gitlab-webhook-secret"

    bitbucket:
      enabled: true
      clientId: "bitbucket-client-id"
      clientSecret: "bitbucket-client-secret"

  # Monitoring
  monitoring:
    prometheus:
      enabled: true
      endpoint: "/metrics"
      port: 9090

    grafana:
      enabled: true
      url: "https://grafana.yourcompany.com"

    alerts:
      slack:
        enabled: true
        webhookUrl: "https://hooks.slack.com/..."
        channel: "#datachain-alerts"

      email:
        enabled: true
        smtpHost: "smtp.yourcompany.com"
        smtpPort: 587
        from: "datachain-studio@yourcompany.com"
```

### Backup Configuration

```yaml
# Backup settings
backup:
  enabled: true

  # Database backups
  database:
    enabled: true
    schedule: "0 2 * * *"  # Daily at 2 AM
    retention: 30  # days
    compression: true

    destination:
      type: s3
      bucket: datachain-studio-backups
      path: database/

  # Storage backups
  storage:
    enabled: true
    schedule: "0 3 * * 0"  # Weekly on Sunday at 3 AM
    retention: 12  # weeks

    destination:
      type: s3
      bucket: datachain-studio-backups
      path: storage/
```

## Monitoring Configuration {#monitoring}

### Metrics and Alerting

```yaml
# Monitoring configuration
monitoring:
  # Metrics collection
  metrics:
    enabled: true
    interval: 30s

    collectors:
      - system
      - application
      - database
      - cache
      - storage

  # Health checks
  healthChecks:
    enabled: true
    interval: 10s
    timeout: 5s

    endpoints:
      - /health/live
      - /health/ready
      - /health/database
      - /health/cache

  # Alerting rules
  alerts:
    rules:
      - name: "High CPU Usage"
        condition: "cpu_usage > 80"
        duration: "5m"
        severity: "warning"

      - name: "Database Connection Failed"
        condition: "database_health == 0"
        duration: "1m"
        severity: "critical"

      - name: "Storage Full"
        condition: "storage_usage > 90"
        duration: "5m"
        severity: "critical"
```

## Validation

### Configuration Validation

Validate your configuration before deployment:

```bash
# For Helm deployments
helm template datachain-studio ./chart \
  --values values.yaml \
  --dry-run

# For direct deployments
datachain-studio validate-config config.yaml
```

### Health Checks

Monitor your configuration post-deployment:

```bash
# Check service health
curl https://studio.yourcompany.com/health

# Check database connectivity
curl https://studio.yourcompany.com/health/database

# Check storage connectivity
curl https://studio.yourcompany.com/health/storage
```

## Troubleshooting

### Common Configuration Issues

**Database connection failures:**
- Verify connection string format
- Check network connectivity
- Confirm credentials and permissions

**SSL/TLS certificate issues:**
- Validate certificate chain
- Check certificate expiration
- Verify domain name matches

**Storage access problems:**
- Confirm bucket permissions
- Check access key validity
- Verify network connectivity

### Configuration Testing

```yaml
# Test configuration
test:
  enabled: true

  # Unit tests
  unit:
    database: true
    cache: true
    storage: true
    auth: true

  # Integration tests
  integration:
    gitForges: true
    webhooks: true
    api: true

  # Load tests
  load:
    enabled: false
    users: 100
    duration: "10m"
```

## Next Steps

- Configure [SSL/TLS certificates](ssl-tls.md)
- Set up [Git forge integrations](git-forges/index.md)
- Review [upgrading procedures](../upgrading/index.md)
- Check [troubleshooting guides](../troubleshooting/index.md)
