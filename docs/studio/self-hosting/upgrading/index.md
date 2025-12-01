# Upgrading DataChain Studio

This section covers how to upgrade your self-hosted DataChain Studio instance to newer versions safely and efficiently.

## Overview

DataChain Studio upgrades involve updating:

- **Application Code**: Core DataChain Studio services
- **Database Schema**: Database migrations and updates
- **Configuration**: New configuration options and changes
- **Dependencies**: Updated system dependencies and containers

## Upgrade Methods

- **[Regular Procedure](regular-procedure.md)** - Standard upgrade process for most deployments
- **[Airgap Procedure](airgap-procedure.md)** - Upgrade process for air-gapped environments

## Before You Begin

### Prerequisites

- **Backup**: Complete backup of data and configuration
- **Maintenance Window**: Scheduled downtime for the upgrade
- **Access**: Administrative access to your deployment
- **Resources**: Sufficient system resources for the upgrade

### Pre-upgrade Checklist

- [ ] Review release notes for breaking changes
- [ ] Backup database and configuration files
- [ ] Test upgrade in staging environment
- [ ] Verify system requirements are met
- [ ] Plan rollback strategy if needed
- [ ] Notify users of scheduled maintenance

## Upgrade Planning

### Version Compatibility

Check version compatibility before upgrading:

- **Supported Upgrades**: Direct upgrades from previous major version
- **Skip Versions**: Intermediate versions may be required for large jumps
- **Breaking Changes**: Review changelog for breaking changes

### System Requirements

Verify system requirements for the target version:

```yaml
# Minimum requirements may change between versions
systemRequirements:
  kubernetes: "1.21+"
  helm: "3.7+"
  nodes:
    minimum: 2
    recommended: 3
  resources:
    cpu: "4 cores"
    memory: "16GB RAM"
    storage: "100GB"
```

### Backup Strategy {#backup}

Always backup before upgrading:

#### Database Backup
```bash
# PostgreSQL backup
kubectl exec -it postgres-pod -n datachain-studio -- \
  pg_dump -U studio datachain_studio > backup-$(date +%Y%m%d).sql

# Or using helm backup job
helm install backup-job datachain/backup \
  --namespace datachain-studio \
  --set backup.type=database
```

#### Configuration Backup
```bash
# Backup Helm values
helm get values datachain-studio -n datachain-studio > values-backup.yaml

# Backup Kubernetes resources
kubectl get all -n datachain-studio -o yaml > k8s-resources-backup.yaml

# Backup secrets
kubectl get secrets -n datachain-studio -o yaml > secrets-backup.yaml
```

#### Storage Backup
```bash
# Backup persistent volumes (depends on storage provider)
kubectl get pv,pvc -n datachain-studio

# For cloud storage, use provider tools:
# AWS: aws s3 sync s3://studio-bucket s3://studio-backup-bucket
# GCS: gsutil -m cp -r gs://studio-bucket gs://studio-backup-bucket
```

## Upgrade Process Overview

### Standard Upgrade Flow

1. **Preparation**
   - Review release notes
   - Plan maintenance window
   - Create backups

2. **Pre-upgrade Tasks**
   - Update Helm repositories
   - Validate configuration
   - Check resource availability

3. **Upgrade Execution**
   - Apply configuration changes
   - Perform database migrations
   - Update application containers

4. **Post-upgrade Tasks**
   - Verify system health
   - Test functionality
   - Monitor performance

5. **Validation**
   - Run integration tests
   - Verify data integrity
   - Confirm user access

## Version Management

### Semantic Versioning

DataChain Studio follows semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes requiring manual intervention
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, security updates

### Release Channels

Choose appropriate release channel:

- **Stable**: Production-ready releases
- **Beta**: Pre-release versions for testing
- **Alpha**: Early development versions

```yaml
# Configure release channel in values.yaml
global:
  image:
    tag: "1.2.3"  # Specific version
    # tag: "stable"   # Latest stable
    # tag: "beta"     # Latest beta
```

## Rollback Strategy

### Automated Rollback

Prepare for potential rollback:

```yaml
# Enable automated rollback on failure
upgrade:
  rollback:
    enabled: true
    onFailure: true
    timeout: 600s

  # Health checks for validation
  healthChecks:
    enabled: true
    initialDelay: 30s
    timeout: 10s
    failureThreshold: 3
```

### Manual Rollback

Steps for manual rollback:

```bash
# Rollback using Helm
helm rollback datachain-studio -n datachain-studio

# Restore database from backup
kubectl exec -it postgres-pod -n datachain-studio -- \
  psql -U studio datachain_studio < backup-20240115.sql

# Restore configuration
helm upgrade datachain-studio datachain/studio \
  --namespace datachain-studio \
  --values values-backup.yaml
```

## Monitoring During Upgrades

### Health Monitoring

Monitor system health during upgrades:

```yaml
monitoring:
  upgrade:
    enabled: true

    # Metrics to monitor
    metrics:
      - cpu_usage
      - memory_usage
      - database_connections
      - response_time
      - error_rate

    # Alert thresholds
    alerts:
      - name: "High Error Rate During Upgrade"
        condition: "error_rate > 5%"
        duration: "2m"
        action: "pause_upgrade"
```

### Log Monitoring

Key logs to monitor:

```bash
# Application logs
kubectl logs -f deployment/datachain-studio-backend -n datachain-studio

# Database migration logs
kubectl logs -f job/migration-job -n datachain-studio

# Ingress logs
kubectl logs -f deployment/nginx-ingress-controller -n ingress-nginx
```

## Testing After Upgrade

### Automated Testing

Run automated tests after upgrade:

```bash
# Health check tests
curl -f https://studio.yourcompany.com/health

# API functionality tests
curl -H "Authorization: Bearer $TOKEN" \
  https://studio.yourcompany.com/api/datasets

# Database connectivity tests
kubectl exec -it postgres-pod -n datachain-studio -- \
  psql -U studio -c "SELECT version();"
```

### Manual Testing

Perform manual testing:

- [ ] User login functionality
- [ ] Dataset creation and access
- [ ] Job submission and execution
- [ ] Git integration functionality
- [ ] Webhook delivery
- [ ] API endpoints
- [ ] User interface responsiveness

## Upgrade Troubleshooting

### Common Issues

**Database migration failures:**
- Check database connectivity
- Verify migration scripts
- Review database logs
- Ensure sufficient disk space

**Container startup failures:**
- Check resource availability
- Verify image availability
- Review configuration changes
- Check dependency services

**Configuration conflicts:**
- Compare old vs new configuration
- Review breaking changes in release notes
- Validate YAML syntax
- Check required vs optional fields

### Recovery Procedures

**Service degradation:**
1. Check resource utilization
2. Review application logs
3. Verify configuration
4. Consider scaling resources
5. Rollback if necessary

**Data corruption:**
1. Stop write operations
2. Assess corruption extent
3. Restore from backup
4. Verify data integrity
5. Resume operations

## Best Practices

### Upgrade Preparation

1. **Test in Staging**: Always test upgrades in staging first
2. **Read Release Notes**: Review all changes and breaking changes
3. **Plan Downtime**: Schedule appropriate maintenance windows
4. **Prepare Rollback**: Have rollback plan ready

### During Upgrade

1. **Monitor Closely**: Watch logs and metrics during upgrade
2. **Validate Each Step**: Confirm each step completes successfully
3. **Document Issues**: Record any problems encountered
4. **Stay Calm**: Follow procedures methodically

### Post-Upgrade

1. **Thorough Testing**: Test all critical functionality
2. **Performance Monitoring**: Watch for performance regressions
3. **User Communication**: Notify users when service is restored
4. **Document Lessons**: Record lessons learned for next time

## Automation

### CI/CD Integration

Automate upgrades using CI/CD:

```yaml
# GitLab CI example
upgrade-staging:
  stage: upgrade
  script:
    - helm repo update
    - helm upgrade datachain-studio datachain/studio
      --namespace datachain-studio-staging
      --values values-staging.yaml
  only:
    - main

upgrade-production:
  stage: upgrade
  script:
    - helm upgrade datachain-studio datachain/studio
      --namespace datachain-studio
      --values values-production.yaml
  when: manual
  only:
    - main
```

### Automated Validation

```yaml
# Automated post-upgrade validation
validation:
  enabled: true

  tests:
    - name: "Health Check"
      url: "https://studio.yourcompany.com/health"
      expected_status: 200

    - name: "API Test"
      url: "https://studio.yourcompany.com/api/version"
      expected_status: 200
      timeout: 30s

    - name: "Database Test"
      type: "sql"
      query: "SELECT COUNT(*) FROM datasets"
      expected_result: "> 0"
```

## Next Steps

Choose your upgrade method:

- **[Regular Procedure](regular-procedure.md)** - For connected environments
- **[Airgap Procedure](airgap-procedure.md)** - For air-gapped environments

For additional information:

- [Configuration Guide](../configuration/index.md) for post-upgrade configuration
- [Troubleshooting Guide](../troubleshooting/index.md) for resolving issues
- [Installation Guide](../installation/index.md) for fresh installations
