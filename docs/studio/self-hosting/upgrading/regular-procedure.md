# Regular Upgrade Procedure

This guide covers the standard upgrade procedure for DataChain Studio deployments with internet access.

## Prerequisites

- Administrative access to your Kubernetes cluster or AMI instance
- Internet connectivity for downloading new container images
- Backup of current configuration and data
- Scheduled maintenance window

## Pre-upgrade Preparation

### 1. Review Release Notes

Before upgrading, review the release notes for:
- Breaking changes that may affect your deployment
- New configuration options
- Deprecated features
- Security updates

### 2. Create Backups

#### Database Backup
```bash
# For Kubernetes deployments
kubectl exec -it deployment/datachain-studio-postgres -n datachain-studio -- \
  pg_dump -U studio datachain_studio > backup-$(date +%Y%m%d-%H%M%S).sql

# For AMI deployments
sudo -u postgres pg_dump datachain_studio > backup-$(date +%Y%m%d-%H%M%S).sql
```

#### Configuration Backup
```bash
# Kubernetes: Backup Helm values
helm get values datachain-studio -n datachain-studio > values-backup-$(date +%Y%m%d).yaml

# AMI: Backup configuration file
sudo cp /opt/datachain-studio/config.yml /opt/datachain-studio/config-backup-$(date +%Y%m%d).yml
```

#### Storage Backup (if applicable)
```bash
# Backup persistent volumes
kubectl get pv,pvc -n datachain-studio -o yaml > pv-backup-$(date +%Y%m%d).yaml

# For cloud storage, create snapshots using provider tools
```

### 3. Verify System Health

Before starting the upgrade, ensure the system is healthy:

```bash
# Check pod status
kubectl get pods -n datachain-studio

# Check resource usage
kubectl top nodes
kubectl top pods -n datachain-studio

# Check service availability
curl -f https://studio.yourcompany.com/health
```

## Kubernetes/Helm Upgrade

### 1. Update Helm Repository

```bash
# Update the DataChain Helm repository
helm repo update datachain

# Check available versions
helm search repo datachain/studio --versions
```

### 2. Review Configuration Changes

Compare your current configuration with the new version:

```bash
# Get current values
helm get values datachain-studio -n datachain-studio > current-values.yaml

# Show default values for new version
helm show values datachain/studio --version NEW_VERSION > new-default-values.yaml

# Compare configurations
diff current-values.yaml new-default-values.yaml
```

### 3. Plan the Upgrade

```bash
# Dry run to see what will change
helm upgrade datachain-studio datachain/studio \
  --namespace datachain-studio \
  --values values.yaml \
  --version NEW_VERSION \
  --dry-run --debug
```

### 4. Perform the Upgrade

```bash
# Upgrade DataChain Studio
helm upgrade datachain-studio datachain/studio \
  --namespace datachain-studio \
  --values values.yaml \
  --version NEW_VERSION \
  --wait \
  --timeout 10m

# Monitor the upgrade progress
kubectl get pods -n datachain-studio -w
```

### 5. Verify Upgrade

```bash
# Check upgrade status
helm status datachain-studio -n datachain-studio

# Verify all pods are running
kubectl get pods -n datachain-studio

# Check services
kubectl get services -n datachain-studio

# Test application health
curl -f https://studio.yourcompany.com/health
```

## AMI Upgrade

### 1. Connect to the Instance

```bash
# SSH to your AMI instance
ssh -i your-key.pem ubuntu@your-instance-ip
```

### 2. Update System Packages

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Docker if needed
sudo apt install docker.io
```

### 3. Pull New Images

```bash
# Pull new DataChain Studio images
sudo docker pull datachain/studio-frontend:NEW_VERSION
sudo docker pull datachain/studio-backend:NEW_VERSION
sudo docker pull datachain/studio-worker:NEW_VERSION

# List current images
sudo docker images | grep datachain
```

### 4. Update Configuration

```bash
# Navigate to configuration directory
cd /opt/datachain-studio

# Backup current configuration
sudo cp config.yml config-backup-$(date +%Y%m%d).yml

# Update configuration if needed (based on release notes)
sudo nano config.yml
```

### 5. Stop Current Services

```bash
# Stop DataChain Studio services
sudo systemctl stop datachain-studio

# Verify services are stopped
sudo systemctl status datachain-studio
```

### 6. Update and Start Services

```bash
# Update service configuration if needed
sudo systemctl daemon-reload

# Start services with new version
sudo systemctl start datachain-studio

# Enable auto-start
sudo systemctl enable datachain-studio

# Check service status
sudo systemctl status datachain-studio
```

### 7. Verify Upgrade

```bash
# Check container status
sudo docker ps

# Check logs
sudo journalctl -u datachain-studio -f

# Test application
curl -f https://studio.yourcompany.com/health
```

## Database Migrations

### Automatic Migrations

DataChain Studio typically handles database migrations automatically during startup:

```bash
# Monitor migration logs
kubectl logs -f deployment/datachain-studio-backend -n datachain-studio | grep -i migration

# For AMI deployments
sudo journalctl -u datachain-studio -f | grep -i migration
```

### Manual Migration (if required)

If manual intervention is needed:

```bash
# Run migration job manually (Kubernetes)
kubectl create job manual-migration \
  --from=deployment/datachain-studio-backend \
  -n datachain-studio

# Monitor migration job
kubectl logs -f job/manual-migration -n datachain-studio

# For AMI, run migration command
cd /opt/datachain-studio
sudo -u datachain python manage.py migrate
```

## Post-Upgrade Validation

### 1. System Health Checks

```bash
# Check all services are running
kubectl get pods -n datachain-studio
sudo systemctl status datachain-studio  # For AMI

# Verify resource usage is normal
kubectl top pods -n datachain-studio
top  # For AMI

# Test external connectivity
curl -f https://studio.yourcompany.com/health
```

### 2. Functional Testing

Test critical functionality:

#### Authentication
```bash
# Test login page
curl -f https://studio.yourcompany.com/login

# Test OAuth endpoints
curl -f https://studio.yourcompany.com/auth/github/login
```

#### API Endpoints
```bash
# Test API availability
curl -H "Authorization: Bearer $TOKEN" \
  https://studio.yourcompany.com/api/datasets

# Test version endpoint
curl https://studio.yourcompany.com/api/version
```

#### Database Connectivity
```bash
# Test database connection
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  python -c "import django; django.setup(); from django.db import connection; print('DB OK' if connection.ensure_connection() is None else 'DB Error')"
```

#### Git Integration
```bash
# Test Git forge connectivity
curl -f https://studio.yourcompany.com/api/git/github/test
curl -f https://studio.yourcompany.com/api/git/gitlab/test
```

### 3. Performance Validation

Monitor performance after upgrade:

```bash
# Check response times
time curl -f https://studio.yourcompany.com/health

# Monitor resource usage
kubectl top pods -n datachain-studio
htop  # For AMI

# Check application metrics (if monitoring is enabled)
curl https://studio.yourcompany.com/metrics
```

## Rollback Procedure

If issues are encountered during or after upgrade:

### Kubernetes Rollback

```bash
# List release history
helm history datachain-studio -n datachain-studio

# Rollback to previous version
helm rollback datachain-studio -n datachain-studio

# Or rollback to specific revision
helm rollback datachain-studio REVISION_NUMBER -n datachain-studio

# Verify rollback
helm status datachain-studio -n datachain-studio
```

### AMI Rollback

```bash
# Stop current services
sudo systemctl stop datachain-studio

# Restore configuration backup
sudo cp /opt/datachain-studio/config-backup-DATE.yml /opt/datachain-studio/config.yml

# Pull previous version images
sudo docker pull datachain/studio-frontend:PREVIOUS_VERSION
sudo docker pull datachain/studio-backend:PREVIOUS_VERSION

# Update service to use previous version
# (Edit systemd service files or docker-compose as needed)

# Restore database if needed
sudo -u postgres psql datachain_studio < backup-DATE.sql

# Start services
sudo systemctl start datachain-studio
```

## Troubleshooting Common Issues

### Upgrade Hangs or Fails

**Container image pull failures:**
```bash
# Check image availability
docker pull datachain/studio-backend:NEW_VERSION

# Check registry connectivity
kubectl describe pod POD_NAME -n datachain-studio
```

**Database migration failures:**
```bash
# Check database connectivity
kubectl exec -it deployment/datachain-studio-postgres -n datachain-studio -- \
  psql -U studio -c "SELECT version();"

# Check migration logs
kubectl logs deployment/datachain-studio-backend -n datachain-studio | grep -i migration

# Manually run problematic migrations
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  python manage.py migrate --verbosity=2
```

**Resource constraints:**
```bash
# Check node resources
kubectl describe nodes

# Check pod resource requests/limits
kubectl describe pod POD_NAME -n datachain-studio

# Scale down other services temporarily if needed
kubectl scale deployment OTHER_DEPLOYMENT --replicas=0 -n datachain-studio
```

### Post-Upgrade Issues

**Service unavailable:**
```bash
# Check pod status
kubectl get pods -n datachain-studio

# Check service endpoints
kubectl get endpoints -n datachain-studio

# Check ingress configuration
kubectl describe ingress -n datachain-studio
```

**Performance degradation:**
```bash
# Check resource usage
kubectl top pods -n datachain-studio

# Review application logs for errors
kubectl logs -f deployment/datachain-studio-backend -n datachain-studio

# Check database performance
kubectl exec -it deployment/datachain-studio-postgres -n datachain-studio -- \
  psql -U studio -c "SELECT * FROM pg_stat_activity;"
```

## Best Practices

### Before Upgrade
1. **Test in staging** environment first
2. **Schedule maintenance windows** during low usage
3. **Communicate** with users about planned downtime
4. **Document** current configuration and customizations

### During Upgrade
1. **Monitor closely** throughout the process
2. **Have rollback plan ready** and tested
3. **Keep logs** of all commands and outputs
4. **Stay calm** and follow procedures methodically

### After Upgrade
1. **Validate thoroughly** before declaring success
2. **Monitor performance** for several hours/days
3. **Update documentation** with any changes
4. **Clean up** old images and backups after verification

## Next Steps

- Review [configuration changes](../configuration/index.md) that may be needed
- Update [monitoring and alerting](../configuration/index.md#monitoring) if applicable
- Check [troubleshooting guide](../troubleshooting/index.md) if issues occur
- Plan for [next upgrade cycle](index.md) based on release schedule

## Support

If you encounter issues during the upgrade:

1. Check the [troubleshooting guide](../troubleshooting/index.md)
2. Review application logs for error messages
3. Consult the release notes for known issues
4. Contact DataChain support with specific error details
