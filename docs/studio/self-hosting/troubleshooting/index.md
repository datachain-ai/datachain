# Troubleshooting

This section provides solutions to common issues encountered when running self-hosted DataChain Studio.

## Common Issues

- **[502 Errors](502-errors.md)** - Troubleshoot HTTP 502 Bad Gateway errors
- **[Support Bundle](support-bundle.md)** - Generate diagnostic information for support

## General Troubleshooting

### System Health Checks

#### Check Service Status

```bash
# Kubernetes deployments
kubectl get pods -n datachain-studio
kubectl get services -n datachain-studio
kubectl get ingress -n datachain-studio

# AMI deployments
sudo systemctl status datachain-studio
sudo docker ps
```

#### Check Resource Usage

```bash
# Kubernetes
kubectl top nodes
kubectl top pods -n datachain-studio

# AMI
htop
df -h
free -h
```

#### Check Logs

```bash
# Kubernetes - Application logs
kubectl logs -f deployment/datachain-studio-backend -n datachain-studio
kubectl logs -f deployment/datachain-studio-frontend -n datachain-studio
kubectl logs -f deployment/datachain-studio-worker -n datachain-studio

# AMI - System logs
sudo journalctl -u datachain-studio -f
sudo docker logs datachain-studio-backend
```

### Network Connectivity Issues

#### Test External Connectivity

```bash
# Test internet connectivity (if not air-gapped)
curl -I https://api.github.com
curl -I https://gitlab.com

# Test internal DNS resolution
nslookup studio.yourcompany.com
dig studio.yourcompany.com

# Test port connectivity
telnet studio.yourcompany.com 443
nc -zv studio.yourcompany.com 443
```

#### Test Service Connectivity

```bash
# Test database connectivity
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  python -c "import psycopg2; conn=psycopg2.connect('postgresql://user:pass@host:port/db'); print('DB OK')"

# Test Redis connectivity
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  redis-cli -h redis-host -p 6379 ping

# Test Git forge connectivity
curl -f https://studio.yourcompany.com/api/git/github/test
```

### Authentication Issues

#### OAuth Problems

```bash
# Check OAuth configuration
kubectl get configmap datachain-studio-config -n datachain-studio -o yaml | grep -A 10 oauth

# Test OAuth endpoints
curl -f https://studio.yourcompany.com/auth/github/login
curl -f https://studio.yourcompany.com/auth/gitlab/login

# Check OAuth callback URLs
curl -I https://studio.yourcompany.com/auth/github/callback
```

#### SSL/TLS Certificate Issues

```bash
# Check certificate validity
openssl s_client -connect studio.yourcompany.com:443 -servername studio.yourcompany.com

# Check certificate expiration
echo | openssl s_client -connect studio.yourcompany.com:443 2>/dev/null | openssl x509 -dates -noout

# Check certificate chain
openssl s_client -connect studio.yourcompany.com:443 -showcerts
```

### Database Issues

#### Connection Problems

```bash
# Test database connection
kubectl exec -it deployment/datachain-studio-postgres -n datachain-studio -- \
  psql -U studio -c "SELECT version();"

# Check database logs
kubectl logs -f deployment/datachain-studio-postgres -n datachain-studio

# Check connection pooling
kubectl exec -it deployment/datachain-studio-postgres -n datachain-studio -- \
  psql -U studio -c "SELECT * FROM pg_stat_activity;"
```

#### Performance Issues

```bash
# Check database performance
kubectl exec -it deployment/datachain-studio-postgres -n datachain-studio -- \
  psql -U studio -c "SELECT * FROM pg_stat_database WHERE datname='datachain_studio';"

# Check slow queries
kubectl exec -it deployment/datachain-studio-postgres -n datachain-studio -- \
  psql -U studio -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Check database size
kubectl exec -it deployment/datachain-studio-postgres -n datachain-studio -- \
  psql -U studio -c "SELECT pg_size_pretty(pg_database_size('datachain_studio'));"
```

### Storage Issues

#### Cloud Storage Connectivity

```bash
# Test S3 connectivity
aws s3 ls s3://your-studio-bucket/ --region your-region

# Test GCS connectivity
gsutil ls gs://your-studio-bucket/

# Test Azure Blob connectivity
az storage blob list --container-name your-container --account-name your-account
```

#### Persistent Volume Issues

```bash
# Check PV status
kubectl get pv,pvc -n datachain-studio

# Check storage class
kubectl get storageclass

# Check volume mount issues
kubectl describe pod POD_NAME -n datachain-studio | grep -A 10 -i volume
```

### Performance Troubleshooting

#### High CPU Usage

```bash
# Check CPU usage by pod
kubectl top pods -n datachain-studio --sort-by=cpu

# Check CPU usage inside container
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- top

# Profile application performance
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  python -m cProfile -o profile.stats your_script.py
```

#### High Memory Usage

```bash
# Check memory usage by pod
kubectl top pods -n datachain-studio --sort-by=memory

# Check memory usage inside container
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- free -h

# Check for memory leaks
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  ps aux --sort=-%mem | head -10
```

#### Slow Response Times

```bash
# Test response times
time curl -f https://studio.yourcompany.com/health
time curl -f https://studio.yourcompany.com/api/datasets

# Check application metrics
curl https://studio.yourcompany.com/metrics

# Check database query performance
kubectl exec -it deployment/datachain-studio-postgres -n datachain-studio -- \
  psql -U studio -c "EXPLAIN ANALYZE SELECT * FROM datasets LIMIT 10;"
```

### Configuration Issues

#### Invalid Configuration

```bash
# Validate Helm configuration
helm template datachain-studio ./chart --values values.yaml --dry-run

# Check for configuration errors
kubectl describe pod POD_NAME -n datachain-studio | grep -i error

# Validate YAML syntax
yamllint values.yaml
```

#### Missing Environment Variables

```bash
# Check environment variables
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- env | grep -i studio

# Check ConfigMap
kubectl get configmap datachain-studio-config -n datachain-studio -o yaml

# Check Secrets
kubectl get secrets -n datachain-studio
kubectl describe secret SECRET_NAME -n datachain-studio
```

## Diagnostic Commands

### Comprehensive Health Check

```bash
#!/bin/bash
# health-check.sh

echo "=== DataChain Studio Health Check ==="

echo "1. Pod Status:"
kubectl get pods -n datachain-studio

echo "2. Service Status:"
kubectl get services -n datachain-studio

echo "3. Ingress Status:"
kubectl get ingress -n datachain-studio

echo "4. Resource Usage:"
kubectl top pods -n datachain-studio

echo "5. Recent Events:"
kubectl get events -n datachain-studio --sort-by='.lastTimestamp' | tail -10

echo "6. Application Health:"
curl -s -o /dev/null -w "%{http_code}" https://studio.yourcompany.com/health

echo "7. Database Connectivity:"
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  python -c "from django.db import connection; connection.ensure_connection(); print('DB OK')" 2>/dev/null || echo "DB ERROR"
```

### Log Collection

```bash
#!/bin/bash
# collect-logs.sh

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR="datachain-studio-logs-${TIMESTAMP}"

mkdir -p ${LOG_DIR}

echo "Collecting DataChain Studio logs..."

# Pod logs
kubectl logs deployment/datachain-studio-frontend -n datachain-studio > ${LOG_DIR}/frontend.log
kubectl logs deployment/datachain-studio-backend -n datachain-studio > ${LOG_DIR}/backend.log
kubectl logs deployment/datachain-studio-worker -n datachain-studio > ${LOG_DIR}/worker.log

# System information
kubectl get pods -n datachain-studio -o wide > ${LOG_DIR}/pods.txt
kubectl describe pods -n datachain-studio > ${LOG_DIR}/pod-descriptions.txt
kubectl get events -n datachain-studio --sort-by='.lastTimestamp' > ${LOG_DIR}/events.txt

# Configuration
helm get values datachain-studio -n datachain-studio > ${LOG_DIR}/helm-values.yaml
kubectl get configmap datachain-studio-config -n datachain-studio -o yaml > ${LOG_DIR}/configmap.yaml

tar -czf ${LOG_DIR}.tar.gz ${LOG_DIR}
echo "Logs collected in ${LOG_DIR}.tar.gz"
```

## Getting Help

### Self-Help Resources

1. **Check Release Notes**: Review release notes for known issues
2. **Search Documentation**: Look for similar issues in documentation
3. **Community Forums**: Search community forums and discussions
4. **GitHub Issues**: Check the DataChain GitHub repository for similar issues

### Contacting Support

When contacting support, include:

1. **System Information**:
   - DataChain Studio version
   - Kubernetes version (if applicable)
   - Operating system and version
   - Hardware/resource specifications

2. **Problem Description**:
   - Detailed description of the issue
   - Steps to reproduce the problem
   - Expected vs actual behavior
   - Screenshots if applicable

3. **Diagnostic Information**:
   - Relevant log excerpts
   - Configuration files (sanitized)
   - Error messages
   - System resource usage

4. **Environment Details**:
   - Network configuration
   - Security settings
   - External integrations
   - Recent changes

### Support Bundle

Generate a comprehensive support bundle using our [support bundle tool](support-bundle.md).

## Prevention

### Monitoring and Alerting

Set up monitoring to catch issues early:

```yaml
# Example monitoring configuration
monitoring:
  enabled: true

  alerts:
    - name: "High Error Rate"
      condition: "error_rate > 5%"
      duration: "5m"
      severity: "warning"

    - name: "Service Down"
      condition: "up == 0"
      duration: "1m"
      severity: "critical"

    - name: "High Memory Usage"
      condition: "memory_usage > 80%"
      duration: "10m"
      severity: "warning"
```

### Regular Maintenance

1. **Update regularly**: Keep DataChain Studio and dependencies updated
2. **Monitor resources**: Watch CPU, memory, and storage usage trends
3. **Review logs**: Regularly review logs for warnings and errors
4. **Test backups**: Regularly test backup and restore procedures
5. **Security updates**: Apply security updates promptly

### Best Practices

1. **Use staging environment**: Test changes in staging before production
2. **Document configuration**: Keep configuration documented and version controlled
3. **Monitor performance**: Set up comprehensive monitoring and alerting
4. **Plan for scale**: Monitor usage trends and plan for capacity needs
5. **Security hygiene**: Regularly review and update security configurations

## Next Steps

For specific issues:

- **[502 Errors](502-errors.md)** - Detailed troubleshooting for HTTP 502 errors
- **[Support Bundle](support-bundle.md)** - Generate diagnostic information

For other topics:

- [Configuration Guide](../configuration/index.md) for configuration issues
- [Upgrading Guide](../upgrading/index.md) for upgrade-related problems
- [Installation Guide](../installation/index.md) for installation issues
