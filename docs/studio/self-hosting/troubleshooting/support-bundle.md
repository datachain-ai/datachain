# Support Bundle Generation

When experiencing issues with your self-hosted DataChain Studio instance, a support bundle provides comprehensive diagnostic information to help identify and resolve problems quickly.

## Overview

A support bundle collects:
- System configuration and status
- Application logs and metrics
- Resource usage information
- Network connectivity details
- Database and storage status
- Error messages and stack traces

## Automated Support Bundle

### Kubernetes Deployment

For Kubernetes deployments, use the automated support bundle script:

```bash
#!/bin/bash
# generate-support-bundle.sh

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BUNDLE_DIR="datachain-studio-support-${TIMESTAMP}"
NAMESPACE="datachain-studio"

echo "Generating DataChain Studio support bundle..."
mkdir -p ${BUNDLE_DIR}

# System Information
echo "Collecting system information..."
kubectl version --client > ${BUNDLE_DIR}/kubectl-version.txt
helm version > ${BUNDLE_DIR}/helm-version.txt
kubectl get nodes -o wide > ${BUNDLE_DIR}/nodes.txt
kubectl describe nodes > ${BUNDLE_DIR}/nodes-detailed.txt

# Cluster Resources
echo "Collecting cluster resources..."
kubectl get all -n ${NAMESPACE} -o wide > ${BUNDLE_DIR}/all-resources.txt
kubectl describe pods -n ${NAMESPACE} > ${BUNDLE_DIR}/pods-detailed.txt
kubectl get events -n ${NAMESPACE} --sort-by='.lastTimestamp' > ${BUNDLE_DIR}/events.txt

# Configuration
echo "Collecting configuration..."
helm get values datachain-studio -n ${NAMESPACE} > ${BUNDLE_DIR}/helm-values.yaml
kubectl get configmap datachain-studio-config -n ${NAMESPACE} -o yaml > ${BUNDLE_DIR}/configmap.yaml
kubectl get secrets -n ${NAMESPACE} -o name > ${BUNDLE_DIR}/secrets-list.txt

# Logs
echo "Collecting logs..."
for pod in $(kubectl get pods -n ${NAMESPACE} -o name); do
    pod_name=$(basename ${pod})
    kubectl logs ${pod} -n ${NAMESPACE} --previous > ${BUNDLE_DIR}/logs-${pod_name}-previous.log 2>/dev/null || true
    kubectl logs ${pod} -n ${NAMESPACE} > ${BUNDLE_DIR}/logs-${pod_name}.log 2>/dev/null || true
done

# Resource Usage
echo "Collecting resource usage..."
kubectl top nodes > ${BUNDLE_DIR}/resource-usage-nodes.txt 2>/dev/null || echo "Metrics server not available" > ${BUNDLE_DIR}/resource-usage-nodes.txt
kubectl top pods -n ${NAMESPACE} > ${BUNDLE_DIR}/resource-usage-pods.txt 2>/dev/null || echo "Metrics server not available" > ${BUNDLE_DIR}/resource-usage-pods.txt

# Ingress and Networking
echo "Collecting networking information..."
kubectl get ingress -n ${NAMESPACE} -o yaml > ${BUNDLE_DIR}/ingress.yaml
kubectl get services -n ${NAMESPACE} -o yaml > ${BUNDLE_DIR}/services.yaml
kubectl get endpoints -n ${NAMESPACE} > ${BUNDLE_DIR}/endpoints.txt

# Storage
echo "Collecting storage information..."
kubectl get pv,pvc -n ${NAMESPACE} -o yaml > ${BUNDLE_DIR}/storage.yaml
kubectl get storageclass -o yaml > ${BUNDLE_DIR}/storage-classes.yaml

# Health Checks
echo "Running health checks..."
kubectl exec -it deployment/datachain-studio-backend -n ${NAMESPACE} -- curl -s http://localhost:8000/health > ${BUNDLE_DIR}/health-check.json 2>/dev/null || echo "Health check failed" > ${BUNDLE_DIR}/health-check.json

# Database Status
echo "Collecting database information..."
kubectl exec -it deployment/datachain-studio-postgres -n ${NAMESPACE} -- psql -U studio -c "SELECT version();" > ${BUNDLE_DIR}/database-version.txt 2>/dev/null || echo "Database not accessible" > ${BUNDLE_DIR}/database-version.txt
kubectl exec -it deployment/datachain-studio-postgres -n ${NAMESPACE} -- psql -U studio -c "SELECT * FROM pg_stat_database WHERE datname='datachain_studio';" > ${BUNDLE_DIR}/database-stats.txt 2>/dev/null || true

# Package the bundle
echo "Creating support bundle archive..."
tar -czf ${BUNDLE_DIR}.tar.gz ${BUNDLE_DIR}
rm -rf ${BUNDLE_DIR}

echo "Support bundle created: ${BUNDLE_DIR}.tar.gz"
echo "Please provide this file when contacting support."
```

### AMI Deployment

For AMI deployments, use this script:

```bash
#!/bin/bash
# generate-ami-support-bundle.sh

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BUNDLE_DIR="datachain-studio-ami-support-${TIMESTAMP}"

echo "Generating DataChain Studio AMI support bundle..."
mkdir -p ${BUNDLE_DIR}

# System Information
echo "Collecting system information..."
uname -a > ${BUNDLE_DIR}/system-info.txt
lsb_release -a > ${BUNDLE_DIR}/os-version.txt 2>/dev/null || cat /etc/os-release > ${BUNDLE_DIR}/os-version.txt
docker version > ${BUNDLE_DIR}/docker-version.txt
free -h > ${BUNDLE_DIR}/memory-info.txt
df -h > ${BUNDLE_DIR}/disk-info.txt
lscpu > ${BUNDLE_DIR}/cpu-info.txt

# Service Status
echo "Collecting service status..."
sudo systemctl status datachain-studio > ${BUNDLE_DIR}/service-status.txt
sudo systemctl status docker > ${BUNDLE_DIR}/docker-status.txt
sudo docker ps -a > ${BUNDLE_DIR}/containers.txt
sudo docker images > ${BUNDLE_DIR}/images.txt

# Configuration
echo "Collecting configuration..."
sudo cp /opt/datachain-studio/config.yml ${BUNDLE_DIR}/config.yml 2>/dev/null || echo "Config file not found" > ${BUNDLE_DIR}/config.yml
sudo systemctl cat datachain-studio > ${BUNDLE_DIR}/systemd-service.txt

# Logs
echo "Collecting logs..."
sudo journalctl -u datachain-studio --no-pager > ${BUNDLE_DIR}/service-logs.txt
sudo journalctl -u docker --no-pager > ${BUNDLE_DIR}/docker-logs.txt
sudo docker logs datachain-studio-backend > ${BUNDLE_DIR}/backend-logs.txt 2>/dev/null || true
sudo docker logs datachain-studio-frontend > ${BUNDLE_DIR}/frontend-logs.txt 2>/dev/null || true
sudo docker logs datachain-studio-worker > ${BUNDLE_DIR}/worker-logs.txt 2>/dev/null || true

# System Logs
tail -1000 /var/log/syslog > ${BUNDLE_DIR}/syslog.txt 2>/dev/null || true
tail -1000 /var/log/messages > ${BUNDLE_DIR}/messages.txt 2>/dev/null || true

# Network Information
echo "Collecting network information..."
ip addr show > ${BUNDLE_DIR}/network-interfaces.txt
netstat -tulpn > ${BUNDLE_DIR}/network-ports.txt
ss -tulpn > ${BUNDLE_DIR}/socket-stats.txt

# Health Checks
echo "Running health checks..."
curl -s http://localhost:8000/health > ${BUNDLE_DIR}/health-check.json 2>/dev/null || echo "Health check failed" > ${BUNDLE_DIR}/health-check.json
curl -s -I https://studio.yourcompany.com/health > ${BUNDLE_DIR}/external-health-check.txt 2>/dev/null || echo "External health check failed" > ${BUNDLE_DIR}/external-health-check.txt

# Database Information (if accessible)
echo "Collecting database information..."
sudo docker exec datachain-studio-postgres psql -U studio -c "SELECT version();" > ${BUNDLE_DIR}/database-version.txt 2>/dev/null || echo "Database not accessible" > ${BUNDLE_DIR}/database-version.txt

# Package the bundle
echo "Creating support bundle archive..."
tar -czf ${BUNDLE_DIR}.tar.gz ${BUNDLE_DIR}
rm -rf ${BUNDLE_DIR}

echo "Support bundle created: ${BUNDLE_DIR}.tar.gz"
echo "Please provide this file when contacting support."
```

## Manual Support Bundle

If the automated scripts don't work, collect information manually:

### System Information

```bash
# Basic system info
uname -a
cat /etc/os-release
free -h
df -h
lscpu

# Kubernetes info (if applicable)
kubectl version
kubectl get nodes
kubectl cluster-info
```

### Application Status

```bash
# Kubernetes
kubectl get pods -n datachain-studio
kubectl get services -n datachain-studio
kubectl describe pods -n datachain-studio

# AMI
sudo systemctl status datachain-studio
sudo docker ps -a
sudo docker images
```

### Configuration Files

```bash
# Kubernetes
helm get values datachain-studio -n datachain-studio
kubectl get configmap datachain-studio-config -n datachain-studio -o yaml

# AMI
sudo cat /opt/datachain-studio/config.yml
sudo systemctl cat datachain-studio
```

### Logs Collection

```bash
# Kubernetes - Recent logs (last 1000 lines)
kubectl logs --tail=1000 deployment/datachain-studio-backend -n datachain-studio
kubectl logs --tail=1000 deployment/datachain-studio-frontend -n datachain-studio
kubectl logs --tail=1000 deployment/datachain-studio-worker -n datachain-studio

# AMI - Service logs
sudo journalctl -u datachain-studio --lines=1000
sudo docker logs --tail=1000 datachain-studio-backend
sudo docker logs --tail=1000 datachain-studio-frontend
```

### Network and Connectivity

```bash
# Network configuration
ip addr show
netstat -tulpn

# DNS resolution
nslookup studio.yourcompany.com
dig studio.yourcompany.com

# Connectivity tests
curl -I https://studio.yourcompany.com/health
telnet studio.yourcompany.com 443
```

### Database and Storage

```bash
# Database status (Kubernetes)
kubectl exec -it deployment/datachain-studio-postgres -n datachain-studio -- \
  psql -U studio -c "SELECT version();"

# Database status (AMI)
sudo docker exec datachain-studio-postgres \
  psql -U studio -c "SELECT version();"

# Storage information
kubectl get pv,pvc -n datachain-studio  # Kubernetes
df -h  # AMI
```

## Sensitive Information Handling

### Data Sanitization

Before sharing support bundles, sanitize sensitive information:

```bash
# Remove sensitive data from configuration files
sed -i 's/password: .*/password: [REDACTED]/g' config.yml
sed -i 's/secret: .*/secret: [REDACTED]/g' config.yml
sed -i 's/token: .*/token: [REDACTED]/g' config.yml

# Remove sensitive environment variables from logs
sed -i 's/PASSWORD=.*/PASSWORD=[REDACTED]/g' logs.txt
sed -i 's/SECRET=.*/SECRET=[REDACTED]/g' logs.txt
sed -i 's/TOKEN=.*/TOKEN=[REDACTED]/g' logs.txt
```

### What to Redact

Always redact these items:
- Passwords and passphrases
- API keys and tokens
- Database connection strings with passwords
- OAuth client secrets
- Private keys
- Personal identifiable information (PII)
- Internal IP addresses (if security sensitive)
- Domain names (if security sensitive)

### What to Keep

Keep these items for troubleshooting:
- Error messages and stack traces
- Configuration structure (without sensitive values)
- Resource usage statistics
- Network connectivity information
- Service status and health checks
- Log entries showing application behavior

## Support Bundle Analysis

### Common Issues Identified

Support bundles help identify:

1. **Resource Constraints**
   - Out of memory conditions
   - CPU throttling
   - Disk space issues
   - Network bandwidth problems

2. **Configuration Errors**
   - Invalid YAML syntax
   - Missing required settings
   - Incorrect service endpoints
   - Certificate issues

3. **Connectivity Problems**
   - DNS resolution failures
   - Network routing issues
   - Firewall blocks
   - SSL/TLS handshake failures

4. **Application Errors**
   - Database connection failures
   - Authentication issues
   - Missing dependencies
   - Version incompatibilities

### Automated Analysis

Create a script to perform basic analysis:

```bash
#!/bin/bash
# analyze-support-bundle.sh

BUNDLE_DIR=$1

if [ -z "$BUNDLE_DIR" ]; then
    echo "Usage: $0 <bundle-directory>"
    exit 1
fi

echo "Analyzing support bundle: $BUNDLE_DIR"
echo "========================================"

# Check for common error patterns
echo "Common Errors Found:"
grep -r -i "error\|failed\|exception" ${BUNDLE_DIR}/logs-* 2>/dev/null | head -10

echo ""
echo "Resource Usage Issues:"
if [ -f "${BUNDLE_DIR}/resource-usage-pods.txt" ]; then
    awk 'NR>1 && ($3 > 80 || $4 > 80) {print $1 " - High resource usage: CPU " $3 " Memory " $4}' ${BUNDLE_DIR}/resource-usage-pods.txt
fi

echo ""
echo "Pod Status Issues:"
if [ -f "${BUNDLE_DIR}/all-resources.txt" ]; then
    grep -E "CrashLoopBackOff|ImagePullBackOff|Error|Failed" ${BUNDLE_DIR}/all-resources.txt
fi

echo ""
echo "Recent Events:"
if [ -f "${BUNDLE_DIR}/events.txt" ]; then
    tail -10 ${BUNDLE_DIR}/events.txt
fi
```

## Sharing Support Bundles

### Secure Transfer

When sharing support bundles:

1. **Encrypt the bundle**:
   ```bash
   gpg --symmetric --cipher-algo AES256 datachain-studio-support-bundle.tar.gz
   ```

2. **Use secure channels**:
   - Support portal file upload
   - Encrypted email
   - Secure file sharing services
   - Corporate file transfer tools

3. **Share decryption key separately**:
   - Different communication channel
   - Phone call or secure messaging
   - Time-limited access

### Support Portal Upload

If using a support portal:

1. Create support ticket with issue description
2. Upload sanitized support bundle
3. Include reproduction steps
4. Specify urgency level
5. Provide contact information

## Custom Support Bundle

### Organization-Specific Information

Add organization-specific diagnostic information:

```bash
#!/bin/bash
# custom-diagnostics.sh

# Custom health checks
echo "Running custom health checks..."

# Check integration with internal services
curl -s -f https://internal-ldap.company.com/health > custom-ldap-check.txt || echo "LDAP check failed" > custom-ldap-check.txt

# Check custom storage mounts
df -h /mnt/company-storage > custom-storage-check.txt 2>/dev/null || echo "Custom storage not mounted" > custom-storage-check.txt

# Check VPN connectivity
ping -c 3 internal-gateway.company.com > custom-network-check.txt 2>&1

# Check custom certificates
openssl s_client -connect internal-ca.company.com:443 -servername internal-ca.company.com < /dev/null 2>&1 | openssl x509 -dates -noout > custom-cert-check.txt 2>/dev/null || echo "Custom CA check failed" > custom-cert-check.txt
```

### Environment-Specific Checks

```bash
# Check air-gapped environment specifics
if [ -f "/etc/airgap-marker" ]; then
    echo "Air-gapped environment detected"

    # Check internal registry connectivity
    curl -I https://registry.internal.company.com > internal-registry-check.txt 2>&1

    # Check internal DNS
    nslookup studio.internal.company.com > internal-dns-check.txt 2>&1

    # Check offline documentation
    ls -la /opt/datachain-docs/ > offline-docs-check.txt 2>&1
fi
```

## Next Steps

After generating a support bundle:

1. **Review the bundle** for sensitive information
2. **Sanitize** any confidential data
3. **Compress and encrypt** if required
4. **Upload to support portal** or send via secure channel
5. **Include detailed problem description** with the bundle
6. **Provide steps to reproduce** the issue
7. **Specify urgency level** and business impact

For immediate assistance while waiting for support:
- Check the [main troubleshooting guide](index.md)
- Review [502 error troubleshooting](502-errors.md)
- Consult [configuration documentation](../configuration/index.md)
- Search community forums for similar issues
