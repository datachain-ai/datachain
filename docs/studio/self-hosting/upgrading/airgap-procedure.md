# Airgap Upgrade Procedure

This guide covers the upgrade procedure for DataChain Studio deployments in air-gapped environments without direct internet access.

## Overview

Air-gapped upgrades require manually transferring container images and Helm charts to your isolated environment before performing the upgrade.

## Prerequisites

- Administrative access to your air-gapped Kubernetes cluster
- Access to a connected system for downloading images and charts
- Container registry in your air-gapped environment
- Backup of current configuration and data
- Scheduled maintenance window

## Pre-upgrade Preparation

### 1. Review Release Notes

On a connected system, review the release notes for:
- Breaking changes that may affect your deployment
- New configuration options
- Deprecated features
- Security updates

### 2. Create Backups

Follow the same backup procedures as the [regular upgrade](regular-procedure.md#pre-upgrade-preparation):

```bash
# Database backup
kubectl exec -it deployment/datachain-studio-postgres -n datachain-studio -- \
  pg_dump -U studio datachain_studio > backup-$(date +%Y%m%d-%H%M%S).sql

# Configuration backup
helm get values datachain-studio -n datachain-studio > values-backup-$(date +%Y%m%d).yaml

# Storage backup
kubectl get pv,pvc -n datachain-studio -o yaml > pv-backup-$(date +%Y%m%d).yaml
```

## Download Assets (Connected System)

### 1. Download Container Images

On a system with internet access, download the required container images:

```bash
# Set target version
TARGET_VERSION="1.2.3"

# Download DataChain Studio images
docker pull datachain/studio-frontend:${TARGET_VERSION}
docker pull datachain/studio-backend:${TARGET_VERSION}
docker pull datachain/studio-worker:${TARGET_VERSION}
docker pull datachain/studio-scheduler:${TARGET_VERSION}

# Download dependency images (if updated)
docker pull postgres:14
docker pull redis:7
docker pull nginx:1.24

# Export images to tar files
docker save datachain/studio-frontend:${TARGET_VERSION} > studio-frontend-${TARGET_VERSION}.tar
docker save datachain/studio-backend:${TARGET_VERSION} > studio-backend-${TARGET_VERSION}.tar
docker save datachain/studio-worker:${TARGET_VERSION} > studio-worker-${TARGET_VERSION}.tar
docker save datachain/studio-scheduler:${TARGET_VERSION} > studio-scheduler-${TARGET_VERSION}.tar

# Export dependency images if needed
docker save postgres:14 > postgres-14.tar
docker save redis:7 > redis-7.tar
docker save nginx:1.24 > nginx-1.24.tar
```

### 2. Download Helm Chart

```bash
# Update Helm repository
helm repo add datachain https://charts.datachain.ai
helm repo update

# Download specific chart version
helm pull datachain/studio --version ${TARGET_VERSION}

# This creates a file like: studio-${TARGET_VERSION}.tgz
```

### 3. Download Dependencies (if needed)

```bash
# Download any additional charts or dependencies
helm dependency update studio-${TARGET_VERSION}.tgz
```

## Transfer Assets to Air-gapped Environment

### 1. Transfer Files

Transfer the downloaded files to your air-gapped environment:

```bash
# Files to transfer:
# - studio-frontend-${TARGET_VERSION}.tar
# - studio-backend-${TARGET_VERSION}.tar
# - studio-worker-${TARGET_VERSION}.tar
# - studio-scheduler-${TARGET_VERSION}.tar
# - postgres-14.tar (if updated)
# - redis-7.tar (if updated)
# - nginx-1.24.tar (if updated)
# - studio-${TARGET_VERSION}.tgz

# Use secure transfer method appropriate for your environment:
# - Physical media (USB drive, CD/DVD)
# - Secure file transfer over isolated network
# - Air-gapped file transfer tools
```

### 2. Verify Transfer Integrity

```bash
# Verify checksums to ensure files weren't corrupted
sha256sum studio-frontend-${TARGET_VERSION}.tar
sha256sum studio-backend-${TARGET_VERSION}.tar
sha256sum studio-worker-${TARGET_VERSION}.tar
sha256sum studio-scheduler-${TARGET_VERSION}.tar
sha256sum studio-${TARGET_VERSION}.tgz

# Compare with checksums from connected system
```

## Load Assets in Air-gapped Environment

### 1. Load Container Images

```bash
# Load images into local Docker daemon
docker load < studio-frontend-${TARGET_VERSION}.tar
docker load < studio-backend-${TARGET_VERSION}.tar
docker load < studio-worker-${TARGET_VERSION}.tar
docker load < studio-scheduler-${TARGET_VERSION}.tar

# Load dependency images if needed
docker load < postgres-14.tar
docker load < redis-7.tar
docker load < nginx-1.24.tar

# Verify images are loaded
docker images | grep datachain
```

### 2. Tag and Push to Internal Registry

```bash
# Set your internal registry URL
INTERNAL_REGISTRY="registry.internal.company.com"

# Tag images for internal registry
docker tag datachain/studio-frontend:${TARGET_VERSION} ${INTERNAL_REGISTRY}/datachain/studio-frontend:${TARGET_VERSION}
docker tag datachain/studio-backend:${TARGET_VERSION} ${INTERNAL_REGISTRY}/datachain/studio-backend:${TARGET_VERSION}
docker tag datachain/studio-worker:${TARGET_VERSION} ${INTERNAL_REGISTRY}/datachain/studio-worker:${TARGET_VERSION}
docker tag datachain/studio-scheduler:${TARGET_VERSION} ${INTERNAL_REGISTRY}/datachain/studio-scheduler:${TARGET_VERSION}

# Push to internal registry
docker push ${INTERNAL_REGISTRY}/datachain/studio-frontend:${TARGET_VERSION}
docker push ${INTERNAL_REGISTRY}/datachain/studio-backend:${TARGET_VERSION}
docker push ${INTERNAL_REGISTRY}/datachain/studio-worker:${TARGET_VERSION}
docker push ${INTERNAL_REGISTRY}/datachain/studio-scheduler:${TARGET_VERSION}

# Push dependency images if needed
docker tag postgres:14 ${INTERNAL_REGISTRY}/postgres:14
docker tag redis:7 ${INTERNAL_REGISTRY}/redis:7
docker push ${INTERNAL_REGISTRY}/postgres:14
docker push ${INTERNAL_REGISTRY}/redis:7
```

### 3. Extract and Install Helm Chart

```bash
# Extract Helm chart
tar -xzf studio-${TARGET_VERSION}.tgz

# Add to local Helm repository (if using chartmuseum or similar)
# Or use directly from extracted directory
```

## Update Configuration for Air-gapped Environment

### 1. Update Image References

Update your `values.yaml` to reference internal registry:

```yaml
# values.yaml
global:
  imageRegistry: "registry.internal.company.com"

images:
  frontend:
    repository: datachain/studio-frontend
    tag: "1.2.3"
    pullPolicy: IfNotPresent

  backend:
    repository: datachain/studio-backend
    tag: "1.2.3"
    pullPolicy: IfNotPresent

  worker:
    repository: datachain/studio-worker
    tag: "1.2.3"
    pullPolicy: IfNotPresent

  scheduler:
    repository: datachain/studio-scheduler
    tag: "1.2.3"
    pullPolicy: IfNotPresent

# Update dependency images if needed
postgresql:
  image:
    registry: registry.internal.company.com
    repository: postgres
    tag: "14"

redis:
  image:
    registry: registry.internal.company.com
    repository: redis
    tag: "7"
```

### 2. Configure Image Pull Secrets (if needed)

```bash
# Create image pull secret for internal registry
kubectl create secret docker-registry internal-registry-secret \
  --namespace datachain-studio \
  --docker-server=registry.internal.company.com \
  --docker-username=your-username \
  --docker-password=your-password

# Reference in values.yaml
```

```yaml
# values.yaml
imagePullSecrets:
  - name: internal-registry-secret
```

## Perform the Upgrade

### 1. Plan the Upgrade

```bash
# Dry run to see what will change
helm upgrade datachain-studio ./studio \
  --namespace datachain-studio \
  --values values.yaml \
  --dry-run --debug
```

### 2. Execute the Upgrade

```bash
# Perform the upgrade using local chart
helm upgrade datachain-studio ./studio \
  --namespace datachain-studio \
  --values values.yaml \
  --wait \
  --timeout 10m

# Monitor upgrade progress
kubectl get pods -n datachain-studio -w
```

### 3. Verify Upgrade

```bash
# Check upgrade status
helm status datachain-studio -n datachain-studio

# Verify all pods are running with new images
kubectl get pods -n datachain-studio -o wide

# Check pod images
kubectl describe pod POD_NAME -n datachain-studio | grep -i image

# Test application health
curl -f https://studio.yourcompany.com/health
```

## Post-Upgrade Validation

### 1. Image Verification

Verify that pods are using the correct images from your internal registry:

```bash
# Check pod images
kubectl get pods -n datachain-studio -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[*].image}{"\n"}{end}'

# Verify images are from internal registry
kubectl describe pods -n datachain-studio | grep -i "image:"
```

### 2. Functional Testing

Follow the same functional testing procedures as the [regular upgrade](regular-procedure.md#post-upgrade-validation):

- Test authentication and authorization
- Verify API endpoints functionality
- Test database connectivity
- Validate Git integration
- Check webhook delivery

### 3. Performance Validation

Monitor system performance after upgrade:

```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n datachain-studio

# Monitor application logs
kubectl logs -f deployment/datachain-studio-backend -n datachain-studio

# Test response times
time curl -f https://studio.yourcompany.com/health
```

## Rollback Procedure for Air-gapped Environment

### 1. Rollback Using Helm

```bash
# List release history
helm history datachain-studio -n datachain-studio

# Rollback to previous version
helm rollback datachain-studio -n datachain-studio

# Verify rollback
helm status datachain-studio -n datachain-studio
kubectl get pods -n datachain-studio
```

### 2. Image Rollback

If you need to rollback to previous container images:

```bash
# Load previous version images (if available)
docker load < studio-frontend-PREVIOUS_VERSION.tar
docker load < studio-backend-PREVIOUS_VERSION.tar

# Tag and push to internal registry
docker tag datachain/studio-frontend:PREVIOUS_VERSION ${INTERNAL_REGISTRY}/datachain/studio-frontend:PREVIOUS_VERSION
docker push ${INTERNAL_REGISTRY}/datachain/studio-frontend:PREVIOUS_VERSION

# Update values.yaml with previous version tags
# Then run helm upgrade with previous configuration
```

## Troubleshooting Air-gapped Upgrades

### Image Pull Failures

**Cannot pull images from internal registry:**

```bash
# Check registry connectivity
nslookup registry.internal.company.com
telnet registry.internal.company.com 443

# Check authentication
kubectl get secret internal-registry-secret -n datachain-studio -o yaml

# Test image pull manually
docker pull registry.internal.company.com/datachain/studio-frontend:VERSION
```

**Images not found in registry:**

```bash
# Check if images were pushed correctly
curl -u username:password https://registry.internal.company.com/v2/_catalog
curl -u username:password https://registry.internal.company.com/v2/datachain/studio-frontend/tags/list

# Re-push images if necessary
docker push ${INTERNAL_REGISTRY}/datachain/studio-frontend:${TARGET_VERSION}
```

### Chart Installation Issues

**Chart not found:**

```bash
# Verify chart directory structure
ls -la ./studio/

# Check chart.yaml
cat ./studio/Chart.yaml

# Validate chart
helm lint ./studio
```

### Network Connectivity Issues

**Services cannot communicate:**

```bash
# Check internal DNS resolution
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- nslookup postgres-service

# Check service endpoints
kubectl get endpoints -n datachain-studio

# Test internal service connectivity
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- curl http://postgres-service:5432
```

## Best Practices for Air-gapped Upgrades

### Planning
1. **Test thoroughly** in air-gapped staging environment
2. **Prepare all assets** in advance on connected system
3. **Verify checksums** of all transferred files
4. **Document procedures** specific to your environment

### Execution
1. **Minimize downtime** by preparing everything in advance
2. **Monitor carefully** during the upgrade process
3. **Have rollback assets ready** (previous version images)
4. **Test connectivity** to internal services

### Maintenance
1. **Keep internal registry updated** with required images
2. **Maintain image versioning** strategy
3. **Document air-gap specific configurations**
4. **Plan for dependency updates**

## Automation for Air-gapped Environments

### Scripted Asset Preparation

Create scripts to automate asset preparation:

```bash
#!/bin/bash
# prepare-airgap-upgrade.sh

TARGET_VERSION=$1
INTERNAL_REGISTRY=$2

# Download images
docker pull datachain/studio-frontend:${TARGET_VERSION}
docker pull datachain/studio-backend:${TARGET_VERSION}

# Save to tar files
docker save datachain/studio-frontend:${TARGET_VERSION} > studio-frontend-${TARGET_VERSION}.tar
docker save datachain/studio-backend:${TARGET_VERSION} > studio-backend-${TARGET_VERSION}.tar

# Download Helm chart
helm pull datachain/studio --version ${TARGET_VERSION}

echo "Assets prepared for air-gapped upgrade to version ${TARGET_VERSION}"
```

### Deployment Scripts

```bash
#!/bin/bash
# deploy-airgap-upgrade.sh

TARGET_VERSION=$1
INTERNAL_REGISTRY=$2

# Load and push images
docker load < studio-frontend-${TARGET_VERSION}.tar
docker tag datachain/studio-frontend:${TARGET_VERSION} ${INTERNAL_REGISTRY}/datachain/studio-frontend:${TARGET_VERSION}
docker push ${INTERNAL_REGISTRY}/datachain/studio-frontend:${TARGET_VERSION}

# Extract and upgrade
tar -xzf studio-${TARGET_VERSION}.tgz
helm upgrade datachain-studio ./studio --namespace datachain-studio --values values.yaml --wait

echo "Air-gapped upgrade to version ${TARGET_VERSION} completed"
```

## Next Steps

- Review [configuration changes](../configuration/index.md) for new version
- Update [monitoring setup](../configuration/index.md#monitoring) if needed
- Plan for [next upgrade cycle](index.md) with lessons learned
- Document air-gap specific procedures for your organization

For issues during air-gapped upgrades, consult the [troubleshooting guide](../troubleshooting/index.md) and adapt solutions for your air-gapped environment.
