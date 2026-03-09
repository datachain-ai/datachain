# Kubernetes (Helm) Installation

This guide covers installing DataChain Studio on Kubernetes using Helm charts.

## Prerequisites

### Kubernetes Cluster

- **Kubernetes version**: 1.19+
- **Node requirements**:
  - Minimum: 2 nodes with 8GB RAM, 4 vCPUs each
  - Recommended: 3+ nodes with 16GB RAM, 8 vCPUs each
- **Storage**: 100GB persistent storage
- **Networking**: Cluster networking with ingress controller

### Required Tools

- `kubectl` configured to access your cluster
- `helm` 3.0+
- Access to DataChain Studio container images

### Access Requirements

- Container registry access (provided by DataChain team)
- Valid DNS domain for DataChain Studio
- SSL certificates for HTTPS

## Installation Steps

### 1. Add DataChain Helm Repository

```bash
helm repo add datachain https://charts.datachain.ai
helm repo update
```

### 2. Create Namespace

```bash
kubectl create namespace datachain-studio
```

### 3. Configure Container Registry Access

Create a secret for accessing DataChain Studio container images:

```bash
kubectl create secret docker-registry datachain-registry \
  --namespace datachain-studio \
  --docker-server=registry.datachain.ai \
  --docker-username=<provided-username> \
  --docker-password=<provided-password>
```

### 4. Configure SSL Certificates

Create TLS secret for your domain:

```bash
kubectl create secret tls studio-tls \
  --namespace datachain-studio \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key
```

### 5. Create Configuration File

Create a `values.yaml` file with your configuration:

```yaml
# Basic configuration
global:
  domain: studio.yourcompany.com
  storageClass: gp2  # or your preferred storage class

# Image pull secrets
imagePullSecrets:
  - name: datachain-registry

# SSL/TLS configuration
ingress:
  enabled: true
  className: nginx  # or your ingress class
  tls:
    enabled: true
    secretName: studio-tls
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"

# Database configuration
postgresql:
  enabled: true
  auth:
    postgresPassword: "secure-postgres-password"
    database: "datachain_studio"
  primary:
    persistence:
      enabled: true
      size: 50Gi
      storageClass: gp2

# Redis configuration
redis:
  enabled: true
  auth:
    enabled: true
    password: "secure-redis-password"

# Storage configuration
storage:
  type: s3
  s3:
    bucket: your-studio-bucket
    region: us-east-1
    accessKey: your-access-key
    secretKey: your-secret-key

# Git integrations
git:
  github:
    enabled: true
    appId: "your-github-app-id"
    privateKey: |
      -----BEGIN RSA PRIVATE KEY-----
      your-github-private-key-content
      -----END RSA PRIVATE KEY-----

  gitlab:
    enabled: true
    url: "https://gitlab.com"
    clientId: "your-gitlab-client-id"
    clientSecret: "your-gitlab-client-secret"

# Resource limits
resources:
  frontend:
    requests:
      memory: "512Mi"
      cpu: "250m"
    limits:
      memory: "1Gi"
      cpu: "500m"

  backend:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "1000m"

  worker:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "1000m"

# Autoscaling
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### 6. Install DataChain Studio

```bash
helm install datachain-studio datachain/studio \
  --namespace datachain-studio \
  --values values.yaml \
  --wait --timeout=10m
```

### 7. Verify Installation

Check pod status:
```bash
kubectl get pods -n datachain-studio
```

Check services:
```bash
kubectl get services -n datachain-studio
```

Check ingress:
```bash
kubectl get ingress -n datachain-studio
```

## Configuration Options

### Database Options

#### External PostgreSQL
```yaml
postgresql:
  enabled: false

externalDatabase:
  type: postgresql
  host: your-postgres-host
  port: 5432
  database: datachain_studio
  username: studio_user
  password: your-password
```

#### External Redis
```yaml
redis:
  enabled: false

externalRedis:
  host: your-redis-host
  port: 6379
  password: your-redis-password
```

### Storage Options

#### AWS S3
```yaml
storage:
  type: s3
  s3:
    bucket: your-bucket
    region: us-east-1
    accessKey: your-access-key
    secretKey: your-secret-key
```

#### Google Cloud Storage
```yaml
storage:
  type: gcs
  gcs:
    bucket: your-bucket
    projectId: your-project-id
    keyFile: |
      {
        "type": "service_account",
        "project_id": "your-project-id",
        ...
      }
```

#### Azure Blob Storage
```yaml
storage:
  type: azure
  azure:
    accountName: your-account-name
    accountKey: your-account-key
    containerName: your-container
```

### High Availability Configuration

```yaml
# Multiple replicas
replicaCount:
  frontend: 3
  backend: 3
  worker: 2

# Pod disruption budgets
podDisruptionBudget:
  enabled: true
  minAvailable: 1

# Node affinity
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app.kubernetes.io/name
            operator: In
            values:
            - datachain-studio
        topologyKey: kubernetes.io/hostname
```

## Upgrading

### Check Current Version
```bash
helm list -n datachain-studio
```

### Upgrade to Latest Version
```bash
helm repo update
helm upgrade datachain-studio datachain/studio \
  --namespace datachain-studio \
  --values values.yaml \
  --wait
```

### Rollback if Needed
```bash
helm rollback datachain-studio -n datachain-studio
```

## Monitoring and Logging

### Enable Monitoring
```yaml
monitoring:
  enabled: true
  serviceMonitor:
    enabled: true

  prometheus:
    enabled: true

  grafana:
    enabled: true
    adminPassword: your-grafana-password
```

### Log Configuration
```yaml
logging:
  level: INFO
  format: json

  # External log aggregation
  fluentd:
    enabled: true
    host: your-log-aggregator
    port: 24224
```

## Security Considerations

### Network Policies
```yaml
networkPolicy:
  enabled: true
  ingress:
    - from:
      - namespaceSelector:
          matchLabels:
            name: ingress-nginx
  egress:
    - to:
      - namespaceSelector: {}
```

### Security Context
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  capabilities:
    drop:
    - ALL
```

### Pod Security Standards
```yaml
podSecurityContext:
  seccompProfile:
    type: RuntimeDefault
```

## Troubleshooting

### Common Issues

**Pods stuck in Pending:**
```bash
kubectl describe pod <pod-name> -n datachain-studio
```

**Database connection issues:**
```bash
kubectl logs <backend-pod> -n datachain-studio
```

**SSL certificate problems:**
```bash
kubectl describe ingress -n datachain-studio
```

### Debug Commands

```bash
# Check all resources
kubectl get all -n datachain-studio

# Check events
kubectl get events -n datachain-studio --sort-by='.lastTimestamp'

# Check logs
kubectl logs -f deployment/datachain-studio-backend -n datachain-studio

# Port forward for local access
kubectl port-forward service/datachain-studio-frontend 8080:80 -n datachain-studio
```

## Next Steps

- [Configure additional settings](../configuration/index.md)
- [Set up monitoring and alerting](../configuration/index.md#monitoring)
- [Learn about backup procedures](../upgrading/index.md#backup)
- [Review security hardening](../configuration/index.md#security)
