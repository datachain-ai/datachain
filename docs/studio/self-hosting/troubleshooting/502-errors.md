# 502 Bad Gateway Errors

Getting HTTP 502 Bad Gateway errors when accessing DataChain Studio indicates that the web server cannot connect to the backend application services. This guide covers diagnosing and resolving these issues.

## Overview

502 Bad Gateway errors occur when:
- Backend services are not running or accessible
- Network connectivity issues between components
- Resource constraints preventing service startup
- Configuration problems with load balancers or ingress

## Initial Diagnosis

### Check Service Status

#### Kubernetes Deployments

```bash
# Check pod status
kubectl get pods -n datachain-studio

# Check service status
kubectl get services -n datachain-studio

# Check ingress status
kubectl get ingress -n datachain-studio

# Look for events
kubectl get events -n datachain-studio --sort-by='.lastTimestamp'
```

#### AMI Deployments

```bash
# SSH to the instance first
ssh -i your-key.pem ubuntu@your-instance-ip

# Check system service status
sudo systemctl status datachain-studio

# Check container status
sudo docker ps -a

# Check logs
sudo journalctl -u datachain-studio -f
```

### Identify the Problem

Common pod statuses indicating issues:

- `ImagePullBackOff` / `ErrImagePull` - Container image issues
- `CrashLoopBackOff` - Application startup failures
- `Pending` - Resource or scheduling issues
- `CreateContainerConfigError` - Configuration problems

## Container Image Issues

### Image Pull Problems

If pods show `ImagePullBackOff` or `ErrImagePull`:

#### For Cloud Deployments

```bash
# Check image pull secrets
kubectl get secrets -n datachain-studio | grep registry

# Recreate registry secret if needed
kubectl delete secret datachain-registry -n datachain-studio

kubectl create secret docker-registry datachain-registry \
  --namespace datachain-studio \
  --docker-server=registry.datachain.ai \
  --docker-username=your-username \
  --docker-password=your-password

# Restart deployments
kubectl rollout restart deployment/datachain-studio-backend -n datachain-studio
kubectl rollout restart deployment/datachain-studio-frontend -n datachain-studio
kubectl rollout restart deployment/datachain-studio-worker -n datachain-studio
```

#### For Air-gapped Deployments

```bash
# Check if images exist in internal registry
kubectl describe pod POD_NAME -n datachain-studio | grep -i image

# Verify internal registry connectivity
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  curl -I https://registry.internal.company.com

# Re-tag and push images if needed
docker tag datachain/studio-backend:VERSION registry.internal.company.com/datachain/studio-backend:VERSION
docker push registry.internal.company.com/datachain/studio-backend:VERSION
```

### Image Version Mismatches

```bash
# Check configured image versions
kubectl get deployment -n datachain-studio -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.template.spec.containers[0].image}{"\n"}{end}'

# Update to correct versions if needed
kubectl set image deployment/datachain-studio-backend \
  datachain-studio-backend=registry.datachain.ai/studio-backend:CORRECT_VERSION \
  -n datachain-studio
```

## Application Startup Issues

### Configuration Problems

#### Check Configuration

```bash
# Review configuration
kubectl get configmap datachain-studio-config -n datachain-studio -o yaml

# Check for missing environment variables
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- env | grep -i studio

# Validate secrets
kubectl get secrets -n datachain-studio
kubectl describe secret datachain-studio-secrets -n datachain-studio
```

#### Database Connection Issues

```bash
# Test database connectivity
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  python -c "
import os
import psycopg2
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    print('Database connection: OK')
except Exception as e:
    print(f'Database connection failed: {e}')
"

# Check database pod status
kubectl get pods -l app=postgres -n datachain-studio

# Check database logs
kubectl logs -f deployment/datachain-studio-postgres -n datachain-studio
```

#### Redis Connection Issues

```bash
# Test Redis connectivity
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  python -c "
import redis
import os
try:
    r = redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379'))
    r.ping()
    print('Redis connection: OK')
except Exception as e:
    print(f'Redis connection failed: {e}')
"

# Check Redis pod status
kubectl get pods -l app=redis -n datachain-studio

# Check Redis logs
kubectl logs -f deployment/datachain-studio-redis -n datachain-studio
```

### Resource Constraints

#### Check Resource Usage

```bash
# Check node resources
kubectl describe nodes | grep -A 5 "Allocated resources"

# Check pod resource requests and limits
kubectl describe pod POD_NAME -n datachain-studio | grep -A 10 -i resources

# Check actual resource usage
kubectl top nodes
kubectl top pods -n datachain-studio
```

#### Resolve Resource Issues

```bash
# Scale down other workloads temporarily
kubectl scale deployment other-deployment --replicas=0 -n other-namespace

# Increase resource limits in Helm values
# values.yaml
resources:
  backend:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "1000m"

# Apply changes
helm upgrade datachain-studio datachain/studio \
  --namespace datachain-studio \
  --values values.yaml
```

## Network Connectivity Issues

### Service Discovery Problems

```bash
# Check service endpoints
kubectl get endpoints -n datachain-studio

# Test internal service connectivity
kubectl exec -it deployment/datachain-studio-frontend -n datachain-studio -- \
  curl -I http://datachain-studio-backend:8000/health

# Check DNS resolution
kubectl exec -it deployment/datachain-studio-frontend -n datachain-studio -- \
  nslookup datachain-studio-backend.datachain-studio.svc.cluster.local
```

### Ingress Configuration Issues

```bash
# Check ingress configuration
kubectl describe ingress datachain-studio-ingress -n datachain-studio

# Check ingress controller logs
kubectl logs -f deployment/nginx-ingress-controller -n ingress-nginx

# Test ingress rules
curl -H "Host: studio.yourcompany.com" http://INGRESS_IP/health
```

### Load Balancer Issues

```bash
# Check load balancer status
kubectl get service datachain-studio-lb -n datachain-studio

# Check load balancer endpoints
kubectl describe service datachain-studio-lb -n datachain-studio

# Test load balancer connectivity
curl -I http://LOAD_BALANCER_IP:80/health
```

## SSL/TLS Related Issues

### Certificate Problems

```bash
# Check TLS secret
kubectl describe secret datachain-studio-tls -n datachain-studio

# Verify certificate validity
kubectl get secret datachain-studio-tls -n datachain-studio -o jsonpath='{.data.tls\.crt}' | \
  base64 -d | openssl x509 -dates -noout

# Test SSL connectivity
openssl s_client -connect studio.yourcompany.com:443 -servername studio.yourcompany.com
```

### SSL Termination Issues

```bash
# Check if SSL is terminated at ingress
kubectl describe ingress datachain-studio-ingress -n datachain-studio | grep -i tls

# Test without SSL (if applicable)
curl -I http://studio.yourcompany.com/health

# Check SSL redirect configuration
curl -I -L http://studio.yourcompany.com/health
```

## Advanced Troubleshooting

### Deep Dive Debugging

#### Application Logs Analysis

```bash
# Get detailed application logs
kubectl logs -f deployment/datachain-studio-backend -n datachain-studio --previous

# Search for specific error patterns
kubectl logs deployment/datachain-studio-backend -n datachain-studio | grep -i error
kubectl logs deployment/datachain-studio-backend -n datachain-studio | grep -i "502\|bad gateway"

# Check application startup sequence
kubectl logs deployment/datachain-studio-backend -n datachain-studio | head -50
```

#### Network Packet Analysis

```bash
# Capture network traffic (requires privileged access)
kubectl exec -it deployment/datachain-studio-frontend -n datachain-studio -- \
  tcpdump -i any -n port 8000

# Test specific network paths
kubectl exec -it deployment/datachain-studio-frontend -n datachain-studio -- \
  traceroute datachain-studio-backend.datachain-studio.svc.cluster.local
```

#### Health Check Validation

```bash
# Test health endpoints directly
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  curl -f http://localhost:8000/health

# Test with verbose output
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  curl -v http://localhost:8000/health

# Check health endpoint response time
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- \
  time curl -f http://localhost:8000/health
```

## AMI-Specific Troubleshooting

### Docker Container Issues

```bash
# Check container status
sudo docker ps -a

# Check container logs
sudo docker logs datachain-studio-backend
sudo docker logs datachain-studio-frontend

# Restart containers
sudo docker restart datachain-studio-backend
sudo docker restart datachain-studio-frontend

# Check container health
sudo docker exec datachain-studio-backend curl -f http://localhost:8000/health
```

### System Service Issues

```bash
# Check systemd service status
sudo systemctl status datachain-studio
sudo systemctl status docker

# Restart services
sudo systemctl restart datachain-studio
sudo systemctl restart docker

# Check service logs
sudo journalctl -u datachain-studio -f
sudo journalctl -u docker -f

# Check service configuration
sudo systemctl cat datachain-studio
```

### Nginx Configuration

```bash
# Check nginx configuration
sudo nginx -t

# Check nginx logs
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log

# Restart nginx
sudo systemctl restart nginx

# Test nginx upstream
curl -I http://localhost:8000/health  # Direct backend test
```

## Recovery Procedures

### Quick Recovery Steps

1. **Restart all services**:
   ```bash
   # Kubernetes
   kubectl rollout restart deployment -n datachain-studio

   # AMI
   sudo systemctl restart datachain-studio
   ```

2. **Check and fix resource constraints**:
   ```bash
   kubectl top nodes
   kubectl describe nodes | grep -A 5 "Allocated resources"
   ```

3. **Verify configuration**:
   ```bash
   kubectl get configmap datachain-studio-config -n datachain-studio -o yaml
   ```

4. **Test connectivity**:
   ```bash
   curl -f https://studio.yourcompany.com/health
   ```

### Full Recovery Process

1. **Stop all services**
2. **Check system resources and fix constraints**
3. **Verify configuration files**
4. **Check network connectivity**
5. **Start services in order**: Database → Redis → Backend → Frontend → Worker
6. **Validate each component before starting the next**
7. **Test full application functionality**

## Prevention

### Monitoring and Alerting

Set up monitoring to catch 502 errors early:

```yaml
# Prometheus alert example
- alert: High502ErrorRate
  expr: rate(nginx_ingress_controller_requests{status="502"}[5m]) > 0.1
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "High rate of 502 errors"
    description: "502 error rate is {{ $value }} per second"

- alert: BackendServiceDown
  expr: up{job="datachain-studio-backend"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Backend service is down"
```

### Health Checks

Implement comprehensive health checks:

```yaml
# Kubernetes readiness probe
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 30
  timeoutSeconds: 10
  failureThreshold: 3
```

### Regular Maintenance

1. **Monitor resource usage trends**
2. **Review logs regularly for warnings**
3. **Keep services updated**
4. **Test failover procedures**
5. **Document configuration changes**

## Next Steps

If 502 errors persist after trying these solutions:

1. **Generate a [support bundle](support-bundle.md)** with diagnostic information
2. **Review recent changes** to configuration or infrastructure
3. **Check the [main troubleshooting guide](index.md)** for other common issues
4. **Contact support** with detailed error information and logs

For other issues:
- [Configuration problems](../configuration/index.md)
- [Installation issues](../installation/index.md)
- [Upgrade problems](../upgrading/index.md)
