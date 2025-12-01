# CA Certificates

This guide covers how to configure custom Certificate Authority (CA) certificates for your self-hosted DataChain Studio instance. This is necessary when your organization uses internal CAs or when connecting to services with custom certificates.

## Overview

DataChain Studio may need to trust custom CA certificates in several scenarios:

- **Internal Git Servers**: Self-hosted GitLab, GitHub Enterprise with custom certificates
- **Storage Services**: S3-compatible storage with custom certificates
- **Corporate Proxies**: HTTPS proxies with internal certificates
- **Database Connections**: PostgreSQL/Redis with SSL using custom CAs

## Configuration Methods

### Kubernetes/Helm Deployment

#### Method 1: ConfigMap with Certificate Files

1. Create a ConfigMap with your CA certificates:

```bash
kubectl create configmap custom-ca-certs \
  --namespace datachain-studio \
  --from-file=ca1.crt=/path/to/your/ca1.crt \
  --from-file=ca2.crt=/path/to/your/ca2.crt
```

2. Configure Helm values to mount the certificates:

```yaml
# values.yaml
global:
  customCaCerts:
    enabled: true
    configMapName: custom-ca-certs

# Alternative: Inline certificates
global:
  customCaCerts:
    certificates:
      - name: "internal-ca"
        certificate: |
          -----BEGIN CERTIFICATE-----
          MIIDXTCCAkWgAwIBAgIJAKoK/heBjcOuMA0GCSqGSIb3DQEBBQUAMEUxCzAJBgNV
          BAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBX
          ... (certificate content) ...
          -----END CERTIFICATE-----
      - name: "corporate-ca"
        certificate: |
          -----BEGIN CERTIFICATE-----
          ... (another certificate) ...
          -----END CERTIFICATE-----
```

3. Apply the configuration:

```bash
helm upgrade datachain-studio datachain/studio \
  --namespace datachain-studio \
  --values values.yaml
```

#### Method 2: Direct Certificate Configuration

Add certificates directly to your Helm values:

```yaml
# values.yaml
global:
  customCaCerts:
    - |-
      -----BEGIN CERTIFICATE-----
      MIIDXTCCAkWgAwIBAgIJAKoK/heBjcOuMA0GCSqGSIb3DQEBBQUAMEUxCzAJBgNV
      BAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBX
      ... (rest of certificate) ...
      -----END CERTIFICATE-----
    - |-
      -----BEGIN CERTIFICATE-----
      ... (another certificate) ...
      -----END CERTIFICATE-----
```

### AWS AMI Deployment

For AMI deployments, configure CA certificates directly on the instance:

#### 1. Upload CA Certificates

```bash
# Copy certificate files to the instance
scp -i your-key.pem ca-certificates.crt ubuntu@your-instance:/tmp/

# SSH to the instance
ssh -i your-key.pem ubuntu@your-instance

# Install CA certificates
sudo cp /tmp/ca-certificates.crt /usr/local/share/ca-certificates/
sudo update-ca-certificates
```

#### 2. Configure DataChain Studio

Update the configuration file:

```yaml
# /opt/datachain-studio/config.yml
global:
  ssl:
    caCertificates:
      - /usr/local/share/ca-certificates/ca-certificates.crt

    # Alternatively, inline certificates
    customCaCerts:
      - |
        -----BEGIN CERTIFICATE-----
        ... certificate content ...
        -----END CERTIFICATE-----
```

#### 3. Restart Services

```bash
sudo systemctl restart datachain-studio
```

## Use Cases and Examples

### Internal GitLab Server

Configure CA certificates for connecting to an internal GitLab server:

```yaml
global:
  customCaCerts:
    - |-
      -----BEGIN CERTIFICATE-----
      MIIDXTCCAkWgAwIBAgIJAKoK/heBjcOuMA0GCSqGSIb3DQEBBQUAMEUxCzAJBgNV
      ... (your GitLab CA certificate) ...
      -----END CERTIFICATE-----

  git:
    gitlab:
      enabled: true
      url: "https://gitlab.internal.company.com"
      clientId: "your-gitlab-client-id"
      clientSecret: "your-gitlab-client-secret"

      # SSL verification settings
      ssl:
        verify: true
        caCertificate: true  # Use custom CA
```

### S3-Compatible Storage with Custom CA

Configure certificates for custom S3-compatible storage:

```yaml
global:
  customCaCerts:
    - |-
      -----BEGIN CERTIFICATE-----
      ... (your storage provider's CA certificate) ...
      -----END CERTIFICATE-----

storage:
  type: s3
  s3:
    endpoint: "https://s3.internal.company.com"
    bucket: "datachain-studio-storage"
    region: "us-east-1"

    # SSL settings
    ssl:
      enabled: true
      verify: true
      caCertificate: true
```

### Corporate Proxy with Custom CA

Configure certificates for accessing external services through a corporate proxy:

```yaml
global:
  # Proxy configuration
  proxy:
    enabled: true
    http: "http://proxy.company.com:8080"
    https: "https://proxy.company.com:8080"

  # CA certificates for proxy SSL
  customCaCerts:
    - |-
      -----BEGIN CERTIFICATE-----
      ... (your proxy's CA certificate) ...
      -----END CERTIFICATE-----
```

### Multiple CA Certificates

Configure multiple CA certificates for different services:

```yaml
global:
  customCaCerts:
    # Internal root CA
    - |-
      -----BEGIN CERTIFICATE-----
      ... (internal root CA) ...
      -----END CERTIFICATE-----

    # GitLab intermediate CA
    - |-
      -----BEGIN CERTIFICATE-----
      ... (GitLab intermediate CA) ...
      -----END CERTIFICATE-----

    # Storage service CA
    - |-
      -----BEGIN CERTIFICATE-----
      ... (storage service CA) ...
      -----END CERTIFICATE-----
```

## Certificate Chain Validation

### Obtaining CA Certificates

Get CA certificates from various sources:

#### From a Website
```bash
# Extract certificate chain from a website
openssl s_client -showcerts -connect gitlab.company.com:443 </dev/null 2>/dev/null | openssl x509 -outform PEM > gitlab-ca.crt
```

#### From a Certificate File
```bash
# Extract CA from a certificate bundle
openssl x509 -in certificate-bundle.crt -out ca-certificate.crt
```

#### From System Trust Store
```bash
# Export system CA certificates
cat /etc/ssl/certs/ca-certificates.crt > system-cas.crt
```

### Validating Certificate Chains

Test certificate validation:

```bash
# Verify certificate against CA
openssl verify -CAfile ca-certificate.crt server-certificate.crt

# Test SSL connection with custom CA
openssl s_client -connect gitlab.company.com:443 -CAfile ca-certificate.crt
```

## Troubleshooting

### Common Issues

**SSL verification errors:**
```
ERROR: SSL certificate verification failed
```

**Solution:**
1. Verify CA certificate is correct and complete
2. Check certificate format (PEM format required)
3. Ensure certificate chain is complete

**Certificate format issues:**
```
ERROR: Invalid certificate format
```

**Solution:**
1. Ensure certificates are in PEM format
2. Check for proper BEGIN/END markers
3. Validate certificate using OpenSSL

### Debugging CA Certificate Issues

#### 1. Check Certificate Details

```bash
# View certificate information
openssl x509 -in ca-certificate.crt -text -noout

# Check certificate validity
openssl x509 -in ca-certificate.crt -noout -dates
```

#### 2. Test Certificate Trust

```bash
# Test connection with custom CA
curl --cacert ca-certificate.crt https://gitlab.company.com

# Test with verbose output
curl -v --cacert ca-certificate.crt https://gitlab.company.com
```

#### 3. Validate Certificate Chain

```bash
# Check certificate chain
openssl verify -verbose -CAfile ca-certificate.crt intermediate.crt

# Show certificate chain
openssl s_client -connect gitlab.company.com:443 -showcerts
```

### Container-Level Debugging

For Kubernetes deployments, debug CA certificate issues:

```bash
# Check if certificates are mounted correctly
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- ls -la /etc/ssl/certs/

# Test certificate from within container
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- openssl s_client -connect gitlab.company.com:443 -CApath /etc/ssl/certs

# Check certificate trust store
kubectl exec -it deployment/datachain-studio-backend -n datachain-studio -- cat /etc/ssl/certs/ca-certificates.crt | grep -A 20 "Your CA Name"
```

## Security Best Practices

### Certificate Management

1. **Regular Updates**: Keep CA certificates updated
2. **Secure Storage**: Store CA certificates securely
3. **Access Control**: Limit access to CA certificate files
4. **Validation**: Regularly validate certificate chains

### Certificate Rotation

```yaml
# Automated certificate rotation
global:
  certificates:
    rotation:
      enabled: true
      schedule: "0 2 * * 0"  # Weekly on Sunday at 2 AM
      backup: true

    validation:
      enabled: true
      checkExpiry: true
      daysBeforeExpiry: 30
```

### Monitoring and Alerting

```yaml
# Certificate monitoring
monitoring:
  certificates:
    enabled: true
    alerts:
      - name: "Certificate Expiry Warning"
        condition: "certificate_days_until_expiry < 30"
        severity: "warning"

      - name: "Certificate Expiry Critical"
        condition: "certificate_days_until_expiry < 7"
        severity: "critical"
```

## Validation and Testing

### Post-Configuration Validation

```bash
# Test DataChain Studio connectivity
curl -I https://studio.company.com

# Verify certificate trust
openssl s_client -connect studio.company.com:443 -verify_return_error

# Check service logs for SSL errors
kubectl logs -f deployment/datachain-studio-backend -n datachain-studio | grep -i ssl
```

### Integration Testing

```bash
# Test Git integration
curl -k https://studio.company.com/api/git/test-connection

# Test storage connectivity
curl -k https://studio.company.com/api/storage/health

# Test webhook delivery
curl -k https://studio.company.com/api/webhooks/test
```

## Next Steps

- Configure [SSL/TLS certificates](ssl-tls.md) for secure communications
- Set up [Git forge integrations](git-forges/index.md) with custom certificates
- Review [troubleshooting guides](../troubleshooting/index.md) for SSL/TLS issues
- Learn about [upgrading procedures](../upgrading/index.md) with certificate considerations
