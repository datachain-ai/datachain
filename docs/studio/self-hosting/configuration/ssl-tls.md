# SSL/TLS Configuration

This guide covers how to configure SSL/TLS certificates for secure HTTPS access to your self-hosted DataChain Studio instance.

## Overview

DataChain Studio supports both SSL and TLS certificate configurations for secure communication. This includes:

- **SSL Certificates**: Traditional SSL certificate configuration
- **TLS Certificates**: Modern TLS certificate setup (recommended)
- **Certificate Management**: Automated certificate renewal and validation
- **Security Hardening**: Advanced SSL/TLS security configurations

## TLS Certificate Configuration (Recommended)

### Prerequisites

- Valid domain name pointing to your DataChain Studio instance
- TLS certificate and private key files
- Proper DNS resolution

### Kubernetes/Helm Deployment

For Kubernetes deployments, create a TLS secret and configure the ingress:

#### 1. Create TLS Secret

```bash
kubectl create secret tls datachain-studio-tls \
  --namespace datachain-studio \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key
```

#### 2. Configure Helm Values

Add the following to your `values.yaml`:

```yaml
ingress:
  enabled: true
  className: nginx

  # TLS configuration
  tls:
    enabled: true
    secretName: datachain-studio-tls

  # Security annotations
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/ssl-ciphers: "ECDHE-RSA-AES128-GCM-SHA256,ECDHE-RSA-AES256-GCM-SHA384"
    nginx.ingress.kubernetes.io/ssl-protocols: "TLSv1.2 TLSv1.3"

global:
  # Enforce HTTPS
  tls:
    enabled: true
    minVersion: "1.2"
    cipherSuites:
      - "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
      - "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"
      - "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305"
```

#### 3. Apply Configuration

```bash
helm upgrade datachain-studio datachain/studio \
  --namespace datachain-studio \
  --values values.yaml \
  --wait
```

### AWS AMI Deployment

For AWS AMI deployments, configure SSL/TLS directly on the instance:

#### 1. Upload Certificates

```bash
# Copy certificate files to the instance
scp -i your-key.pem tls.crt ubuntu@your-instance:/tmp/
scp -i your-key.pem tls.key ubuntu@your-instance:/tmp/

# SSH to the instance
ssh -i your-key.pem ubuntu@your-instance

# Move certificates to proper location
sudo mkdir -p /etc/ssl/datachain-studio/
sudo mv /tmp/tls.crt /etc/ssl/datachain-studio/
sudo mv /tmp/tls.key /etc/ssl/datachain-studio/
sudo chown root:root /etc/ssl/datachain-studio/*
sudo chmod 644 /etc/ssl/datachain-studio/tls.crt
sudo chmod 600 /etc/ssl/datachain-studio/tls.key
```

#### 2. Configure DataChain Studio

Update the configuration file:

```yaml
# /opt/datachain-studio/config.yml
global:
  domain: studio.yourcompany.com

  tls:
    enabled: true
    certFile: /etc/ssl/datachain-studio/tls.crt
    keyFile: /etc/ssl/datachain-studio/tls.key
    minVersion: "1.2"

  # Nginx SSL configuration
  nginx:
    ssl:
      protocols: "TLSv1.2 TLSv1.3"
      ciphers: "ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384"
      prefer_server_ciphers: "on"
      session_cache: "shared:SSL:10m"
      session_timeout: "10m"
```

#### 3. Restart Services

```bash
sudo systemctl restart datachain-studio
```

## SSL Certificate Configuration (Legacy)

### Self-Signed Certificates

For development or internal use, you can create self-signed certificates:

```bash
# Generate private key
openssl genrsa -out studio.key 2048

# Generate certificate signing request
openssl req -new -key studio.key -out studio.csr \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=studio.yourcompany.com"

# Generate self-signed certificate
openssl x509 -req -days 365 -in studio.csr -signkey studio.key -out studio.crt

# Clean up CSR
rm studio.csr
```

### Certificate Authority (CA) Signed Certificates

For production use, obtain certificates from a trusted CA:

#### 1. Generate Certificate Signing Request

```bash
openssl genrsa -out studio.key 2048
openssl req -new -key studio.key -out studio.csr
```

#### 2. Submit CSR to Certificate Authority

Submit the CSR to your chosen CA (Let's Encrypt, DigiCert, etc.) and obtain the signed certificate.

#### 3. Install Certificates

Follow the same installation process as described in the TLS section above.

## Let's Encrypt Integration

### Automatic Certificate Management

For Kubernetes deployments, use cert-manager for automatic Let's Encrypt certificates:

#### 1. Install cert-manager

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

#### 2. Create ClusterIssuer

```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@yourcompany.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

#### 3. Configure Ingress for Auto-SSL

```yaml
ingress:
  enabled: true
  className: nginx

  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"

  tls:
    enabled: true
    secretName: datachain-studio-tls-auto
```

### Manual Let's Encrypt (Certbot)

For AMI deployments, use certbot for Let's Encrypt certificates:

```bash
# Install certbot
sudo apt update
sudo apt install certbot

# Obtain certificate (requires port 80 to be accessible)
sudo certbot certonly --standalone \
  -d studio.yourcompany.com \
  --email admin@yourcompany.com \
  --agree-tos \
  --no-eff-email

# Certificates will be available at:
# /etc/letsencrypt/live/studio.yourcompany.com/fullchain.pem
# /etc/letsencrypt/live/studio.yourcompany.com/privkey.pem
```

## Certificate Validation

### Testing SSL/TLS Configuration

```bash
# Test SSL certificate
openssl s_client -connect studio.yourcompany.com:443 -servername studio.yourcompany.com

# Check certificate expiration
echo | openssl s_client -connect studio.yourcompany.com:443 2>/dev/null | openssl x509 -dates -noout

# Test SSL configuration
curl -I https://studio.yourcompany.com

# Detailed SSL test (requires ssllabs-scan tool)
ssllabs-scan studio.yourcompany.com
```

### Certificate Chain Validation

```bash
# Verify certificate chain
openssl verify -CAfile ca-bundle.crt studio.crt

# Check certificate details
openssl x509 -in studio.crt -text -noout
```

## Security Hardening

### Advanced TLS Configuration

```yaml
# Advanced security configuration
global:
  tls:
    enabled: true
    minVersion: "1.2"
    maxVersion: "1.3"

    # Strong cipher suites only
    cipherSuites:
      - "TLS_AES_128_GCM_SHA256"
      - "TLS_AES_256_GCM_SHA384"
      - "TLS_CHACHA20_POLY1305_SHA256"
      - "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
      - "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"

    # HSTS configuration
    hsts:
      enabled: true
      maxAge: 31536000  # 1 year
      includeSubdomains: true
      preload: true

    # OCSP stapling
    ocsp:
      enabled: true
      cache: true
```

### Security Headers

```yaml
# Additional security headers
ingress:
  annotations:
    nginx.ingress.kubernetes.io/configuration-snippet: |
      add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
      add_header X-Content-Type-Options "nosniff" always;
      add_header X-Frame-Options "DENY" always;
      add_header X-XSS-Protection "1; mode=block" always;
      add_header Referrer-Policy "strict-origin-when-cross-origin" always;
```

## Certificate Renewal

### Automated Renewal

For Let's Encrypt certificates, set up automatic renewal:

```bash
# Add cron job for certificate renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

### Renewal Monitoring

```bash
# Check certificate expiration
openssl x509 -in /path/to/certificate.crt -noout -dates

# Set up expiration monitoring
#!/bin/bash
CERT_FILE="/etc/ssl/datachain-studio/tls.crt"
EXPIRY_DATE=$(openssl x509 -in $CERT_FILE -noout -enddate | cut -d= -f2)
EXPIRY_EPOCH=$(date -d "$EXPIRY_DATE" +%s)
CURRENT_EPOCH=$(date +%s)
DAYS_LEFT=$(( ($EXPIRY_EPOCH - $CURRENT_EPOCH) / 86400 ))

if [ $DAYS_LEFT -lt 30 ]; then
    echo "Certificate expires in $DAYS_LEFT days!"
    # Send alert
fi
```

## Troubleshooting

### Common SSL/TLS Issues

**Certificate not trusted:**
- Verify certificate chain is complete
- Check CA bundle includes intermediate certificates
- Ensure root CA is trusted by client systems

**TLS handshake failures:**
- Check cipher suite compatibility
- Verify TLS version support
- Review server and client configurations

**Mixed content warnings:**
- Ensure all resources load over HTTPS
- Update HTTP references to use HTTPS
- Configure proper redirects

### Debug Commands

```bash
# Check certificate chain
openssl s_client -connect studio.yourcompany.com:443 -showcerts

# Test specific TLS version
openssl s_client -connect studio.yourcompany.com:443 -tls1_2

# Check cipher suites
nmap --script ssl-enum-ciphers -p 443 studio.yourcompany.com

# Monitor SSL logs
kubectl logs -f deployment/datachain-studio-frontend -n datachain-studio | grep -i ssl
```

## Next Steps

- Configure [CA certificates](ca-certificates.md) for custom certificate authorities
- Set up [Git forge integrations](git-forges/index.md)
- Review [troubleshooting guides](../troubleshooting/index.md)
- Learn about [upgrading procedures](../upgrading/index.md)
