# AWS AMI Installation

## Prerequisites

### DataChain Studio Images

The DataChain Studio machine image (AMI) and access to the DataChain Studio Docker images need to be provided by the DataChain team to enable the installation.

### DNS

Create a DNS record pointing to the IP address of the EC2 instance. This hostname will be used for DataChain Studio.

## Installation

1. Open the AWS Console

2. Navigate to EC2 -> Instances

3. Click **Launch instances**

4. Provide a name for your EC2 instance

5. Select **studio-selfhosted** from the AMI catalog

6. Select an appropriate instance type.
   - Minimum requirements: 16 GB RAM, 4 vCPUs
   - Recommended requirements: 32 GB RAM, 8 vCPUs

7. To enable SSH connections to the instance, select an existing key pair to use or create a new one. We recommend ED25519 keys.

8. In the network settings, use either the default VPC or change it to a desired one. Under the Firewall setting, create a new security group with SSH, HTTP, and HTTPS access or use an existing one with the same level of access.

!!! warning
    It's important to ensure that your VPC has connectivity to your Git forge provider (GitHub.com, GitLab.com, Bitbucket.org) and your storage provider (S3, GCS, etc.), to ensure DataChain Studio can access these resources.

9. Configure storage:
   - Use at least 100 GB of EBS storage
   - Consider using GP3 for better performance
   - Enable encryption for security

10. Launch the instance

## Configuration

Once the instance is running, you need to configure DataChain Studio:

### Initial Setup

1. SSH into the instance:
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

2. Navigate to the configuration directory:
   ```bash
   cd /opt/datachain-studio
   ```

3. Copy the example configuration:
   ```bash
   sudo cp config.example.yml config.yml
   ```

4. Edit the configuration file:
   ```bash
   sudo nano config.yml
   ```

### Configuration Parameters

Edit the following parameters in `config.yml`:

```yaml
# Basic configuration
domain: your-studio-domain.com
ssl:
  enabled: true
  cert_path: /etc/ssl/certs/studio.crt
  key_path: /etc/ssl/private/studio.key

# Database configuration
database:
  host: localhost
  port: 5432
  name: datachain_studio
  user: studio
  password: your-secure-password

# Storage configuration
storage:
  type: s3
  bucket: your-studio-bucket
  region: us-east-1
  access_key: your-access-key
  secret_key: your-secret-key

# Git forge configuration
git:
  github:
    enabled: true
    app_id: your-github-app-id
    private_key_path: /etc/studio/github-private-key.pem
  gitlab:
    enabled: true
    url: https://gitlab.com
    app_id: your-gitlab-app-id
    secret: your-gitlab-secret
```

### SSL Configuration

1. Upload your SSL certificate and private key to the instance
2. Update the paths in the configuration file
3. Ensure proper file permissions:
   ```bash
   sudo chmod 600 /etc/ssl/private/studio.key
   sudo chmod 644 /etc/ssl/certs/studio.crt
   ```

### Start Services

1. Start DataChain Studio services:
   ```bash
   sudo systemctl enable datachain-studio
   sudo systemctl start datachain-studio
   ```

2. Check service status:
   ```bash
   sudo systemctl status datachain-studio
   ```

3. View logs:
   ```bash
   sudo journalctl -u datachain-studio -f
   ```

## Verification

1. Access DataChain Studio at `https://your-domain.com`
2. Check that all services are running:
   ```bash
   sudo docker ps
   ```
3. Verify database connectivity:
   ```bash
   sudo docker exec -it studio-db psql -U studio -d datachain_studio -c "SELECT version();"
   ```

## Security Considerations

### Network Security
- Use security groups to restrict access
- Enable VPC flow logs for monitoring
- Consider using AWS WAF for web application protection

### Data Security
- Enable EBS encryption
- Use IAM roles instead of access keys where possible
- Regularly rotate secrets and keys
- Enable CloudTrail for audit logging

### Backup Strategy
- Set up automated EBS snapshots
- Configure database backups
- Test restore procedures regularly

## Troubleshooting

### Common Issues

**Services won't start:**
- Check configuration file syntax
- Verify SSL certificate paths and permissions
- Check Docker service status

**Cannot access Studio:**
- Verify DNS resolution
- Check security group rules
- Confirm SSL certificate validity

**Database connection issues:**
- Check database service status
- Verify connection parameters
- Check database logs

### Getting Help

- Check service logs: `sudo journalctl -u datachain-studio`
- Review configuration: `sudo cat /opt/datachain-studio/config.yml`
- Contact support with instance details and error messages

## Next Steps

- [Configure additional settings](../configuration/index.md)
- [Set up Git forge connections](../configuration/git-forges/index.md)
- [Configure SSL/TLS](../configuration/ssl-tls.md)
- [Learn about upgrading](../upgrading/index.md)
