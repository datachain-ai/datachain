# Self-hosting DataChain Studio

DataChain Studio Enterprise users can host DataChain Studio on their own infrastructure (on-premises) or in their cloud accounts.

Please note that our support is needed to make DataChain Studio's cloud/Docker images available to you to enable installation.

Below are the supported installation methods:

- [AMI (AWS)](installation/aws-ami.md)
- [Kubernetes (Helm)](installation/k8s-helm.md)

## System requirements

### VM (AMI)

Recommended requirements:

- 32 GB RAM
- 4 vCPUs
- 100 GB disk space

### Helm

We recommend deploying DataChain Studio in an auto-scaling node group with a minimum of 2 nodes.

Each node should have at least 16 GB of RAM and 4 vCPUs.

Additionally, you'll need 100 GB of block storage for DataChain Studio's `PersistentVolume`

## DataChain Studio's architecture

DataChain Studio is composed of four pieces:

- **Frontend Server**: Renders the web interface
- **Backend Server**: Stores all user information and handles API requests
- **Celery Beat**: Coordinates background tasks and job scheduling
- **Celery Worker**: Processes background tasks and data processing jobs

## Key Features of Self-hosted Studio

### Security and Privacy
- **Data Sovereignty**: Keep all data within your infrastructure
- **Network Isolation**: Deploy in private networks and VPCs
- **Access Control**: Integrate with your existing authentication systems
- **Compliance**: Meet regulatory requirements for data handling

### Customization
- **Custom Domains**: Use your own domain names and SSL certificates
- **Branding**: Customize the interface with your organization's branding
- **Resource Management**: Control computational resources and scaling
- **Integration**: Connect with internal systems and tools

### Administration
- **User Management**: Centralized user and team administration
- **Monitoring**: Built-in monitoring and alerting capabilities
- **Backup**: Automated backup and disaster recovery options
- **Updates**: Controlled update process for new features

## Getting Started

1. **[Installation](installation/index.md)** - Choose your deployment method
2. **[Configuration](configuration/index.md)** - Configure Studio for your environment
3. **[Upgrading](upgrading/index.md)** - Keep your installation up to date
4. **[Troubleshooting](troubleshooting/index.md)** - Resolve common issues

## Support

For self-hosting support:

- Contact our enterprise support team
- Review the [troubleshooting guide](troubleshooting/index.md)
- Check the [configuration documentation](configuration/index.md)

## Prerequisites

Before installing DataChain Studio:

- **Enterprise License**: Self-hosting requires an Enterprise license
- **Infrastructure Access**: Administrative access to your deployment environment
- **SSL Certificates**: Valid SSL certificates for secure communication
- **Database**: PostgreSQL database for storing application data
- **Storage**: Object storage for data and artifacts (S3, GCS, Azure Blob)

## Next Steps

- Review [installation options](installation/index.md)
- Plan your [configuration settings](configuration/index.md)
- Set up [monitoring and alerting](troubleshooting/index.md)
- Configure [authentication and access control](configuration/index.md)
