# Installation

DataChain Studio supports multiple installation methods to accommodate different infrastructure requirements and preferences.

## Installation Methods

### AWS AMI
Deploy DataChain Studio using a pre-configured Amazon Machine Image (AMI) for quick setup on AWS.

**Best for:**
- Quick proof-of-concept deployments
- Single-instance installations
- Teams familiar with AWS EC2

[Get started with AWS AMI installation →](aws-ami.md)

### Kubernetes (Helm)
Deploy DataChain Studio on Kubernetes using Helm charts for scalable, production-ready installations.

**Best for:**
- Production deployments
- Scalable installations
- Teams with Kubernetes expertise
- Multi-environment deployments

[Get started with Kubernetes installation →](k8s-helm.md)

## Choosing Your Installation Method

| Feature | AWS AMI | Kubernetes (Helm) |
|---------|---------|-------------------|
| **Setup Complexity** | Low | Medium |
| **Scalability** | Limited | High |
| **High Availability** | Manual setup | Built-in |
| **Resource Management** | Manual | Automatic |
| **Monitoring** | Basic | Advanced |
| **Backup/Recovery** | Manual | Automated |
| **Multi-environment** | Limited | Excellent |

## Prerequisites

Before installing DataChain Studio, ensure you have:

### Infrastructure Requirements
- **Compute Resources**: See [system requirements](../index.md#system-requirements)
- **Network Access**: Internet connectivity for downloading dependencies
- **SSL Certificates**: Valid certificates for secure HTTPS communication
- **Domain Name**: Custom domain for accessing Studio

### Dependencies
- **Database**: PostgreSQL 12+ for application data
- **Object Storage**: S3, GCS, or Azure Blob Storage for data and artifacts
- **Redis**: For caching and session management (optional but recommended)

### Access Requirements
- **Administrative Access**: Full access to your deployment environment
- **DNS Control**: Ability to configure DNS records for your domain
- **Certificate Management**: Access to SSL certificate management

## Planning Your Installation

### 1. Choose Installation Method
Based on your requirements and expertise, select either:
- [AWS AMI](aws-ami.md) for simple, single-instance deployments
- [Kubernetes Helm](k8s-helm.md) for scalable, production deployments

### 2. Plan Your Infrastructure
- **Networking**: VPC, subnets, security groups, load balancers
- **Storage**: Database sizing, object storage configuration
- **Security**: IAM roles, security groups, access policies
- **Monitoring**: Logging, metrics, alerting setup

### 3. Prepare Configuration
- **Environment Variables**: Database URLs, storage credentials
- **SSL Certificates**: Certificate files and private keys
- **Authentication**: SSO configuration, user management
- **Feature Flags**: Enable/disable specific functionality

## Installation Process Overview

1. **Environment Setup**: Prepare your infrastructure and dependencies
2. **Installation**: Deploy DataChain Studio using your chosen method
3. **Initial Configuration**: Configure basic settings and authentication
4. **Verification**: Test the installation and verify functionality
5. **Post-installation**: Set up monitoring, backups, and maintenance

## Getting Support

For installation support:

- **Documentation**: Follow the detailed guides for your chosen method
- **Enterprise Support**: Contact our support team for assistance
- **Community**: Join our community for peer support and discussions

## Next Steps

Choose your installation method:

- **[AWS AMI Installation](aws-ami.md)** - Quick single-instance deployment
- **[Kubernetes Installation](k8s-helm.md)** - Scalable production deployment

After installation, proceed to:

- **[Configuration](../configuration/index.md)** - Configure Studio for your environment
- **[Upgrading](../upgrading/index.md)** - Learn about the upgrade process
- **[Troubleshooting](../troubleshooting/index.md)** - Resolve common issues
