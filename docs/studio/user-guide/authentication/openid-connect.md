# OpenID Connect (OIDC)

DataChain Studio can use OpenID Connect to access cloud resources securely, without requiring manual configuration of static credentials.

To use OIDC, first follow the [cloud configuration](#cloud-configuration) instructions and then the [Studio configuration](#studio-configuration) instructions.

## Cloud configuration

### Generic configuration details

- OpenID Connect Discovery URL: https://studio.datachain.ai/api/.well-known/openid-configuration

- Subject claim format: `credentials:{owner}/{name}` where `{owner}` is the name of the DataChain Studio **user** or **team** owning the credentials, and `{name}` is the name of the DataChain Studio [credentials](../account-management.md#cloud-credentials).

### Terraform examples

The following Terraform examples illustrate how to configure the supported cloud providers, granting DataChain Studio access to object storage resources through OpenID Connect. Update the fields as described below and then apply the Terraform configuration. Make note of the outputs of `terraform apply`, since you will need to enter those for [Studio configuration](#studio-configuration).

!!! tip
    Replace the sample `credentials:example-team/example-credentials` subject claim condition. Replace `example-team` with the Studio **user** or **team** owning the credentials, and replace `example-credentials` with any name you want to use for those credentials. This name must match what you enter during [Studio configuration](#studio-configuration).

#### Amazon Web Services

```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }

  required_version = ">= 1.2.0"
}

provider "aws" {
  region = "us-east-1"
}

locals {
  provider  = "studio.datachain.ai/api"
  condition = "credentials:example-team/example-credentials"
}

data "tls_certificate" "studio" {
  url = "https://${local.provider}"
}

data "aws_iam_policy_document" "studio_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.studio.arn]
    }

    condition {
      test     = "StringLike"
      variable = "${aws_iam_openid_connect_provider.studio.url}:sub"
      values   = [local.condition]
    }
  }
}

data "aws_iam_policy_document" "studio" {
  statement {
    actions   = ["s3:*"]
    resources = ["*"]
  }
}

resource "aws_iam_openid_connect_provider" "studio" {
  url             = data.tls_certificate.studio.url
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.studio.certificates.0.sha1_fingerprint]
}

resource "aws_iam_role" "studio" {
  max_session_duration = 12 * 60 * 60 # 12 hours
  assume_role_policy   = data.aws_iam_policy_document.studio_assume_role.json

  inline_policy {
    name   = "studio"
    policy = data.aws_iam_policy_document.studio.json
  }
}

output "role_arn" {
  value = aws_iam_role.studio.arn
}
```

#### Google Cloud

```hcl
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "5.13.0"
    }
  }
}

provider "google" {
  project = "your-project-id"
  region  = "us-central1"
}

locals {
  provider  = "studio.datachain.ai/api"
  condition = "credentials:example-team/example-credentials"
}

data "google_project" "current" {}

resource "google_iam_workload_identity_pool" "studio" {
  workload_identity_pool_id = "datachain-studio"
  display_name             = "DataChain Studio"
}

resource "google_iam_workload_identity_pool_provider" "studio" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.studio.workload_identity_pool_id
  workload_identity_pool_provider_id = "datachain-studio"
  display_name                       = "DataChain Studio"

  attribute_mapping = {
    "google.subject" = "assertion.sub"
  }

  attribute_condition = "assertion.sub == '${local.condition}'"

  oidc {
    issuer_uri = "https://${local.provider}"
  }
}

resource "google_service_account" "studio" {
  account_id   = "datachain-studio"
  display_name = "DataChain Studio"
}

resource "google_service_account_iam_binding" "studio" {
  service_account_id = google_service_account.studio.name
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.studio.name}/*"
  ]
}

output "google_service_account_email" {
  value = google_service_account.studio.email
}

output "google_workload_identity_provider" {
  value = google_iam_workload_identity_pool_provider.studio.name
}
```

## Studio configuration

[Create new credentials](../account-management.md#cloud-credentials) and configure them as follows:

1. Choose an adequate OIDC variant on the provider field; e.g. _Amazon Web Services (OIDC)_.
2. Enter the name for the credentials. This must match the name used during [cloud configuration](#cloud-configuration).
3. Fill the provider-specific fields with the outputs from `terraform apply`.

## Troubleshooting

If you encounter issues with OIDC setup:

1. Verify that the subject claim format matches exactly
2. Check that the Terraform configuration has been applied successfully
3. Ensure that the credential names match between cloud configuration and Studio configuration
4. Verify that the workload identity pool and provider are configured correctly

For more help, see our [troubleshooting guide](../troubleshooting.md).
