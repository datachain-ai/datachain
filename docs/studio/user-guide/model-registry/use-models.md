# Use models

Whether you need to download your models to use them, or you're looking to set
up some automation in CI/CD to deploy them, DataChain Studio provides these
capabilities.

## Download models

If your model file is tracked by DVC (acquired by lakeFS), you can download any
of its registered versions using the DataChain Studio REST API,
`dvc artifacts get`, or DVC Python API.

Prerequisites:

- Model stored with DVC with S3, Azure, http or https remote.
- The DataChain Studio project you like to download your model from needs access to
  your remote storage credentials.
- Access to your [DataChain Studio client access token] with Model registry operations
  scope.

Without these prerequisites, you can still download a model artifact with DVC.
However, it can be easier to use the DataChain Studio API since you only need to have
the Studio access token. You do not need direct access to your remote storage or
Git repository, and you do not need to install DVC.

[DataChain Studio client access token]: ../account-management.md#client-access-tokens

You can download the files that make up your model directly from DataChain Studio.
Head to the model details page of the model you would like to download and click
`Access Model`. Here, you find different ways to download your model.

=== "CLI (DVC)"

Use the `dvc artifacts get` command to download an artifact by name. Learn more
on the command reference page for `dvc artifacts get`.

=== "cURL / Python"

Directly call the Studio REST API from your terminal
using `cURL` or in your `Python` code.

=== "Direct Download"

Here you can generate download links for your model files. After generation,
these download links are valid for 1 hour. You can click the link to directly
download the file.
## Deploying and publishing models in CI/CD

A popular deployment option is to **use CI/CD pipelines triggered by new Git
tags to publish or deploy a new model version**. Since GTO registers versions
and assigns stages by creating Git tags, you can set up a CI/CD pipeline to be
triggered when the tags are pushed to the repository.

You can use [the GTO GitHub Action](https://github.com/iterative/gto-action)
that interprets a Git tag to find out the model's version and stage assignment
(if any), reads annotation details such as `path`, `type` and `description`, and
downloads the model binaries if needed.

For help building an end-to-flow from model training to deployment using the
DVC model registry, refer to the
[tutorial on automating model deployment to Sagemaker](https://iterative.ai/blog/sagemaker-model-deployment).
[Here](https://github.com/iterative/example-get-started-experiments/blob/main/.github/workflows/deploy-model.yml)
is the complete workflow script.
