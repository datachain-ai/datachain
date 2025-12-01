# Register a model version

New model versions can signify an important, published or released iteration. To
register version, you first need to
[add a model to the model registry](add-a-model.md).

To register a new version of a model, DataChain Studio uses GTO to create an
annotated [Git tag][git tag] with the specified version number.

You can [write CI/CD actions][CI/CD] that can actually build and publish models
(for example, build Docker image with the model and publish it to a Docker
Registry) upon the creation of a new Git tag for version registration. For that,
you can leverage any ML model deployment tool, such as [MLEM].

You can register a version in any of the following ways:

1. Use GTO CLI or API. An example would be
   `gto register pool-segmentation --version v0.0.1`, assuming
   `dvc.yaml` with the model annotation is located in the root of the repo. If
   not, you should append its parent directory to the model's name like this:
   `gto register cv:pool-segmentation --version v0.0.1` (here, `cv`
   is the parent directory).
2. To register versions using DataChain Studio, watch this tutorial video or read on
   below.

https://www.youtube.com/watch?v=eA70puzOp1o

1. On the models dashboard, open the 3-dot menu for the model whose version you
   want to register. Then, click on `Register new version`. The registration
   action can also be initiated from the model details page or from the related
   project's experiment table - look for the `Register version` button or icon.

2. Select the Git commit which corresponds to the new version of your model. If
   the desired commit does not appear in the commit picker, type in the
   40-character sha-1 hash of the commit.
3. Enter a version name. Version names must start with the letter `v` and should
   follow the [SemVer] format after the letter `v`. Below are some examples of
   valid and invalid version names:
   - Valid: v0.0.1, v1.0.0, v12.5.7
   - Invalid: 0.0.1 (missing `v` in the beginning), v1.0 (missing the patch
     segment of the [Semver], v1.0.new (using an invalid value `new` as the
     patch number).

4. Optionally, provide a Git tag message.
5. Click on `Register version`.

Once the action is successful, the newly registered version will show up in the
`Latest version` column of the models dashboard. Note that this will happen only
if the newly registered version is the greatest semantic version for your model.
For example, if your model already had v3.0.0 registered, then if you register a
smaller version (e.g., v2.0.0), then the new version will not appear in the
`Latest version` column.

If you open the model details page, the newly registered version will be
available in the model `History` section as well as in the versions drop down.

If you go to your Git repository, you will see that a new Git tag referencing
the selected commit has been created, representing the new version.

[git tag]: https://git-scm.com/docs/git-tag
[semver]: https://semver.org/
[CI/CD]: use-models.md#deploying-and-publishing-models-in-cicd
[MLEM]: https://mlem.ai/
