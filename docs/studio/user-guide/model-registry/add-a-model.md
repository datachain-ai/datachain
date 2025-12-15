# Add a model

You can add models from any ML project to the model registry. To add a model,
DataChain Studio creates an annotation for it in a `dvc.yaml` file in your Git
repository. You can add a model in any of the following ways:

1. Log your model during the training process using dvclive by calling
   `live.log_artifact(path, type="model")` log_artifact method.
2. Edit `dvc.yaml` directly and add your model to `artifacts` section.
3. Use the DataChain Studio interface (watch this tutorial video or read on below).

https://www.youtube.com/watch?v=szzv4ZXmYAs

1. Click on `Add a model`.

2. Select a [connected project] to which you want to add the model.

   <admon>

   If your model file or the `.dvc` file for your model already exists in a Git
   repo, select that repo. If your model file resides in remote storage (S3,
   GCS, etc.), select the Git repo where you want to add the model.

   </admon>

3. Enter the path to `dvc.yaml` the model will be added to. Adding your model to
   non-root `dvc.yaml` can be helpful if you develop this ML model in a specific
   subfolder or if this repo is a monorepo.

4. Enter the path of the model file as follows:
   - If the model file is in the Git repository or is in the cloud but is
     tracked by DVC (acquired by lakeFS), enter the relative path of the model
     (from the repository root).
   - Otherwise, enter the URL to the model file in the cloud. DataChain Studio will
     ask you for the repository path where the dvc reference to the model should
     be saved.

5. Provide labels for your model. For example, if your model is about reviewing
   sentiment analysis using natural language processing, one of the labels may
   be `nlp` or `sentiment_analysis`.

6. Optionally, add a brief description for your model.

7. Enter a Git commit message. Then, select the branch to commit to. You can
   commit to either the base branch or a new branch. DataChain Studio will commit the
   changes to the selected branch. If you commit to a new branch, DataChain Studio
   will also create a Git pull request from the new branch to the base branch.

8. Now, click on `Commit changes`.

At this point, the new model appears in the models dashboard.

In your Git repository, you will find that an entry for the new model has been
created in the `dvc.yaml` that was specified. If you had committed to a new
branch, a new pull request (or merge request in the case of GitLab) will also
have been created to merge the new branch into the base branch.

If you had added a model from a cloud storage, the following will also happen
before the commit is created:

- If the repository does not contain DVC, DataChain Studio will run `dvc init`. It is
  needed to version the model in the git repository.
- If the specified directory does not exist yet, it will be created.
- DataChain Studio will import the model to the repository by executing
  `dvc import-url <remote_path> <directory_path>/<filename from remote_path> --no-exec`.

[connected project]: ../experiments/create-a-project.md
