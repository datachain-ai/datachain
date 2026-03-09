# Remove a model, version, or stage assignment

When you remove (deprecate) a model, deregister a version or unassign a stage,
DataChain Studio creates Git tags that indicate the action and saves the tags in
your Git repository.

These actions can be found in the 3-dot menu next to the model name in the
models dashboard (see the section highlighted in purple below).

![](https://static.iterative.ai/img/studio/model-registry-undo-actions.png)

These actions are also available in the model details page:

- `Deprecate model` action is present in the 3-dot menu next to the model name.

<p align="center">
<img src="https://static.iterative.ai/img/studio/model-registry-deprecate.png" alt="Deprecate model" width="400px"/>
</p>

- `Deregister version` button is present next to the version dropdown.

<p align="center">
<img src="https://static.iterative.ai/img/studio/model-registry-deregister.png" alt="Deprecate model" width="400px"/>
</p>

- Click on the relevant stage assignment pill in the `Stages` section to reveal
  the `Unassign stage` menu item.

<p align="center">
<img src="https://static.iterative.ai/img/studio/model-registry-unassign.png" alt="Deprecate model" width="400px"/>
</p>

<admon type="tip">
To remove all of a project's models from DataChain Studio without deprecating them, you can simply delete the project.
</admon>

<admon type="info">
You can also remove a model version or stage assignment by removing the corresponding Git tag directly from your Git repository. But this destroys the audit trail of the original version registration or stage assignment action.
</admon>
