# Explore ML Experiments

The projects dashboard in DataChain Studio contains all your projects. Click on a
project name to open the project table, which contains:

- [Git history and live experiments](#git-history-and-live-experiments) of the
  project
- [Display preferences](#display-preferences)
- Buttons to
  [visualize and compare experiments](#visualize-and-compare-experiments).
- Button to [export project data](#export-project-data).

## Git history and live experiments

Branches and commits in your Git repository are displayed along with the
corresponding models, metrics, hyperparameters, and DVC-tracked files
(acquired by lakeFS).

Experiments that you push using the `dvc exp push` command as well as any live
experiments that you send using [DVCLive] are displayed in a special experiment
row nested under the parent Git commit. More details of how live experiments are
displayed can be found in the
[live metrics and plots guide](live-metrics-and-plots.md).

To manually check for updates in your repository, use the `Reload` button 🔄
located above the project table.

![](https://static.iterative.ai/img/studio/view_components_1.gif)

<admon type="tip">

One simple way to briefly describe your experiments is to use meaningful commit
messages.

</admon>

### Nested branches

When a Git branch (e.g., `feature-branch-1`) is created from another branch
(e.g., `main`), two possibilities exist:

- `feature-branch-1` is still active (contains commits that are not present in
  `main`). This can happen if the user has pushed new commits to this branch and
  - either hasn't merged it into `main` yet
  - or has merged it into `main` but has continued to push more new commits to
    it after the merger.

  Since the branch now contains new unique commits, the project table will
  display both `main` and `feature-branch-1` separately. `feature-branch-1` will
  show the new commits that are not part of `main` while all the merged commits
  will be shown inside `main`.

- `feature-branch-1` is inactive (does not contain any commits that are not
  present in `main`). This can happen in two cases:
  - if the user has not pushed any new commits to `feature-branch-1`
  - if the user has merged `feature-branch-1` into `main` and has not pushed any
    new commits to it after the merger.

  Since the branch does not contain any new unique commits, DataChain Studio considers
  `feature-branch-1` as **"nested"** within `main` and does not display it as a
  separate branch. This helps to keep the project table concise and reduce
  clutter that can accumulate over time when inactive branches are not cleaned
  from the Git repository. After all, those inactive branches usually carry no
  new information for the purpose of managing experiments. If you would like to
  display all commits of such an inactive branch, use the
  [`Commits on branch = feature-branch-1` display filter](#filters).

## Display preferences

The table contains buttons to specify filters and other preferences regarding
which commits and columns to display.

### Filters:

Click on the `Filters` button to specify which rows you want to show in the
project table.

![Project filters](https://static.iterative.ai/img/studio/project_filters.png)

There are two types of filters:

- **Quick filters** (highlighted in orange above): Use the quick filter buttons
  to
  - Show only DVC experiments
  - Show only selected experiments
  - Toggle hidden commits (include or exclude hidden commits in the project
    table)

- **Custom filters** (highlighted in purple above): Filter commits by one or
  more of the following fields:
  - Column values (values of metrics, hyperparameters, etc.) and their deltas
  - Git related fields such as Git branch, commit message, tag and author

    <admon type="info">

    The `Branch` filter displays only the specified branch and its commits.

    On the other hand, the `Commits on branch` filter will also display branches
    [inside which the specified branch is nested](#nested-branches).

    </admon>

    <details>

    ### More details on nested branches

    When a Git branch is nested inside another branch, the project table
    [does not display the nested branch](#nested-branches). If
    `feature-branch-1` is nested within `main`, `feature-branch-1` is NOT
    displayed in the project table even if you apply the
    `Branch = feature-brach-1` filter.

    In this case, if you would like to filter for commits in `feature-branch-1`,
    you should use the `Commits on branch = feature-branch-1` filter. This will
    display the `main` branch with commits that were merged from
    `feature-branch-1` into `main`. A hint is present to indicate that even
    though the commits appear inside `main`, they are part of the nested branch
    `feature-branch-1`.

    ![Result of commits on branch filter](https://static.iterative.ai/img/studio/commits_on_branch_filter.png)

    </details>

  - The `Custom filters` can be un-applied without deletion, allowing you to
    create the filters once and toggle them on and off as needed.

    <video width="99%" height="540" autoplay loop muted>
      <source src="https://static.iterative.ai/img/studio/project-custom-filters.mp4" type="video/mp4">
    </video>

### Columns:

Select the columns you want to display and hide the rest.
![Showing and hiding columns](https://static.iterative.ai/img/studio/show_hide_columns.gif)

If your project is missing some required columns or includes columns that you do
not want, refer to the [troubleshooting guide](../troubleshooting.md) for more
information on managing project columns and settings.

To reorder the columns, click and drag them in the table or from the Columns
dropdown.
![Showing and hiding columns](https://static.iterative.ai/img/studio/reorder_columns.gif)

**Columns menu and goals:** Click on the column header to open a context menu
with actions such as sorting and filtering the project table by the column's
values.

For metrics, you can also specify goals, which indicate whether an increase or a
decrease in the metric's value is desirable. Once a goal is set, the metric's
values for all rows are compared against the value in the baseline row. Values
that are better (higher or lower, depending on the goal) than that in the
baseline row are highlighted in green, with the best one shown with a green
border. Values that are worse than that in the baseline row are marked in pink.

![Columns menu and goals](https://static.iterative.ai/img/studio/columns_menu_and_goals.gif)

<admon type="info">

To change the baseline row in your project, use the 3-dot menu of the row which
you want to set as the new baseline.

![Set baseline row](https://static.iterative.ai/img/studio/set-baseline-row.gif)

</admon>

### Hide commits:

Commits can be hidden from the project table in the following ways:

- **DataChain Studio auto-hides irrelevant commits:** DataChain Studio identifies commits
  where metrics, files and hyperparameters did not change and hides them
  automatically.
- **DataChain Studio auto-hides commits that contain `[skip studio]` in the commit
  message:** This is particularly useful if your workflow creates multiple
  commits per experiment and you would like to hide all those commits except the
  final one.

  For example, suppose you create a Git commit with hyper-parameter changes for
  running a new experiment, and your training CI job creates a new Git commit
  with the experiment results (metrics and plots). You may want to hide the
  first commit and only display the second commit, which has the new values for
  the hyper-parameters as well as experiment results. For this, you can use the
  string `[skip studio]` in the commit message of the first commit.

- **Hide commits and branches manually:** This can be useful if there are
  commits that do not add much value in your project. To hide a commit or
  branch, click on the 3-dot menu next to the commit or branch name and click on
  `Hide commit` or `Hide branch`.

  ![Hide commit](https://static.iterative.ai/img/studio/hide_commit.png)

- **Unhide commits:** You can unhide commits as needed, so that you don't lose
  any experimentation history. To display all hidden commits, click on the
  `Show hidden commits` toggle (refer [filters](#filters)). This will display
  all hidden commits, with a `hidden` (closed eye) indicator.

  ![Hidden commit indicator](https://static.iterative.ai/img/studio/hidden_commit_indicator.png)

  To unhide any commit, click on the 3-dot menu for that commit and click on
  `Show commit`.

  ![Show hidden commit](https://static.iterative.ai/img/studio/show_hidden_commit.png)

### Delta mode

For metrics, models and files columns with numeric values, you can display
either the absolute values or their delta (difference) from the baseline row. To
toggle between these two options, use the `Delta mode` button.

![Delta mode](https://static.iterative.ai/img/studio/delta_mode.png)

### Save changes:

Whenever you make any changes to your project's columns, commits or filters, a
notification to save or discard your changes is displayed at the top of the
project table. Saved changes remain intact even after you log out of DataChain Studio
and log back in later.

![Save or discard changes](https://static.iterative.ai/img/studio/save_discard_changes.png)

## Visualize and compare experiments

Use the following buttons to visualize and compare experiments:

- **Plots:** Open the `Plots` pane and
  [display plots](visualize-and-compare.md#display-plots-and-images) for the
  selected commits.
- **Trends:** [Generate trend charts](visualize-and-compare.md#generate-trend-charts)
  to see how the metrics have changed over time.
- **Compare:** [Compare experiments](visualize-and-compare.md#compare-experiments)
  side by side.

These buttons appear above your project table as shown below.
![example export to csv](https://static.iterative.ai/img/studio/project_action_buttons_big_screen.png)

On smaller screens, the buttons might appear without text labels, as shown
below.

![example export to csv](https://static.iterative.ai/img/studio/project_action_buttons_small_screen.png)

## Export project data

The button to export data from the project table to CSV is present next to the
[`Delta mode`](#delta-mode) button.

![export to csv](https://static.iterative.ai/img/studio/project_export_to_csv.png)

Below is an example of the downloaded CSV file.

![example export to csv](https://static.iterative.ai/img/studio/project_export_to_csv_example.png)

[DVCLive]: https://dvc.org/doc/dvclive
