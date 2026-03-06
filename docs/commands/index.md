
# Using DataChain Commands



DataChain is a command-line tool for wrangling unstructured AI data at scale. Use `datachain -h` to list all available commands.



## Typical DataChain Workflow



1.  **Authentication with Studio**


	- Use [`datachain auth login`](auth/login.md) to authenticate with Studio

	- Set your default team with [`datachain auth team`](auth/team.md)

	- View your token with [`datachain auth token`](auth/token.md)

	- Log out from Studio with [`datachain auth logout`](auth/logout.md)



2.  **Job Management**

	- Run jobs in Studio with [`datachain job run`](job/run.md)

	- Monitor job logs with [`datachain job logs`](job/logs.md)

	- Cancel running jobs with [`datachain job cancel`](job/cancel.md)

	- Check for the clusters available for jobs [`datachain job clusters`](job/clusters.md)


3.  **Maintenance**

	- Clean up temporary tables, failed versions, and outdated checkpoints with [`datachain gc`](gc.md)
