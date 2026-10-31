# Environment Variables

List of environment variables used to configure DataChain behavior.

### Core Configuration

- `DATACHAIN_ROOT_DIR` – Specifies the root directory where DataChain will create the `.datachain` folder to store its internal data. (default: the current working directory).
- `DATACHAIN_SYSTEM_CONFIG_DIR` – Overrides the system-wide configuration directory (default depends on the platform).
- `DATACHAIN_GLOBAL_CONFIG_DIR` – Overrides the user's global configuration directory (default depends on the platform).
- `DATACHAIN_NO_ANALYTICS` – Disables telemetry.
- `DATACHAIN_RANGE_LISTING` – Set to `0` or `false` to list each bucket in one sequential pass, without parallel key ranges or listing processes. Use it for an S3-compatible server that skips keys after a start key, such as SeaweedFS ([seaweedfs#11802](https://github.com/seaweedfs/seaweedfs/issues/11802)). Cloudflare R2, which sorts keys differently, is detected and listed in one pass without it.
- `DATACHAIN_LISTING_REPLY_TIMEOUT` – Seconds a listing process may take on one page before the listing kills it and fetches the page itself (default: 300).

### Studio Integration

- `DATACHAIN_STUDIO_URL` – Custom Studio URL.
- `DATACHAIN_STUDIO_TOKEN` – Authentication token for Studio.
- `DATACHAIN_STUDIO_TEAM` – Studio team name.

### Namespaces and projects
- `DATACHAIN_NAMESPACE` – Namespace name to use as default.
- `DATACHAIN_PROJECT` – Project name or combination of namespace name and project name separated by `.` to use as default, example: `DATACHAIN_PROJECT=dev.analytics`

### Checkpoints
.- `DATACHAIN_IGNORE_CHECKPOINTS` – When set to `1` or `true`, ignores all existing checkpoints and runs the script from scratch, forcing DataChain to recreate all datasets.

Note: Some environment variables are used internally and may not be documented here. For the most up-to-date list, refer to the source code.
