import asyncio
import logging
import os
import random
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import dateparser
import requests
import tabulate

from datachain.catalog import get_catalog
from datachain.config import Config, ConfigLevel
from datachain.data_storage.job import JobQueryType, JobStatus
from datachain.dataset import (
    QUERY_DATASET_PREFIX,
    parse_dataset_name,
)
from datachain.error import DataChainError
from datachain.remote.studio import StudioClient
from datachain.utils import flatten

logger = logging.getLogger("datachain")


def _require_studio_token() -> str:
    config = Config().read().get("studio", {})
    token = config.get("token")
    if not token:
        raise DataChainError(
            "Not logged in to Studio. Log in with 'datachain auth login'."
        )
    return token


if TYPE_CHECKING:
    from argparse import Namespace

    from datachain.catalog import Catalog

POST_LOGIN_MESSAGE = (
    "Once you've logged in, return here "
    "and you'll be ready to start using DataChain with Studio."
)
RECONNECT_MAX_ATTEMPTS = 15
RECONNECT_BACKOFF_BASE_SEC = 1
RECONNECT_BACKOFF_MAX_SEC = 60


def _print_help(args):
    print(
        f"Use 'datachain {args.command} --help' to see available options",
        file=sys.stderr,
    )
    return 1


def _dispatch(args, dispatch_map):
    if args.cmd is None:
        return _print_help(args)
    handler = dispatch_map.get(args.cmd)
    if not handler:
        raise DataChainError(f"Unknown command '{args.cmd}'.")
    return handler(args)


def process_jobs_args(args: "Namespace"):
    return _dispatch(
        args,
        {
            "run": lambda a: create_job(
                query_file=a.file,
                team_name=a.team,
                env_file=a.env_file,
                env=a.env,
                workers=a.workers,
                files=a.files,
                python_version=a.python_version,
                repository=a.repository,
                req=a.req,
                req_file=a.req_file,
                priority=a.priority,
                cluster=a.cluster,
                start_time=a.start_time,
                cron=a.cron,
                no_wait=a.no_wait,
                credentials_name=a.credentials_name,
                ignore_checkpoints=a.ignore_checkpoints,
                no_follow=a.no_follow,
            ),
            "cancel": lambda a: cancel_job(a.id, a.team),
            "logs": lambda a: show_job_logs(a.id, a.team),
            "ls": lambda a: list_jobs(a.status, a.team, a.limit),
            "clusters": lambda a: list_clusters(a.team),
        },
    )


def process_pipeline_args(args: "Namespace", catalog: "Catalog"):
    return _dispatch(
        args,
        {
            "create": lambda a: create_pipeline(catalog, a.datasets, a.team),
            "status": lambda a: get_pipeline_status(a.name, a.team),
            "list": lambda a: list_pipelines(a.team, a.status, a.limit, a.search),
            "pause": lambda a: pause_pipeline(a.name, a.team),
            "resume": lambda a: resume_pipeline(a.name, a.team),
            "remove-job": lambda a: remove_job_from_pipeline(
                name=a.name,
                job_id=a.job_id,
                team_name=a.team,
            ),
        },
    )


def process_auth_cli_args(args: "Namespace"):
    return _dispatch(
        args,
        {
            "login": login,
            "logout": lambda a: logout(a.local),
            "token": lambda a: token(),
            "team": set_team,
        },
    )


def _save_default_team(team_name: str, level: ConfigLevel):
    config = Config(level)
    with config.edit() as conf:
        studio_conf = conf.get("studio", {})
        studio_conf["team"] = team_name
        conf["studio"] = studio_conf

    return config.config_file()


def _print_current_team():
    config = Config().read().get("studio", {})
    team = config.get("team")
    if team:
        print(f"Default team is '{team}'")
        return 0
    raise DataChainError(
        "No default team set. Use `datachain auth team <team_name>` to set one."
    )


def set_team(args: "Namespace"):
    if args.team_name is None:
        return _print_current_team()
    level = ConfigLevel.LOCAL if args.local else ConfigLevel.GLOBAL
    file_path = _save_default_team(args.team_name, level)
    print(f"Set default team to '{args.team_name}' in {file_path}")


def _validate_no_existing_token(config: dict, hostname: str):
    if config.get("url", hostname) == hostname and "token" in config:
        raise DataChainError(
            "Token already exists. "
            "To login with a different token, "
            "logout using `datachain auth logout`."
        )


def _authenticate(
    name: str,
    hostname: str,
    scopes: list[str],
    team_names: list[str],
    expires_in_days: int,
    open_browser: bool,
):
    from dvc_studio_client.auth import StudioAuthError, get_access_token

    try:
        _, access_token = get_access_token(
            token_name=name,
            hostname=hostname,
            scopes=scopes,
            team_names=team_names,
            expires_in_days=expires_in_days,
            open_browser=open_browser,
            client_name="DataChain",
            post_login_message=POST_LOGIN_MESSAGE,
        )
    except StudioAuthError as exc:
        raise DataChainError(f"Failed to authenticate with Studio: {exc}") from exc
    except requests.HTTPError as exc:
        response = exc.response
        if response and response.status_code == 400:
            message = response.json().get("detail", "Unknown error")
            raise DataChainError(
                f"Failed to authenticate with Studio: {message}"
            ) from exc
        raise DataChainError(f"Failed to authenticate with Studio: {exc}") from exc
    return access_token


def _finalize_login(
    hostname: str,
    access_token: str,
    team_names: list[str],
    level: ConfigLevel,
    expires_in_days: int,
):
    config_path = save_config(hostname, access_token, level=level)
    print(f"Authentication complete. Saved token to {config_path}.")
    if team_names:
        print(f"Token is scoped to teams: {', '.join(team_names)}")
    print(f"Token will expire in {expires_in_days} days.")

    if team_names and len(team_names) == 1:
        file_path = _save_default_team(team_names[0], level)
        print(f"Set default team to '{team_names[0]}' in {file_path}")
    else:
        print("You can now use 'datachain auth team' to set the default team.")


def login(args: "Namespace"):
    from datachain.remote.studio import get_studio_url

    config = Config().read().get("studio", {})
    hostname = args.hostname or get_studio_url(config)
    _validate_no_existing_token(config, hostname)

    expires_in_days = args.expires_in if args.expires_in is not None else 365
    access_token = _authenticate(
        name=args.name,
        hostname=hostname,
        scopes=args.scopes,
        team_names=args.team,
        expires_in_days=expires_in_days,
        open_browser=not args.no_open,
    )

    level = ConfigLevel.LOCAL if args.local else ConfigLevel.GLOBAL
    _finalize_login(hostname, access_token, args.team, level, expires_in_days)
    return 0


def _revoke_token(studio_url: str, token: str):
    try:
        response = requests.post(
            f"{studio_url}/api/device-logout",
            headers={"Authorization": f"token {token}"},
            timeout=10,
        )
    except requests.RequestException as exc:
        raise DataChainError(
            "Could not reach Studio to revoke the token. Please try again later."
        ) from exc

    if response.status_code == 401:
        print(
            "Token was already revoked or is invalid on Studio.",
            file=sys.stderr,
        )
    elif not response.ok:
        raise DataChainError(
            f"Studio returned HTTP {response.status_code} while revoking "
            f"the token. Please try again later."
        )


def _clear_token_config(level: ConfigLevel):
    with Config(level).edit() as conf:
        del conf["studio"]["token"]


def logout(local: bool = False):
    from datachain.remote.studio import get_studio_url

    token = _require_studio_token()
    level = ConfigLevel.LOCAL if local else ConfigLevel.GLOBAL
    config = Config(level).read().get("studio", {})

    studio_url = get_studio_url(config)
    _revoke_token(studio_url, token)
    _clear_token_config(level)
    print("Logged out from Studio. (you can log back in with 'datachain auth login')")


def token():
    config = Config().read().get("studio", {})
    token = config.get("token")
    if not token:
        raise DataChainError(
            "Not logged in to Studio. Log in with 'datachain auth login'."
        )

    print(token)


def _ds_full_name(ds: dict) -> str:
    return f"{ds['project']['namespace']['name']}.{ds['project']['name']}.{ds['name']}"


def _fetch_datasets(team: str | None):
    client = StudioClient(team=team)
    response = client.ls_datasets()
    if not response.ok:
        raise DataChainError(response.message)
    return response.data


def _yield_dataset_versions(data: list[dict]):
    if not data:
        return
    for d in data:
        name = d.get("name")
        full_name = _ds_full_name(d)
        if name and name.startswith(QUERY_DATASET_PREFIX):
            continue
        for v in d.get("versions", []):
            version = v.get("version")
            yield (full_name, version)


def list_datasets(team: str | None = None, name: str | None = None):
    if name:
        yield from list_dataset_versions(team, name)
        return
    yield from _yield_dataset_versions(_fetch_datasets(team))


def list_dataset_versions(team: str | None = None, name: str = ""):
    namespace_name, project_name, name = parse_dataset_name(name)
    if not namespace_name or not project_name:
        raise DataChainError(f"Missing namespace or project form dataset name {name}")

    client = StudioClient(team=team)
    response = client.dataset_info(namespace_name, project_name, name)
    if not response.ok:
        raise DataChainError(response.message)

    if response.data:
        for v in response.data.get("versions", []):
            yield (name, v.get("version"))


def edit_studio_dataset(
    team_name: str | None,
    name: str,
    namespace: str,
    project: str,
    new_name: str | None = None,
    description: str | None = None,
    attrs: list[str] | None = None,
):
    client = StudioClient(team=team_name)
    response = client.edit_dataset(
        name, namespace, project, new_name, description, attrs
    )
    if not response.ok:
        raise DataChainError(response.message)

    print(f"Dataset '{name}' updated in Studio")


def remove_studio_dataset(
    team_name: str | None,
    name: str,
    namespace: str,
    project: str,
    version: str | None = None,
    force: bool | None = False,
):
    client = StudioClient(team=team_name)
    response = client.rm_dataset(name, namespace, project, version, force)
    if not response.ok:
        raise DataChainError(response.message)

    print(f"Dataset '{name}' removed from Studio")


def save_config(hostname, token, level=ConfigLevel.GLOBAL):
    config = Config(level)
    with config.edit() as conf:
        studio_conf = conf.get("studio", {})
        studio_conf["url"] = hostname
        studio_conf["token"] = token
        conf["studio"] = studio_conf

    return config.config_file()


def parse_start_time(start_time_str: str | None) -> str | None:
    if not start_time_str:
        return None

    parsed_datetime = dateparser.parse(start_time_str)

    if parsed_datetime is None:
        raise DataChainError(
            f"Could not parse datetime string: '{start_time_str}'. "
            f"Supported formats include: '2024-01-15 14:30:00', 'tomorrow 3pm', "
            f"'monday 9am', '2024-01-15T14:30:00Z', 'in 2 hours', etc."
        )

    # Convert to ISO format string
    return parsed_datetime.isoformat()


# Sync usage
async def _fetch_log_blob(blob_url: str, token: str, timeout: float) -> str:
    """Fetch log content from a blob URL asynchronously."""

    def _fetch():
        headers = {"Authorization": f"token {token}"}
        response = requests.get(blob_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text

    return await asyncio.to_thread(_fetch)


async def _show_log_blobs(log_blobs: list[str], client):
    for blob_url in log_blobs:
        try:
            log_content = await _fetch_log_blob(blob_url, client.token, client.timeout)
            if log_content:
                print(log_content, end="")
        except (requests.RequestException, OSError):
            print("\n>>>> Warning: Failed to fetch logs from studio")


def _get_job_status(client, job_id: str) -> str | None:
    try:
        response = client.get_jobs(job_id=job_id)
        if response.ok and response.data and len(response.data) > 0:
            return response.data[0].get("status")
    except (requests.RequestException, OSError, KeyError):
        logger.debug("Failed to get job status: %s", job_id)
    return None


def _print_reconnect_msg(sleep_sec: float) -> str:
    msg = f"\r>>>> WebSocket closed, reconnecting in {sleep_sec:.0f}s..."
    print(msg, end="", flush=True)
    return msg


def _clear_line(msg: str) -> str:
    if msg:
        print("\r" + " " * len(msg) + "\r", end="", flush=True)
    return ""


def _process_logs_message(
    logs: list, last_log_id: int, filter_up_to: int
) -> tuple[bool, int]:
    received = False
    for log in logs:
        log_id = log["id"]
        if log_id <= filter_up_to:
            continue
        last_log_id = max(last_log_id, log_id)
        received = True
        print(log["message"], end="")
    return received, last_log_id


async def _handle_ws_messages(
    client,
    job_id,
    no_follow,
    last_log_id,
    processed_statuses,
    log_blobs_processed,
    reconnect_msg,
):
    received_streaming_data = False
    session_start_id = last_log_id
    latest_status = None
    async for message in client.tail_job_logs(job_id, no_follow=no_follow):
        reconnect_msg = _clear_line(reconnect_msg)
        if "log_blobs" in message and not no_follow:
            log_blobs = message.get("log_blobs", [])
            if log_blobs and not log_blobs_processed:
                log_blobs_processed = True
                received_streaming_data = True
                await _show_log_blobs(log_blobs, client)

        elif "logs" in message and not no_follow:
            received, last_log_id = _process_logs_message(
                message["logs"], last_log_id, session_start_id
            )
            received_streaming_data |= received
        elif "job" in message:
            latest_status = message["job"]["status"]
            if latest_status in processed_statuses:
                continue
            received_streaming_data = True
            processed_statuses.add(latest_status)
            print(f"\n>>>> Job is now in {latest_status} status.")
    return (
        received_streaming_data,
        last_log_id,
        latest_status,
        log_blobs_processed,
        reconnect_msg,
    )


def _should_stop(latest_status, retry_count):
    if latest_status and JobStatus[latest_status] in JobStatus.finished():
        logger.debug("Job is in finished status: %s", latest_status)
        return True
    if retry_count >= RECONNECT_MAX_ATTEMPTS:
        logger.debug("Max reconnect attempts reached: %d", retry_count)
        return True
    return False


def _reconnect_sleep(retry_count):
    sleep_sec = min(
        RECONNECT_BACKOFF_BASE_SEC * 2**retry_count,
        RECONNECT_BACKOFF_MAX_SEC,
    ) + random.uniform(0, 1)  # noqa: S311
    retry_count += 1
    logger.debug(
        "WebSocket closed, reconnecting in %.1fs (attempt %d/%d)",
        sleep_sec,
        retry_count,
        RECONNECT_MAX_ATTEMPTS,
    )
    reconnect_msg = _print_reconnect_msg(sleep_sec)
    return retry_count, sleep_sec, reconnect_msg


def _is_job_finished(status) -> bool:
    try:
        return bool(status and JobStatus[status] in JobStatus.finished())
    except KeyError:
        logger.debug("Job status is not a valid status: %s", status)
        return False


def _print_not_finished(final_status):
    logger.debug("Job is not finished: %s.", final_status or "unknown")
    print(
        f"\n>>>> Failed to reconnect after {RECONNECT_MAX_ATTEMPTS} attempts."
        f" Job status: {final_status or 'unknown'}."
        f"\nThe job may still be running. To resume monitoring:"
        f"\n    datachain job logs {final_status or 'unknown'}"
    )


def _print_dataset_versions(client, job_id):
    response = client.dataset_job_versions(job_id)
    if not response.ok:
        raise DataChainError(response.message)

    if response.data and response.data.get("dataset_versions"):
        dataset_versions = response.data.get("dataset_versions", [])
        print("\n\n>>>> Dataset versions created during the job:")
        for version in dataset_versions:
            print(f"    - {version.get('dataset_name')}@v{version.get('version')}")
    else:
        print("\n\nNo dataset versions created during the job.")


async def _tail_job_logs(client, job_id, no_follow):
    retry_count = last_log_id = 0
    latest_status = None
    processed_statuses = set()
    log_blobs_processed, reconnect_msg = False, ""
    while True:
        (
            received_streaming_data,
            last_log_id,
            msg_status,
            log_blobs_processed,
            reconnect_msg,
        ) = await _handle_ws_messages(
            client,
            job_id,
            no_follow,
            last_log_id,
            processed_statuses,
            log_blobs_processed,
            reconnect_msg,
        )
        if received_streaming_data:
            retry_count = 0
        if msg_status:
            latest_status = msg_status

        rest_status = _get_job_status(client, job_id)
        if rest_status:
            if rest_status != latest_status:
                print(f"\n>>>> Job is now in {rest_status} status.")
            latest_status = rest_status

        try:
            if _should_stop(latest_status, retry_count):
                break
            retry_count, sleep_sec, reconnect_msg = _reconnect_sleep(retry_count)
            await asyncio.sleep(sleep_sec)
        except KeyError:
            break

    return latest_status


def show_logs_from_client(client, job_id: str, no_follow: bool = False):
    final_status = asyncio.run(_tail_job_logs(client, job_id, no_follow))

    if not _is_job_finished(final_status):
        _print_not_finished(final_status)
        return 1

    _print_dataset_versions(client, job_id)

    return {"COMPLETE": 0, "FAILED": 1, "CANCELED": 2}.get(final_status.upper(), 0)


def _read_query(query_file: str) -> tuple[str, str]:
    query_type = "PYTHON" if query_file.endswith(".py") else "SHELL"
    with open(query_file) as f:
        query = f.read()
    return query_type, query


def _read_environment(env: list[str] | None, env_file: str | None) -> str:
    env_values = list(flatten(env)) if env else []
    environment = "\n".join(env_values) if env_values else ""
    if env_file:
        with open(env_file) as f:
            environment = f.read() + "\n" + environment
    return environment


def _read_requirements(req: list[str] | None, req_file: str | None) -> str:
    requirements = "\n".join(req) if req else ""
    if req_file:
        with open(req_file) as f:
            requirements = f.read() + "\n" + requirements
    return requirements


def _resolve_rerun(catalog, script_path: str) -> str | None:
    rerun_from_job = catalog.metastore.get_last_job_by_name(
        script_path, is_remote_execution=True
    )
    return rerun_from_job.id if rerun_from_job else None


def _save_remote_job(
    catalog,
    query: str,
    query_type: str,
    script_path: str,
    job_data: dict,
):
    query_type_value = (
        JobQueryType.PYTHON if query_type == "PYTHON" else JobQueryType.SHELL
    )
    catalog.metastore.create_job(
        name=script_path,
        query=query,
        query_type=query_type_value,
        status=JobStatus.CREATED,
        workers=job_data.get("workers", 0),
        python_version=job_data.get("python_version"),
        params=job_data.get("params", {}),
        parent_job_id=job_data.get("parent_job_id"),
        rerun_from_job_id=job_data.get("rerun_from_job_id"),
        run_group_id=job_data.get("run_group_id"),
        is_remote_execution=True,
        job_id=str(job_data.get("id")),
    )


def create_job(  # noqa: PLR0913
    query_file: str,
    team_name: str | None,
    env_file: str | None = None,
    env: list[str] | None = None,
    workers: int | None = None,
    files: list[str] | None = None,
    python_version: str | None = None,
    repository: str | None = None,
    req: list[str] | None = None,
    req_file: str | None = None,
    priority: int | None = None,
    cluster: str | None = None,
    start_time: str | None = None,
    cron: str | None = None,
    no_wait: bool | None = False,
    credentials_name: str | None = None,
    ignore_checkpoints: bool = False,
    no_follow: bool = False,
):
    catalog = get_catalog()
    query_type, query = _read_query(query_file)
    environment = _read_environment(env, env_file)
    requirements = _read_requirements(req, req_file)
    script_path = os.path.abspath(query_file)
    rerun_from_job_id = _resolve_rerun(catalog, script_path)

    client = StudioClient(team=team_name)
    file_ids = upload_files(client, files) if files else []

    parsed_start_time = parse_start_time(start_time)
    if cron and parsed_start_time is None:
        parsed_start_time = datetime.now(timezone.utc).isoformat()

    response = client.create_job(
        query=query,
        query_type=query_type,
        environment=environment,
        workers=workers,
        query_name=os.path.basename(query_file),
        rerun_from_job_id=rerun_from_job_id,
        reset=ignore_checkpoints,
        files=file_ids,
        python_version=python_version,
        repository=repository,
        requirements=requirements,
        priority=priority,
        cluster=cluster,
        start_time=parsed_start_time,
        cron=cron,
        credentials_name=credentials_name,
    )
    if not response.ok:
        raise DataChainError(response.message)
    if not response.data:
        raise DataChainError("Failed to create job")

    job_data = response.data
    _save_remote_job(catalog, query, query_type, script_path, job_data)
    catalog.close()

    job_id = job_data.get("id")
    if parsed_start_time or cron:
        print(f"Job {job_id} is scheduled as a task in Studio.")
        return 0

    print(f"Job {job_id} created")
    print("Open the job in Studio at", job_data.get("url"))
    print("=" * 40)

    return (
        0
        if no_wait
        else show_logs_from_client(
            client=client, job_id=str(job_id), no_follow=no_follow
        )
    )


def _upload_single_file(client: StudioClient, file_path: str) -> str:
    file_name = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        response = client.upload_file(f, file_name)
    if not response.ok:
        raise DataChainError(response.message)
    if not response.data:
        raise DataChainError(f"Failed to upload file {file_name}")
    return str(response.data.get("id"))


def upload_files(client: StudioClient, files: list[str]) -> list[str]:
    return [_upload_single_file(client, f) for f in files]


def cancel_job(job_id: str, team_name: str | None):
    _require_studio_token()
    client = StudioClient(team=team_name)
    response = client.cancel_job(job_id)
    if not response.ok:
        raise DataChainError(response.message)

    print(f"Job {job_id} canceled")


def _print_table(data: list[dict], column_map: dict[str, str], empty_msg: str):
    if not data:
        print(empty_msg)
        return
    rows = [{k: row.get(v) for k, v in column_map.items()} for row in data]
    print(tabulate.tabulate(rows, headers="keys", tablefmt="grid"))


def list_jobs(status: str | None, team_name: str | None, limit: int):
    client = StudioClient(team=team_name)
    response = client.get_jobs(status, limit)
    if not response.ok:
        raise DataChainError(response.message)

    jobs = response.data or []
    _print_table(
        jobs,
        {
            "ID": "id",
            "Name": "name",
            "Status": "status",
            "Created at": "created_at",
            "Created by": "created_by",
        },
        "No jobs found",
    )


def show_job_logs(job_id: str, team_name: str | None):
    _require_studio_token()
    client = StudioClient(team=team_name)
    return show_logs_from_client(client, job_id)


def list_clusters(team_name: str | None):
    client = StudioClient(team=team_name)
    response = client.get_clusters()
    if not response.ok:
        raise DataChainError(response.message)

    _print_table(
        response.data or [],
        {
            "ID": "id",
            "Name": "name",
            "Status": "status",
            "Cloud Provider": "cloud_provider",
            "Cloud Credentials": "cloud_credentials",
            "Is Active": "is_active",
            "Is Default": "default",
            "Max Workers": "max_workers",
        },
        "No clusters found",
    )


def create_pipeline(
    catalog: "Catalog",
    dataset_names: list[str],
    team_name: str | None = None,
):
    client = StudioClient(team=team_name)
    response = client.create_pipeline(
        datasets=dataset_names,
        team_name=team_name,
        review=True,
    )
    if not response.ok:
        raise DataChainError(response.message)

    pipeline = response.data["pipeline"]
    _print_pipeline_created(pipeline)
    return 0


def _print_pipeline_created(pipeline: dict):
    print(
        f"Pipeline created under name: {pipeline['name']} from:"
        f" {pipeline['triggered_from']} in paused state for review."
    )
    print(
        "Check the pipeline either in Studio or using `datachain pipeline status`, "
        "and resume it when ready using `datachain pipeline resume`"
    )


def _display_pipeline_summary(data: dict):
    print(f"Name: {data.get('name', 'N/A')}")
    print(f"Status: {data.get('status', 'N/A')}")

    completed = data.get("completed", 0)
    total = data.get("total", 0)
    print(f"Progress: {completed}/{total} jobs completed")

    if data.get("error_message"):
        print(f"Error: {data.get('error_message')}")


def _display_job_runs(data: dict):
    job_runs = data.get("job_runs", [])
    if job_runs:
        print("\nJob Runs:")
        rows = [
            {
                "Name": job_run.get("name", "N/A"),
                "Status": job_run.get("status", "N/A"),
                "Job ID": job_run.get("created_job_id", "N/A"),
            }
            for job_run in job_runs
        ]
        print(tabulate.tabulate(rows, headers="keys", tablefmt="grid"))
    else:
        print("\nNo job runs found")


def get_pipeline_status(name: str, team_name: str | None):
    client = StudioClient(team=team_name)
    response = client.get_pipeline(name)
    if not response.ok:
        raise DataChainError(response.message)

    data = response.data
    _display_pipeline_summary(data)
    _display_job_runs(data)
    return 0


def _display_pipelines(data: list[dict] | None):
    if not data:
        print("No pipelines found")
        return
    rows = [
        {
            "Name": pipeline.get("name", "N/A"),
            "Status": pipeline.get("status", "N/A"),
            "Target": pipeline.get("triggered_from", "N/A"),
            "Progress": f"{pipeline.get('completed', 0)}/{pipeline.get('total', 0)}",
            "Created At": pipeline.get("created_at", "N/A")[:19],
        }
        for pipeline in data
    ]
    print(tabulate.tabulate(rows, headers="keys", tablefmt="grid"))


def list_pipelines(
    team_name: str | None,
    status: str | None = None,
    limit: int = 20,
    search: str | None = None,
):
    client = StudioClient(team=team_name)
    response = client.list_pipelines(status, limit, search)
    if not response.ok:
        raise DataChainError(response.message)

    _display_pipelines(response.data)
    return 0


def pause_pipeline(name: str, team_name: str | None):
    client = StudioClient(team=team_name)
    response = client.pause_pipeline(name)
    if not response.ok:
        raise DataChainError(response.message)

    print(f"Pipeline {name} paused")

    return 0


def resume_pipeline(name: str, team_name: str | None):
    client = StudioClient(team=team_name)
    response = client.resume_pipeline(name)
    if not response.ok:
        raise DataChainError(response.message)

    print(f"Pipeline {name} resumed")

    return 0


def remove_job_from_pipeline(name: str, job_id: str, team_name: str | None):
    client = StudioClient(team=team_name)
    response = client.remove_job_from_pipeline(name, job_id)
    if not response.ok:
        raise DataChainError(response.message)

    print(f"Job {job_id} removed from pipeline {name}")

    return 0
