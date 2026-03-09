from datachain.cli.commands.ls import ls_local


def test_ls(benchmark, tmp_dir, datasets, test_session):
    # Use local dataset for faster iteration
    path = datasets.as_uri() + "/"
    benchmark.pedantic(
        ls_local, args=([path],), kwargs={"catalog": test_session.catalog}
    )
