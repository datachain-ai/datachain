"""Automation using nox."""
# /// script
# dependencies = ["nox"]
# ///

import glob

import nox

nox.options.default_venv_backend = "uv|virtualenv"
nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = "lint", "tests"

project = nox.project.load_toml()
python_versions = nox.project.python_versions(project)
locations = "src", "tests"


@nox.session
def docs(session: nox.Session) -> None:
    session.install(".[docs]")
    session.run("mkdocs", "build")


@nox.session
def bench(session: nox.Session) -> None:
    session.install(".[tests]")
    session.run(
        "pytest",
        "--benchmark-only",
        "--benchmark-group-by",
        "func",
        *session.posargs,
    )


@nox.session(python=python_versions)
def tests(session: nox.Session) -> None:
    session.install(".[tests]")
    env = {"COVERAGE_FILE": f".coverage.{session.python}"}
    if session.python in ("3.12", "3.13"):
        # improve performance of tests in Python>=3.12 when used with coverage
        # https://github.com/nedbat/coveragepy/issues/1665
        # https://github.com/python/cpython/issues/107674
        env["COVERAGE_CORE"] = "sysmon"
    session.run(
        "pytest",
        "--cov",
        "--cov-config=pyproject.toml",
        "--cov-report=xml",
        "--durations=10",
        "--numprocesses=logical",
        "--dist=loadgroup",
        *session.posargs,
        env=env,
    )


@nox.session(python=python_versions)
def e2e(session: nox.Session) -> None:
    session.install(".[tests]")
    session.run(
        "pytest",
        "--durations=0",
        "--numprocesses=logical",
        "--dist=loadgroup",
        "-m",
        "e2e",
        *session.posargs,
    )


@nox.session
def lint(session: nox.Session) -> None:
    session.install("pre-commit")
    session.install("-e", ".[dev,vector]")

    args = *(session.posargs or ("--show-diff-on-failure",)), "--all-files"
    session.run("pre-commit", "run", *args)


@nox.session
def build(session: nox.Session) -> None:
    session.install("twine", "uv")
    session.run("uv", "build")
    dists = glob.glob("dist/*")
    session.run("twine", "check", *dists, silent=True)


@nox.session(python=python_versions)
def examples(session: nox.Session) -> None:
    session.install(".[examples]")
    session.run("uv", "pip", "list")
    session.run(
        "pytest",
        "--durations=0",
        "tests/examples",
        "-m",
        "examples",
        *session.posargs,
    )


if __name__ == "__main__":
    nox.main()
