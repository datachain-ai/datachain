import subprocess
import sys

import pytest

TOP_SLOWEST = 15

lazy_modules = [
    "adlfs",
    "boto3",
    "botocore",
    "gcsfs",
    "google",
    "numpy",
    "pyarrow",
    "requests",
    "s3fs",
    "sqlalchemy.dialects.postgresql",
    "torch",
]


def _import_datachain():
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "import datachain"],
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    imports = []
    for line in proc.stderr.splitlines():
        if not line.startswith("import time:"):
            continue
        _, cumulative_us, package = line[len("import time:") :].split("|")
        if cumulative_us.strip().isdigit():
            indent = len(package) - len(package.lstrip())
            imports.append((int(cumulative_us) // 1000, package.strip(), indent))
    return imports


def _import_chain(imports, index):
    chain = [imports[index][1]]
    depth = imports[index][2]
    for _, name, indent in imports[index + 1 :]:
        if indent < depth:
            chain.append(name)
            depth = indent
    return " <- ".join(chain)


def _is_lazy(name, module):
    return name == module or name.startswith(f"{module}.")


# disable coverage for this test to minimize import time overhead
@pytest.mark.no_cover
def test_lazy_modules_not_imported():
    """
    Outside of the test, you can profile the import time with:
        python -Ximporttime -c 'import datachain'

    To visualize the import profile, consider using `tuna`: https://github.com/nschloe/tuna.
    """
    imports = _import_datachain()

    offenders = [
        f"  {name} ({_import_chain(imports, i)})"
        for i, (_, name, _) in enumerate(imports)
        if any(_is_lazy(name, module) for module in lazy_modules)
    ]
    slowest = "\n".join(
        f"  {ms}ms {name}"
        for ms, name, _ in sorted(imports, reverse=True)[:TOP_SLOWEST]
    )
    assert not offenders, (
        "Modules that must stay lazy were imported by `import datachain`:\n"
        + "\n".join(offenders)
        + f"\n\nSlowest imports (cumulative):\n{slowest}"
    )
