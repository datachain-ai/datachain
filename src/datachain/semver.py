# Maximum version number for semver (major.minor.patch) is 999999.999999.999999
# this number was chosen because value("999999.999999.999999") < 2**63 - 1
MAX_VERSION_NUMBER = 999_999


def parse(version: str) -> tuple[int, int, int]:
    """Parsing semver into 3 integers: major, minor, patch"""
    validate(version)
    parts = version.split(".")
    return int(parts[0]), int(parts[1]), int(parts[2])


def validate(version: str) -> None:
    """
    Raises exception if version doesn't have valid semver format which is:
    <major>.<minor>.<patch> or one of version parts is not positive integer
    """
    error_message = (
        "Invalid version. It should be in format: <major>.<minor>.<patch> where"
        " each version part is positive integer"
    )
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(error_message)
    for part in parts:
        if not (part.isascii() and part.isdigit()):
            raise ValueError(error_message)
        if len(part) > 1 and part[0] == "0":
            raise ValueError(error_message)
        if int(part) > MAX_VERSION_NUMBER:
            raise ValueError(error_message)


def create(major: int = 0, minor: int = 0, patch: int = 0) -> str:
    """Creates new semver from 3 integers: major, minor and patch"""
    if not (
        0 <= major <= MAX_VERSION_NUMBER
        and 0 <= minor <= MAX_VERSION_NUMBER
        and 0 <= patch <= MAX_VERSION_NUMBER
    ):
        raise ValueError("Major, minor and patch must be greater or equal to zero")

    return ".".join([str(major), str(minor), str(patch)])


def value(version: str) -> int:
    """
    Calculate integer value of a version. This is useful when comparing two versions.
    """
    major, minor, patch = parse(version)
    limit = MAX_VERSION_NUMBER + 1
    return major * (limit**2) + minor * limit + patch


def compare(v1: str, v2: str) -> int:
    """
    Compares 2 versions and returns:
       -1 if v1 < v2
        0 if v1 == v2
        1 if v1 > v2
    """
    v1_val = value(v1)
    v2_val = value(v2)

    if v1_val < v2_val:
        return -1
    if v1_val > v2_val:
        return 1
    return 0
