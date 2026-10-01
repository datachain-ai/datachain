import pytest

from datachain import semver


@pytest.mark.parametrize(
    "version,expected",
    [
        ("0.0.0", (0, 0, 0)),
        ("0.1.2", (0, 1, 2)),
        ("1.0.0", (1, 0, 0)),
        ("10.20.30", (10, 20, 30)),
        ("100.200.300", (100, 200, 300)),
        ("999999.999999.999999", (999999, 999999, 999999)),
    ],
)
def test_parse(version, expected):
    assert semver.parse(version) == expected


@pytest.mark.parametrize(
    "version",
    [
        "0",
        "1",
        "-1",
        "1.2",
        "1.2.-3",
        "1.2.3-alpha+01",
        "dev",
    ],
)
def test_parse_wrong_format(version):
    with pytest.raises(ValueError) as excinfo:
        semver.parse(version)
    assert str(excinfo.value) == (
        "Invalid version. It should be in format: <major>.<minor>.<patch> where"
        " each version part is positive integer"
    )


@pytest.mark.parametrize(
    "version,expected",
    [
        ((), "0.0.0"),
        ((0, 0, 0), "0.0.0"),
        ((1, 2, 3), "1.2.3"),
        ((10, 20, 30), "10.20.30"),
        ((100, 200, 300), "100.200.300"),
        ((999999, 999999, 999999), "999999.999999.999999"),
    ],
)
def test_create(version, expected):
    assert semver.create(*version) == expected


@pytest.mark.parametrize(
    "version",
    [
        (-1,),
        (1, 1000000),
        (-1, 2, 3),
        (1, -2, 3),
        (1, 2, -3),
        (1000000, 0, 0),
        (0, 1000000, 0),
        (0, 0, 1000000),
    ],
)
def test_create_wrong_values(version):
    with pytest.raises(ValueError) as excinfo:
        semver.create(*version)
    assert str(excinfo.value) == (
        "Major, minor and patch must be greater or equal to zero"
    )


@pytest.mark.parametrize(
    "version,expected",
    [
        ("0.0.0", 0),
        ("1.2.3", 1_000_002_000_003),
        ("10.20.30", 10_000_020_000_030),
        ("100.200.300", 100_000_200_000_300),
        ("999999.999999.999999", 999_999_999_999_999_999),
    ],
)
def test_value(version, expected):
    assert semver.value(version) == expected


@pytest.mark.parametrize(
    "v1,v2,result",
    [
        ("0.0.0", "0.0.0", 0),
        ("1.2.3", "1.2.3", 0),
        ("1.2.3", "1.2.4", -1),
        ("1.2.3", "1.2.2", 1),
        ("0.0.0", "999999.999999.999999", -1),
        ("999999.999999.999999", "0.0.1", 1),
        ("999999.999999.999999", "999998.999999.999999", 1),
        ("999999.999998.999999", "999999.999999.999999", -1),
        ("999999.999999.999999", "999999.999999.999998", 1),
        ("999999.999999.999999", "999999.999999.999999", 0),
    ],
)
def test_compare(v1, v2, result):
    assert semver.compare(v1, v2) == result


@pytest.mark.parametrize(
    "version,valid",
    [
        ("0.0.0", True),
        ("100.100.100", True),
        ("999999.999999.999999", True),
        ("9999999.9999999.9999999", False),
        ("1000000.0.1", False),
        ("-1.2.3", False),
        ("1.2.3-alpha+01", False),
        ("1.2", False),
        ("1", False),
        ("1.2.3.4", False),
        ("01.02.03", False),
        ("1.00.0", False),
        ("10.0.0", True),
        ("+2.0.0", False),
        ("1.+2.3", False),
        ("1.2.-0", False),
        (" 1.2.3", False),
        ("1.2.3 ", False),
        ("1_0.2.3", False),
        ("\u0661.2.3", False),
    ],
)
def test_validate(version, valid):
    if valid:
        semver.validate(version)
    else:
        with pytest.raises(ValueError) as excinfo:
            semver.validate(version)
        assert str(excinfo.value) == (
            "Invalid version. It should be in format: <major>.<minor>.<patch> where"
            " each version part is positive integer"
        )
