import pytest
from sqlalchemy import (
    Float,
    Integer,
    String,
    and_,
    case,
    cast,
    distinct,
    literal,
    not_,
    null,
    or_,
    tuple_,
)
from sqlalchemy import func as sa_func

from datachain import C, func
from datachain.hash_utils import hash_callable, hash_column_elements


def double(x):
    return x * 2


def double_arg_annot(x: int):
    return x * 2


def double_arg_and_return_annot(x: int) -> int:
    return x * 2


lambda1 = lambda x: x * 2  # noqa: E731
lambda2 = lambda y: y + 1  # noqa: E731
lambda3 = lambda z: z - 1  # noqa: E731


@pytest.mark.parametrize(
    "expr,result",
    [
        # Basic column references
        (
            [C("name")],
            "9ff0747010842981e8a973b68d09f65682e14cc850a1b8a4badc2056461b8f9e",
        ),
        (
            [C("name"), C("age")],
            "6cfcbd8bebaf6f635685ea33962a6f060188c5b37e86806cc559c1d8c8c97ab4",
        ),
        # Functions
        (
            [func.avg("age")],
            "1e4e256dbd18e42fdc6ca4889e01b23862122ffe6e47014a8cbc26de79afdfa3",
        ),
        (
            [func.count()],
            "1e1e766d2292b1f57fc93c4730cd4a059aa5223626069c958f3c9d79a85cca7a",
        ),
        (
            [func.sum(C("age"))],
            "db55ee6b19ae21138fb066e51706e6ec6cc2c8aec75f22313ebf2c028fdb2200",
        ),
        (
            [func.min(C("age")), func.max(C("age"))],
            "28dacc69c5d72045f6341c06bc7c3361f9c14c445b1bd783900dae1effba0cc4",
        ),
        (
            [sa_func.coalesce(C("name"), "unknown")],
            "13532bc4a1a250e6ea24420bddda3cc1dc15b0ce710b1d2a73c815129152caad",
        ),
        (
            [sa_func.nullif(C("age"), 0)],
            "57c0d74f3881dd687a1d97717e662ecea668d89be62f75487ab6824d2e176761",
        ),
        # Window functions
        (
            [
                func.row_number().over(
                    func.window(partition_by="file.name", order_by="file.name")
                )
            ],
            "3088d3b926949f61df407ccf3057f6d298873f9e0f8f7497e1430db3048bd6db",
        ),
        # Labels
        (
            [C("age").label("user_age")],
            "cc9bebfc972f7358a77768cd08f4ccae6f6cea8f6ab10ecc5cb74d4bf9348f76",
        ),
        # Arithmetic operations
        (
            [C("age") + 10],
            "a819bb33c57756cf28ce23311cf1d618a17ee0bde8cc9a2d20409727744877ca",
        ),
        (
            [C("age") * 2],
            "89bf2d64c0db60cf20f791ce4b31d37d368ef79802ebce2e40d807a03472deba",
        ),
        (
            [C("age") % 10],
            "17ade84e87a0fc88f90ac3f72bbd331a4d1c8c440f18c13c9ec439ccdf70a4c3",
        ),
        (
            [-C("age")],
            "660f4f7991c53f15709132d3095a616a5fffc45009ff3e0fd56f724afb0b7048",
        ),
        # Comparison operations
        (
            [C("age") > 20],
            "2cbfcbf64022dfc87f828ad3971da29172ee4b48cb1c53c633b65105994422e5",
        ),
        (
            [C("age") >= 21],
            "db078ea46b20bfdb8490ae186aa81d1707b15d38cca7c70ddb9b9cd5896af1a6",
        ),
        (
            [C("age") == 25],
            "7bf196beaeaefdfba89b3ae11e4741d7db1a385c1fc16f91d009992f67e02403",
        ),
        (
            [C("name") != ""],
            "fece00afaa11205b37a4835e783a941752df08af1d474afbbe363337b42cf782",
        ),
        # Logical operations
        (
            [and_(C("age") > 20, C("name") != "")],
            "28927235e702ce883662a8b06c726aa7519bdb8305fc8345bcf7e6b198cde243",
        ),
        (
            [or_(C("age") < 18, C("age") > 65)],
            "2bdeca05439163390ba6c96e87c90f74bf5eb2521acb2833ef3132e6eeca42b1",
        ),
        (
            [not_(C("active"))],
            "c3b9e76d5dc067e35189d021e10ba51f516fc355cd3302905c2f0b530440b3ac",
        ),
        (
            [and_(C("age") > 18, or_(C("city") == "NYC", C("city") == "LA"))],
            "c667270f3bfa16be1e2d4024c1a5f8cd810d38fcdd6df00e0b69743d08cb93ef",
        ),
        # String operations
        (
            [C("name").like("John%")],
            "e5bd392c9add2ddf1c02863320059a9912b45cebf029e16455e8a3a75c52b9ef",
        ),
        (
            [C("name").startswith("A")],
            "91179e5d93da3bb8955c8779e039c1b6b79eb07ecd18965309b38281aac84c8b",
        ),
        (
            [C("name").contains("John")],
            "a1cb94e08a4c1c9e17a50bc43da70f6710cfac035ffb4ad744c398284ca7bb2a",
        ),
        (
            [sa_func.concat(C("first_name"), " ", C("last_name"))],
            "b7b6b68d73eb0ea274e8d1d418541a52023cc978106bcf914d92a0bee810566a",
        ),
        (
            [sa_func.lower(C("name"))],
            "09ce2d422c40e4db6dd0c1a1dc8d34457786dc366c8dca23b87aca2a76af1542",
        ),
        # NULL operations
        (
            [C("name").is_(None)],
            "8dd013d2fb21adf266bde8572f148b637769bd751902011e16ac7769d8167d62",
        ),
        (
            [C("name").isnot(None)],
            "f21c92f266be45b66dc5a91e4918305be7b412c7327c077368bcc60982a5fbbb",
        ),
        # IN operations
        (
            [C("age").in_([18, 21, 65])],
            "b96949143855d2f42de36398a1330c1d0eb76435f667505f3515319f95942772",
        ),
        (
            [C("name").in_(["Alice", "Bob", "Charlie"])],
            "c7812971ae30d3af4a6a9cc2809765482ba515855d981d91f084ad5786f90861",
        ),
        (
            [C("age").notin_([0, 1])],
            "d259260deaa33ce15926f070585efc2e8f5f5f06006877b85d2c945f2a1bb786",
        ),
        # BETWEEN
        (
            [C("age").between(18, 65)],
            "752a3c594167f829cd70119cae8bb010c4d670eae12005526153f04bc39bc2b4",
        ),
        # Cast operations with different types
        (
            [cast(C("age"), Integer)],
            "167e712a70f2100d1d9f8d28ee40170882900f7195463413a4403b0b8beff7c6",
        ),
        (
            [cast(C("price"), Float)],
            "c6962d8f2a74ded308b6aab5d00acee4deddfecce09e0533450cb5031f3eaa39",
        ),
        (
            [cast(C("id"), String)],
            "52cf27d8d6786e7e0d95a79dfcc1188d89e92f9d250fcf9baf76c91f2cb62a99",
        ),
        # CASE expressions
        (
            [case((C("age") > 20, "adult"), else_="child")],
            "c0a28bf0a0b142b781ea687e81df60b66004fa0b8df6d8b0972a4586ba5ba77b",
        ),
        (
            [
                case(
                    (C("age") < 13, "child"),
                    (C("age") < 20, "teen"),
                    (C("age") < 65, "adult"),
                    else_="senior",
                )
            ],
            "8235765854809fb8a394a87b69a988ba240f84e14c830287d2df083c941fe649",
        ),
        (
            [case({1: "one", 2: "two", 3: "three"}, value=C("num"), else_="other")],
            "4f3d2ea0f185e32e7a24d0ac61ff3de3b5e6a64c23684cd871c3be96da739b20",
        ),
        # Tuple operations
        (
            [tuple_(C("name"), C("age"))],
            "83cac95650d48fc1267b045a009173de012fcd2cdcf62812b21df345fe41e3ec",
        ),
        # Grouping
        (
            [(C("age") + 10).self_group()],
            "d2e8c32e9fd0d0906e9364ebfded13f0b4f539773c6174f3bbf99eaf98222c82",
        ),
        # DISTINCT
        (
            [distinct(C("name"))],
            "2662643edb4f4251947d76d78584665d0499a652ea7a48a2831b91097a5f4730",
        ),
        # Literal values
        (
            [literal(42)],
            "96812d801ca9b755942a0a9837c5c43e21a6f443a6e0d34d5318c2644c791d35",
        ),
        (
            [literal("hello")],
            "2e41270813206603bcdbf16916bb5c9140af6395ed83fd70ae125e0277c9ab74",
        ),
        (
            [literal(True)],
            "135ff6f6ae9586a85f39bab816405378f87cc1d6b38b3234020070bc991c7fb0",
        ),
        # Bytes values
        (
            [C("data") == b"hello"],
            "5b6b7a9ede886f536a823ef89d07e219a17659d500d2d0b686f1cdc68dc028bd",
        ),
        (
            [C("data") == b"\x00\x01\x02\xff"],
            "cef7cca1378bf28ed7c719250303023e112db589ed4c989effa00cdf8a29589f",
        ),
        (
            [C("data") == b""],
            "14bc6eb9bc46b0c0fd75c1e884fa0418c480028c639c9256c8a6388428cb8cd4",
        ),
        # NULL literal
        (
            [null()],
            "a21b4f0c7c1b9a3a0348ee750e4b55a73cd2674f61a3c5ad2fc250ec74c90538",
        ),
        # Ordering
        (
            [C("age").desc()],
            "14d06018ff87649a6c48b6e2cd5117817a422aa39ff1ccc78255e18496f78216",
        ),
        (
            [C("name").asc()],
            "82c23e5c3bb16da8652bcc39b94f1d366af3b8e274e2f1a29276bd5f1525c27e",
        ),
        # Complex nested expressions
        (
            [
                case(
                    (C("age") > 65, "senior"),
                    else_=case((C("age") >= 18, "adult"), else_="minor"),
                )
            ],
            "1b8b42c231b033da9da01b4267e370f2210513e66d90898cf045e41febcbb6ec",
        ),
        (
            [((C("price") * C("quantity")) + C("tax")) - C("discount")],
            "03ba855b075db046ae39a3227b403a3d09ea9de4aa9a4643a4c6e73a538c1d28",
        ),
        (
            [
                and_(
                    C("active") == True,  # noqa: E712
                    or_(C("age") >= 18, C("parent_consent") == True),  # noqa: E712
                    C("verified").isnot(None),
                )
            ],
            "8675b93419cb6f3de78ba1a82e93bdcf2fca80007945313366f54b477f7274f4",
        ),
        # Empty list
        (
            [],
            "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        ),
        # Edge cases
        (
            [C("empty_string") == ""],
            "b816b424bcf870ae0b9dd34a5e7f670aa198b807e7cb779734bc4e5b2616471b",
        ),
        (
            [C("zero") == 0],
            "f9ba87e4ec00e1dc402a6926fd9a2e197793895ade204e3b2d28e3f8321ae3ad",
        ),
        (
            [C("negative") < -1000],
            "c0f87a3ca2ee750e8ebb5d431b12f54e6430488b60cb8437c3330214fe4bc1cd",
        ),
        (
            [sa_func.abs(C("value"))],
            "0f32eade11cd2591bf7c948760c8658c66f9c3986b15affc7488a5646e2f6e39",
        ),
        (
            [sa_func.round(C("price"), 2)],
            "1c0223a26f3884bfceca93e69dc9e0adbcb0dbe00d96338c4343b4128330ef63",
        ),
    ],
)
def test_hash_column_elements(expr, result):
    assert hash_column_elements(expr) == result


@pytest.mark.parametrize(
    "func,expected_hash",
    [
        (double, "aba077bec793c25e277923cde6905636a80595d1cb9a92a2c53432fc620d2f44"),
        (
            double_arg_annot,
            "391b2bfe41cfb76a9bb7e72c5ab4333f89124cd256d87cee93378739d078400f",
        ),
        (
            double_arg_and_return_annot,
            "5f6c61c05d2c01a1b3745a69580cbf573ecdce2e09cce332cb83db0b270ff870",
        ),
    ],
)
def test_hash_named_functions(func, expected_hash):
    h = hash_callable(func)
    assert h == expected_hash


@pytest.mark.parametrize(
    "func",
    [
        lambda1,
        lambda2,
        lambda3,
    ],
)
def test_lambda_same_hash(func):
    h1 = hash_callable(func)
    h2 = hash_callable(func)
    assert h1 == h2  # same object produces same hash


def test_lambda_different_hashes():
    h1 = hash_callable(lambda1)
    h2 = hash_callable(lambda2)
    h3 = hash_callable(lambda3)

    # Ensure hashes are all different
    assert len({h1, h2, h3}) == 3


def test_hash_callable_with_dependencies():
    # Define helper and function that uses it
    def helper(x):
        return x + 1

    def func_with_helper(x):
        return helper(x) * 2

    hash1 = hash_callable(func_with_helper)
    assert hash1 == "5b2dbae7cca8695acd62ea2ee2226277962c1c59a098ab948ff1b2e73b3d822c"

    # Redefine helper with different implementation (same name, different code)
    def helper(x):  # noqa: F811
        return x + 10

    def func_with_helper(x):
        return helper(x) * 2

    hash2 = hash_callable(func_with_helper)
    assert hash2 == "099b86b464fb5a901393b28f073b7701f22a31775b5ce8402b4ea1116a50064e"

    # Hashes should be different because helper changed
    assert hash1 != hash2


def test_hash_callable_recursive():
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)

    assert hash_callable(factorial) is not None


def test_hash_callable_mutual_recursion():
    def func_a(n):
        return func_b(n - 1) if n > 0 else 0

    def func_b(n):
        return func_a(n - 1) if n > 0 else 1

    hash_a = hash_callable(func_a)
    hash_b = hash_callable(func_b)

    assert hash_a is not None
    assert hash_b is not None
    # Hashes should be different since functions are different
    assert hash_a != hash_b


def test_hash_callable_global_variable_limitation():
    # This test documents the current limitation - global variables don't affect hash

    global THRESHOLD  # noqa: PLW0603
    THRESHOLD = 100

    def filter_data(x):
        return x > THRESHOLD

    hash1 = hash_callable(filter_data)

    # Change global variable
    THRESHOLD = 200

    hash2 = hash_callable(filter_data)

    # Hash is the same even though behavior changed (limitation)
    assert hash1 == hash2
