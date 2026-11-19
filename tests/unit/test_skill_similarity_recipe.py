import ast
import re

import numpy as np
import pytest

from datachain.cli.commands.skill import _skills_src


@pytest.fixture
def nearest_rows():
    text = (_skills_src() / "datachain" / "SKILL.md").read_text(encoding="utf-8")
    (function,) = [
        node
        for block in re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        for node in ast.parse(block).body
        if isinstance(node, ast.FunctionDef) and node.name == "nearest_rows"
    ]
    namespace = {"np": np}
    exec(  # noqa: S102
        compile(ast.Module(body=[function], type_ignores=[]), "recipe", "exec"),
        namespace,
    )

    def run(embeddings, queries):
        namespace["E"] = embeddings
        return list(namespace["nearest_rows"](queries))

    return run


@pytest.mark.parametrize(
    "rows,queries", [(300, 289), (129, 129), (300, 0), (2, 2), (1, 1), (0, 0)]
)
def test_similarity_recipe_matches_full_scan(nearest_rows, rows, queries):
    random = np.random.default_rng(42)
    embeddings = random.normal(size=(rows, 16)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    if rows > 2:
        # Identical embeddings in separate rows must still exclude only the query row.
        embeddings[1] = embeddings[0]
    query_rows = random.permutation(rows)[:queries]
    result = nearest_rows(embeddings, query_rows)
    if rows < 2 or not queries:
        assert result == []
        return

    scores = embeddings[query_rows] @ embeddings.T
    scores[np.arange(len(query_rows)), query_rows] = -np.inf
    matches = scores.argmax(axis=1)
    assert [(i, j) for i, j, _ in result] == list(zip(query_rows, matches, strict=True))
    assert [s for _, _, s in result] == pytest.approx(
        scores[np.arange(queries), matches]
    )
    assert all(i != j for i, j, _ in result)
    if rows > 2:
        neighbors = {i: j for i, j, _ in result}
        for i in (0, 1):
            if i in neighbors:
                assert neighbors[i] == 1 - i


def test_similarity_recipe_bounds_score_matrix_memory(nearest_rows):
    products = []

    class Embeddings(np.ndarray):
        def __matmul__(self, other):
            products.append((self.shape[0], other.shape[1]))
            return super().__matmul__(other)

    embeddings = np.ones((300, 16), dtype=np.float32).view(Embeddings)
    assert len(nearest_rows(embeddings, range(len(embeddings)))) == 300
    assert sum(rows for rows, _ in products) == 300
    assert all(rows <= 128 and columns == 300 for rows, columns in products)
