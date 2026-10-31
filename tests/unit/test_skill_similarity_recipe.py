import ast
import re
import weakref

import numpy as np
import pytest

from datachain.cli.commands.skill import _skills_src


@pytest.fixture
def nearest_rows():
    text = (_skills_src() / "datachain" / "SKILL.md").read_text()
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
def test_similarity_recipe_matches_full_scan_and_bounds_score_blocks(
    nearest_rows, rows, queries
):
    random = np.random.default_rng(42)
    embeddings = random.normal(size=(rows, 16)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    if rows > 2:
        # Identical embeddings in separate rows must still exclude only the query row.
        embeddings[1] = embeddings[0]
    query_rows = random.permutation(rows)[:queries]
    shapes = []
    previous = None

    class TrackedEmbeddings(np.ndarray):
        def __matmul__(self, other):
            nonlocal previous
            assert previous is None or previous() is None
            scores = np.asarray(self) @ np.asarray(other)
            shapes.append(scores.shape)
            previous = weakref.ref(scores)
            return scores

    result = nearest_rows(embeddings.view(TrackedEmbeddings), query_rows)
    if rows < 2 or not queries:
        assert result == []
        assert shapes == []
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
    assert shapes == [
        (min(128, queries - start), rows) for start in range(0, queries, 128)
    ]
    assert previous() is None
