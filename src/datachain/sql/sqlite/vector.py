import math
from functools import lru_cache

import numpy as np
import ujson


# A scan compares every row against the same query vector; cache its parse.
@lru_cache(maxsize=8)
def _parse(a: str) -> np.ndarray:
    try:
        return np.array(ujson.loads(a), dtype=np.float64)
    except (ValueError, TypeError):
        return np.fromstring(a[1:-1], sep=",")


def euclidean_distance(a: str, b: str):
    return np.linalg.norm(_parse(b) - _parse(a))


def cosine_distance(a: str, b: str):
    u = _parse(a)
    v = _parse(b)

    uv = np.inner(u, v)
    uu = np.inner(u, u)
    vv = np.inner(v, v)

    dist = 1.0 - uv / math.sqrt(uu * vv)

    return max(0, min(dist, 2.0))
