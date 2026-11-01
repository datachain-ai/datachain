import math
from functools import lru_cache

import numpy as np

from datachain import json


# Scans reuse one query vector; joins reuse a small set of query vectors.
@lru_cache(maxsize=64)
def _parse(a: str) -> np.ndarray:
    values = np.array(json.loads(a), dtype=np.float64)
    values.flags.writeable = False
    return values


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
