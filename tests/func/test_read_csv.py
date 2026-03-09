from collections import Counter

import pandas as pd

import datachain as dc


def test_read_csv_boolean_column(tmp_dir, test_session):
    df = pd.DataFrame({"flag": [True, False, True]})
    path = tmp_dir / "bools.csv"
    df.to_csv(path, index=False)

    chain = dc.read_csv(path.as_uri(), session=test_session)
    rows = list(chain.to_iter("flag"))

    assert Counter(rows) == Counter({(True,): 2, (False,): 1})
    assert all(isinstance(value[0], bool) for value in rows)
