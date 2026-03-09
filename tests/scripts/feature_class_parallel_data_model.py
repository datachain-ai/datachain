from typing import Literal

import datachain as dc
from datachain import C
from datachain.lib.data_model import DataModel


class NestedFeature(DataModel):
    value: str


class Embedding(DataModel):
    value: float
    nested: NestedFeature = NestedFeature(value="nested_value")
    literal_field: Literal["end_turn", "max_tokens", "stop_sequence"] | None = None


ds_name = "feature_class"
ds = (
    dc.read_storage("gs://dvcx-datalakes/dogs-and-cats/")
    .filter(C("file.path").glob("*cat*.jpg"))  # type: ignore [attr-defined]
    .order_by("file.path")
    .limit(5)
    .settings(cache=True, parallel=2)
    .map(emd=lambda file: Embedding(value=512), output=Embedding)
    .save(ds_name)
)

for row in ds.results():
    print(row[1])
