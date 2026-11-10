"""Read samples from WebDataset tar shards without extracting files.

`spec` accepts WDSBasic or a subclass. This example uses WDSLaion for
image files, JSON metadata, and text captions.
"""

import datachain as dc
import datachain.error
from datachain import func
from datachain.lib.webdataset_laion import WDSLaion

name = "wds"
try:
    wds = dc.read_dataset(name)
except datachain.error.DatasetNotFoundError:
    wds = dc.read_webdataset(
        "gs://datachain-demo/datacomp-small/shards/00000000.tar",
        spec=WDSLaion,
        column="laion",
        settings={"cache": True},
        anon=True,
    ).save(name)

print(wds.schema)

# Now, use the created dataset with metadata to do some filtering:
filtered = (
    wds.filter(func.string.length("laion.txt") > 5)
    .filter(func.array.length(func.string.split("laion.txt", " ")) > 2)
    .filter(func.least("laion.json.original_width", "laion.json.original_height") > 200)
    .filter(
        func.greatest("laion.json.original_width", "laion.json.original_height")
        / func.least("laion.json.original_width", "laion.json.original_height")
        < 3.0
    )
    .persist()
)

filtered.show(3)

print(f"wds count:      {wds.count():>6}")
print(f"filtered count: {filtered.count():>6}")
