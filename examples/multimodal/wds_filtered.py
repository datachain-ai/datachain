"""
Example demonstrates working with WebDataset format in DataChain.
It is a TAR archive, DataChain can read files inside w/o extracting them.

`process_webdataset` is a utility function that processes tar files,
it accepts a `spec` argument that defines the structure of the dataset.
(in this example we use `WDSLaion` spec that was created to parse LAION
dataset), but it can be WDSBasic or any other subclass of it.
"""

import datachain as dc
import datachain.error
from datachain import C, func
from datachain.lib.webdataset import process_webdataset
from datachain.lib.webdataset_laion import WDSLaion

# In LAION webdataset, there are JSON and text files with metadata.
# and actual images. `process_webdataset(spec=WDSLaion)` below extracts
# metadata from JSON files and text from text files, and creates a dataset.
name = "wds"
try:
    wds = dc.read_dataset(name)
except datachain.error.DatasetNotFoundError:
    wds = (
        dc.read_storage("gs://datachain-demo/datacomp-small/shards", anon=True)
        .filter(C("file.path").glob("*/00000000.tar"))
        .settings(cache=True)
        .gen(laion=process_webdataset(spec=WDSLaion))
        .save(name)
    )

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
