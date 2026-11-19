import os

import datachain as dc
from datachain.func import path
from datachain.lib.webdataset_laion import WDSLaion, process_laion_meta

IMAGE_TARS = os.getenv(
    "IMAGE_TARS", "gs://datachain-demo/datacomp-small/shards/000000[0-5]*.tar"
)
PARQUET_METADATA = os.getenv(
    "PARQUET_METADATA", "gs://datachain-demo/datacomp-small/metadata/0020f*.parquet"
)
NPZ_METADATA = os.getenv(
    "NPZ_METADATA", "gs://datachain-demo/datacomp-small/metadata/0020f*.npz"
)

wds_images = dc.read_webdataset(
    IMAGE_TARS, spec=WDSLaion, column="laion", settings={"cache": True}, anon=True
).persist()

wds_images.show(5)

wds_with_pq = (
    dc.read_parquet(PARQUET_METADATA, anon=True)
    .settings(cache=True)
    .merge(wds_images, on="uid", right_on="laion.json.uid", inner=True)
    .persist()
)

wds_with_pq.show(5)

wds_npz = (
    dc.read_storage(NPZ_METADATA, anon=True)
    .settings(cache=True)
    .gen(emd=process_laion_meta)
    .persist()
)

wds_npz.show(5)

print("Merging all metadata into a single dataset ...")

res = wds_npz.merge(
    wds_with_pq,
    on=[path.file_stem(wds_npz.c("emd.file.path")), "emd.index"],
    right_on=[path.file_stem(wds_with_pq.c("source.file.path")), "source.index"],
    inner=True,
).save("wds")

res.show(5)
