import os

import datachain as dc
from datachain import DataModel
from datachain.lib.meta_formats import gen_datamodel_code


# Sample model for static JSON model
class LicenseModel(DataModel):
    url: str
    id: int
    name: str


# Sample model for static CSV model
class ChatDialog(DataModel):
    id: int | None = None
    count: int | None = None
    sender: str | None = None
    text: str | None = None


def main():
    # Dynamic JSONl schema from 2 objects
    uri = "gs://datachain-demo/jsonl/object.jsonl"
    jsonl_ds = dc.read_json(uri, format="jsonl", anon="True")
    jsonl_ds.show()

    # Dynamic JSON schema from 200 OpenImage json-pairs with validation errors
    uri = "gs://datachain-demo/openimages-v6-test-jsonpairs/*json"
    schema_uri = (
        "gs://datachain-demo/openimages-v6-test-jsonpairs/08392c290ecc9d2a.json"
    )
    json_pairs_ds = dc.read_json(
        uri, schema_from=schema_uri, jmespath="@", model_name="OpenImage", anon="True"
    )
    json_pairs_ds.show()

    uri = "gs://datachain-demo/coco2017/annotations_captions/"

    # Print JSON schema in Pydantic format from main COCO annotation
    chain = dc.read_storage(uri, anon=True).filter(dc.C("file.path").glob("*.json"))
    file = chain.limit(1).to_values("file")[0]
    print(gen_datamodel_code(file, jmespath="@", model_name="Coco"))

    # Static JSON schema test parsing 3/7 objects
    static_json_ds = dc.read_json(
        uri, jmespath="licenses", spec=LicenseModel, nrows=3, anon="True"
    )
    static_json_ds.show()

    # Dynamic JSON schema test parsing 5K objects
    dynamic_json_ds = dc.read_json(uri, jmespath="images", anon="True")
    print(dynamic_json_ds.to_pandas())

    # Static CSV with header schema test parsing 3.5K objects
    uri = "gs://datachain-demo/chatbot-csv/"
    static_csv_ds = dc.read_csv(uri, output=ChatDialog, column="chat", anon="True")
    static_csv_ds.print_schema()
    static_csv_ds.show()

    # Dynamic CSV with header schema test parsing 3/3M objects
    uri = "gs://datachain-demo/laion-aesthetics-csv/laion_aesthetics_1024_33M_1.csv"
    dynamic_csv_ds = dc.read_csv(uri, column="laion", nrows=3, anon="True")
    dynamic_csv_ds.print_schema()
    dynamic_csv_ds.show()


if __name__ == "__main__":
    main()

    # Force exit without cleanup to avoid hanging due to arrow issue
    print(
        "Note: script might warn about leaked semaphore at the end due to https://github.com/apache/arrow/issues/43497"
    )
    os._exit(0)
