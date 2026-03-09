from typing import TYPE_CHECKING, Any

from datachain.lib.data_model import dict_to_data_model
from datachain.query import Session

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    from datachain.lib.data_model import DataType
    from datachain.lib.hf import HFDatasetType

    from .datachain import DataChain

    P = ParamSpec("P")


def read_hf(
    dataset: "HFDatasetType",
    *args: Any,
    session: Session | None = None,
    settings: dict | None = None,
    column: str = "",
    model_name: str = "",
    limit: int = 0,
    **kwargs: Any,
) -> "DataChain":
    """Generate chain from Hugging Face Hub dataset.

    Parameters:
        dataset: Path or name of the dataset to read from Hugging Face Hub,
            or an instance of `datasets.Dataset`-like object.
        args: Additional positional arguments to pass to `datasets.load_dataset`.
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        column: Generated object column name.
        model_name: Generated model name.
        limit: The maximum number of items to read from the HF dataset.
            Applies `take(limit)` to `datasets.load_dataset`.
            Defaults to 0 (no limit).
        kwargs: Parameters to pass to `datasets.load_dataset`.

    Example:
        Load from Hugging Face Hub:
        ```py
        import datachain as dc
        chain = dc.read_hf("beans", split="train")
        ```

        Generate chain from loaded dataset:
        ```py
        from datasets import load_dataset
        ds = load_dataset("beans", split="train")
        import datachain as dc
        chain = dc.read_hf(ds)
        ```

        Streaming with limit, for large datasets:
        ```py
        import datachain as dc
        ds = dc.read_hf("beans", split="train", streaming=True, limit=10)
        ```

        or use HF split syntax (not supported if streaming is enabled):
        ```py
        import datachain as dc
        ds = dc.read_hf("beans", split="train[%10]")
        ```
    """
    from datachain.lib.hf import HFGenerator, get_output_schema, stream_splits

    from .values import read_values

    output: dict[str, DataType] = {}
    ds_dict = stream_splits(dataset, *args, **kwargs)
    if len(ds_dict) > 1:
        output = {"split": str}

    model_name = model_name or column or ""
    hf_features = next(iter(ds_dict.values())).features
    hf_output, normalized_names = get_output_schema(hf_features, list(output.keys()))
    output = output | hf_output
    model = dict_to_data_model(model_name, output, list(normalized_names.values()))
    if column:
        output = {column: model}

    chain = read_values(split=list(ds_dict.keys()), session=session, settings=settings)
    return chain.gen(HFGenerator(dataset, model, limit, *args, **kwargs), output=output)
