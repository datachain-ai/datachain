from typing import TYPE_CHECKING, Any

from datachain.lib.data_model import dict_to_data_model
from datachain.lib.utils import unique_name
from datachain.query import Session

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

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

    With several splits, each row also gets its split name in a "split" signal,
    or in the first free "cN_split" if the data has a "split" column.

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

    ds_dict = stream_splits(dataset, *args, **kwargs)
    model_name = model_name or column or ""
    hf_features = next(iter(ds_dict.values())).features
    output, normalized_names = get_output_schema(hf_features)
    original_names = list(normalized_names.values())
    split_name = ""
    if len(ds_dict) > 1:
        # The data keeps its names, so on a clash the split signal is renamed.
        split_name = unique_name("split", output)
        output = {split_name: str} | output
        original_names = [split_name, *original_names]
    model = dict_to_data_model(model_name, output, original_names)
    if column:
        output = {column: model}

    chain = read_values(split=list(ds_dict.keys()), session=session, settings=settings)
    generator = HFGenerator(
        dataset, model, limit, *args, split_name=split_name, **kwargs
    )
    return chain.gen(generator, output=output)
