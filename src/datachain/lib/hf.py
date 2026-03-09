try:
    from datasets import (
        Array2D,
        Array3D,
        Array4D,
        Array5D,
        Audio,
        ClassLabel,
        Dataset,
        DatasetDict,
        Image,
        IterableDataset,
        IterableDatasetDict,
        List,
        Value,
        load_dataset,
    )
    from datasets.features.features import Features, string_to_arrow
    from datasets.features.image import image_to_bytes

except ImportError as exc:
    raise ImportError(
        "Missing dependencies for huggingface datasets.\n"
        "To install run:\n\n"
        "  pip install 'datachain[hf]'\n"
    ) from exc

from io import BytesIO
from typing import TYPE_CHECKING, Any, TypeAlias

import PIL
from tqdm.auto import tqdm

from datachain.lib.arrow import arrow_type_mapper
from datachain.lib.data_model import DataModel, DataType, dict_to_data_model
from datachain.lib.udf import Generator
from datachain.lib.utils import normalize_col_names

if TYPE_CHECKING:
    import pyarrow as pa
    from pydantic import BaseModel


HFDatasetType: TypeAlias = (
    str | DatasetDict | Dataset | IterableDatasetDict | IterableDataset
)


class HFClassLabel(DataModel):
    string: str
    integer: int

    def read(self):
        return self.integer


class HFImage(DataModel):
    img: bytes

    def read(self):
        return PIL.Image.open(BytesIO(self.img))


class HFAudio(DataModel):
    array: list[float]
    sampling_rate: int


class HFGenerator(Generator):
    def __init__(
        self,
        ds: HFDatasetType,
        output_schema: type["BaseModel"],
        limit: int = 0,
        *args,
        **kwargs,
    ):
        """
        Generator for chain from Hugging Face datasets.

        Parameters:

            ds : Path or name of the dataset to read from Hugging Face Hub,
                or an instance of `datasets.Dataset`-like object.
            limit : Limit the number of items to read from the HF dataset.
                    Defaults to 0 (no limit).
            output_schema : Pydantic model for validation.
        """
        super().__init__()
        self.ds = ds
        self.output_schema = output_schema
        self.limit = limit
        self.args = args
        self.kwargs = kwargs

    def setup(self):
        self.ds_dict = stream_splits(self.ds, *self.args, **self.kwargs)

    def process(self, split: str = ""):
        desc = "Parsed Hugging Face dataset"
        ds = self.ds_dict[split]
        if self.limit > 0:
            ds = ds.take(self.limit)
        if split:
            desc += f" split '{split}'"
        model_fields = self.output_schema._model_fields_by_aliases()  # type: ignore[attr-defined]
        with tqdm(desc=desc, unit=" rows", leave=False) as pbar:
            for row in ds:
                output_dict = {}
                if split and "split" in self.output_schema.model_fields:
                    output_dict["split"] = split
                for name, feat in ds.features.items():
                    normalized_name, info = model_fields[name]
                    anno = info.annotation
                    output_dict[normalized_name] = convert_feature(
                        row[name], feat, anno
                    )
                yield self.output_schema(**output_dict)
                pbar.update(1)


def stream_splits(ds: HFDatasetType, *args, **kwargs):
    if isinstance(ds, str):
        ds = load_dataset(ds, *args, **kwargs)
    if isinstance(ds, (DatasetDict, IterableDatasetDict)):
        return ds
    return {"": ds}


def convert_feature(val: Any, feat: Any, anno: Any) -> Any:
    if isinstance(feat, (Value, Array2D, Array3D, Array4D, Array5D, List)):
        return val
    if isinstance(feat, ClassLabel):
        return HFClassLabel(string=feat.names[val], integer=val)
    if isinstance(feat, dict):
        sdict = {}
        model_fields = anno._model_fields_by_aliases()  # type: ignore[attr-defined]
        for sname in val:
            sfeat = feat[sname]
            norm_name, info = model_fields[sname]
            sanno = info.annotation
            if isinstance(val[sname], list):
                sdict[norm_name] = [
                    convert_feature(v, sfeat, sanno) for v in val[sname]
                ]
            else:
                sdict[norm_name] = convert_feature(val[sname], sfeat, sanno)
        return anno(**sdict)
    if isinstance(feat, Image):
        if isinstance(val, dict):
            return HFImage(img=val["bytes"])
        return HFImage(img=image_to_bytes(val))
    if isinstance(feat, Audio):
        return HFAudio(array=val["array"], sampling_rate=val["sampling_rate"])


def get_output_schema(
    features: Features, existing_column_names: list[str] | None = None
) -> tuple[dict[str, DataType], dict[str, str]]:
    """
    Generate UDF output schema from Hugging Face datasets features. It normalizes the
    column names and returns a mapping of normalized names to original names along with
    the data types. `existing_column_names` is the list of column names that already
    exist in the dataset (to avoid name collisions due to normalization).
    """
    existing_column_names = existing_column_names or []
    fields_dict = {}
    normalized_names = normalize_col_names(
        existing_column_names + list(features.keys())
    )
    # List of tuple(str, str) for HF dataset feature names, (normalized, original)
    new_feature_names = list(normalized_names.items())[len(existing_column_names) :]
    for idx, feat in enumerate(features.items()):
        name, val = feat
        fields_dict[new_feature_names[idx][0]] = _feature_to_chain_type(name, val)
    return fields_dict, normalized_names


def _feature_to_chain_type(name: str, val: Any) -> DataType:  # noqa: PLR0911
    if isinstance(val, Value):
        return arrow_type_mapper(val.pa_type)
    if isinstance(val, ClassLabel):
        return HFClassLabel
    if isinstance(val, dict):
        sequence_dict = {}
        for sname, sval in val.items():
            dtype = _feature_to_chain_type(sname, sval)
            sequence_dict[sname] = dtype  # type: ignore[valid-type]
        return dict_to_data_model(f"HFDataModel_{name}", sequence_dict)  # type: ignore[arg-type]
    if isinstance(val, List):
        return list[_feature_to_chain_type(name, val.feature)]  # type: ignore[arg-type,misc,return-value]
    if isinstance(val, Array2D):
        dtype = arrow_type_mapper(string_to_arrow(val.dtype))
        return list[list[dtype]]  # type: ignore[valid-type]
    if isinstance(val, Array3D):
        dtype = arrow_type_mapper(string_to_arrow(val.dtype))
        return list[list[list[dtype]]]  # type: ignore[valid-type]
    if isinstance(val, Array4D):
        dtype = arrow_type_mapper(string_to_arrow(val.dtype))
        return list[list[list[list[dtype]]]]  # type: ignore[valid-type]
    if isinstance(val, Array5D):
        dtype = arrow_type_mapper(string_to_arrow(val.dtype))
        return list[list[list[list[list[dtype]]]]]  # type: ignore[valid-type]
    if isinstance(val, Image):
        return HFImage
    if isinstance(val, Audio):
        return HFAudio
    raise TypeError(f"Unknown huggingface datasets type {type(val)}")


def schema_from_arrow(schema: "pa.Schema"):
    return Features.from_arrow_schema(schema)
