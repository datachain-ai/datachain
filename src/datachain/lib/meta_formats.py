import csv
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import jmespath as jsp
from pydantic import BaseModel, ConfigDict, Field, ValidationError  # noqa: F401

from datachain import json
from datachain.lib.data_model import DataModel  # noqa: F401
from datachain.lib.file import TextFile


class UserModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


def generate_uuid():
    return uuid.uuid4()  # Generates a random UUID.


# JSON decoder
def load_json_from_string(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        print(f"Failed to decode JSON: {json_string} is not formatted correctly.")
        return None


# Validate and reduce JSON
def process_json(data_string, jmespath):
    json_dict = load_json_from_string(data_string)
    if jmespath:
        json_dict = jsp.search(jmespath, json_dict)
    return json_dict


def pick_datamodel_target_python_version(python_version_enum, major: int, minor: int):
    candidate_name = f"PY_{major}{minor}"
    candidate = getattr(python_version_enum, candidate_name, None)
    if candidate is not None:
        return candidate

    supported_same_major: list[tuple[int, object]] = []
    major_prefix = f"PY_{major}"
    for enum_member in list(python_version_enum):
        name = getattr(enum_member, "name", "")
        if not isinstance(name, str) or not name.startswith(major_prefix):
            continue

        remainder = name.removeprefix(major_prefix)
        if not remainder.isdigit():
            continue

        supported_minor = int(remainder)
        supported_same_major.append((supported_minor, enum_member))

    if supported_same_major:
        supported_same_major.sort(key=lambda t: t[0])
        not_newer_than_runtime = [t for t in supported_same_major if t[0] <= minor]
        if not_newer_than_runtime:
            return not_newer_than_runtime[-1][1]
        return supported_same_major[0][1]

    return list(python_version_enum)[-1]


def gen_datamodel_code(
    source_file, format="json", jmespath=None, model_name=None
) -> str:
    """Generates Python code with Pydantic models that corresponds
    to the provided JSON, CSV, or JSONL file.
    It support root JSON arrays (samples the first entry).
    """
    data_string = ""
    # using uiid to get around issue #1617
    if not model_name:
        # comply with Python class names
        uid_str = str(generate_uuid()).replace("-", "")
        model_name = f"Model{format}{uid_str}"

    with source_file.open() as fd:  # CSV can be larger than memory
        if format == "csv":
            data_string += fd.readline().replace("\r", "")
            data_string += fd.readline().replace("\r", "")
        elif format == "jsonl":
            data_string = fd.readline().replace("\r", "")
        else:
            data_string = fd.read()  # other meta must fit into RAM

    if format in ("json", "jsonl"):
        json_object = process_json(data_string, jmespath)
        if format == "json" and isinstance(json_object, list):
            json_object = json_object[0]  # sample the 1st object from JSON array
        if format == "jsonl":
            format = "json"  # treat json line as plain JSON in auto-schema
        data_string = json.dumps(json_object)

    import datamodel_code_generator

    target_python_version = pick_datamodel_target_python_version(
        datamodel_code_generator.PythonVersion,
        sys.version_info.major,
        sys.version_info.minor,
    )

    input_file_types = {i.value: i for i in datamodel_code_generator.InputFileType}
    input_file_type = input_file_types[format]
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "model.py"
        datamodel_code_generator.generate(
            data_string,
            input_file_type=input_file_type,
            output=output,
            target_python_version=target_python_version,
            base_class="datachain.lib.meta_formats.UserModel",
            class_name=model_name,
            additional_imports=["datachain.lib.data_model.DataModel"],
            use_standard_collections=True,
        )
        epilogue = f"""
DataModel.register({model_name})
spec = {model_name}
"""
        return output.read_text() + epilogue


#
# UDF mapper which calls chain in the setup to infer the dynamic schema
#
def read_meta(  # noqa: C901
    spec=None,
    schema_from=None,
    format="json",
    jmespath=None,
    model_name=None,
    nrows=None,
) -> Callable:
    from datachain import read_storage

    if schema_from:
        file = read_storage(schema_from, type="text").limit(1).to_values("file")[0]
        model_code = gen_datamodel_code(
            file, format=format, jmespath=jmespath, model_name=model_name
        )
        assert isinstance(model_code, str)

        # Below 'spec' should be a dynamically converted DataModel from Pydantic
        if not spec:
            gl = globals()
            exec(model_code, gl)  # type: ignore[arg-type] # noqa: S102
            spec = gl["spec"]

    if not spec and not schema_from:
        raise ValueError(
            "Must provide a static schema in spec: or metadata sample in schema_from:"
        )

    #
    # UDF mapper parsing a JSON or CSV file using schema spec
    #

    def parse_data(
        file: TextFile,
        data_model=spec,
        format=format,
        jmespath=jmespath,
        nrows=nrows,
    ) -> Iterator[spec]:
        def validator(json_object: dict, nrow=0) -> spec:
            json_string = json.dumps(json_object)
            try:
                data_instance = data_model.model_validate_json(json_string)
                yield data_instance
            except ValidationError as e:
                print(f"Validation error occurred in row {nrow} file {file.name}:", e)

        if format == "csv":
            with (
                file.open() as fd
            ):  # TODO: if schema is statically given, should allow CSV without headers
                reader = csv.DictReader(fd)
                for row in reader:  # CSV can be larger than memory
                    yield from validator(row)

        if format == "json":
            try:
                with file.open() as fd:  # JSON must fit into RAM
                    data_string = fd.read()
            except OSError as e:
                print(f"An unexpected file error occurred in file {file.name}: {e}")
            json_object = process_json(data_string, jmespath)
            if not isinstance(json_object, list):
                yield from validator(json_object)

            else:
                nrow = 0
                for json_dict in json_object:
                    nrow = nrow + 1
                    if nrows is not None and nrow > nrows:
                        return
                    yield from validator(json_dict, nrow)

        if format == "jsonl":
            try:
                nrow = 0
                with file.open() as fd:
                    data_string = fd.readline().replace("\r", "")
                    while data_string:
                        nrow = nrow + 1
                        if nrows is not None and nrow > nrows:
                            return
                        json_object = process_json(data_string, jmespath)
                        data_string = fd.readline()
                        yield from validator(json_object, nrow)
            except OSError as e:
                print(f"An unexpected file error occurred in file {file.name}: {e}")

    return parse_data
