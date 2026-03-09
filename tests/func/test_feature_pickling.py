import json
from typing import Literal

import cloudpickle
import pytest
from pydantic import BaseModel

import datachain as dc
from datachain.lib.data_model import DataModel
from datachain.lib.file import File
from datachain.lib.signal_schema import create_feature_model


class FileInfo(DataModel):
    file_name: str = ""
    byte_size: int = 0


class TextBlock(DataModel):
    text: str = ""
    type: str = "text"


class AIMessage(DataModel):
    id: str = ""
    content: list[TextBlock]
    model: str = "Test AI Model"
    type: Literal["message"] = "message"
    input_file_info: FileInfo = FileInfo()


def file_to_message(file):
    if not isinstance(file, File):
        return AIMessage()
    name = file.name
    size = file.size
    return AIMessage(
        id=name,
        content=[TextBlock(text=json.dumps({"file_name": name}))],
        input_file_info=FileInfo(file_name=name, byte_size=size),
    )


def common_df_asserts(df):
    assert df["file"]["path"].tolist() == ["cats/cat1", "cats/cat2"]
    assert df["file"]["size"].tolist() == [4, 4]
    assert df["message"]["id"].tolist() == ["cat1", "cat2"]
    mc = df["message"]["content"].tolist()
    # This is needed due to differences in how JSON is stored
    # between SQLite and ClickHouse
    if isinstance(mc[0][0], str):
        mc_parsed = [[json.loads(m[0])] for m in mc]
    else:
        mc_parsed = mc
    assert mc_parsed == [
        [{"text": '{"file_name": "cat1"}', "type": "text"}],
        [{"text": '{"file_name": "cat2"}', "type": "text"}],
    ]
    assert df["message"]["type"].tolist() == ["message", "message"]
    assert df["message"]["input_file_info"]["file_name"].tolist() == ["cat1", "cat2"]
    assert df["message"]["input_file_info"]["byte_size"].tolist() == [4, 4]


def sort_df_for_tests(df):
    # Sort the dataframe to avoid a PerformanceWarning about unsorted indexing.
    df.sort_index(axis=0, inplace=True, sort_remaining=True)
    df.sort_index(axis=1, inplace=True, sort_remaining=True)
    return df.sort_values(("file", "path")).reset_index(drop=True)


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_feature_udf_parallel(cloud_test_catalog_tmpfile):
    ctc = cloud_test_catalog_tmpfile
    source = ctc.src_uri

    import tests.func.test_feature_pickling as tfp  # noqa: PLW0406

    # This emulates having the functions and classes declared in the __main__ script.
    cloudpickle.register_pickle_by_value(tfp)

    chain = (
        dc.read_storage(source, type="text", session=ctc.session)
        .filter(dc.C("file.path").glob("*cat*"))
        .settings(parallel=2)
        .map(
            message=file_to_message,
            output=AIMessage,
        )
    )

    df = chain.to_pandas()

    df = sort_df_for_tests(df)

    common_df_asserts(df)
    assert df["message"]["model"].tolist() == ["Test AI Model", "Test AI Model"]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_feature_udf_parallel_local(cloud_test_catalog_tmpfile):
    ctc = cloud_test_catalog_tmpfile
    source = ctc.src_uri

    class FileInfoLocal(DataModel):
        file_name: str = ""
        byte_size: int = 0

    class TextBlockLocal(DataModel):
        text: str = ""
        type: str = "text"

    class AIMessageLocal(DataModel):
        id: str = ""
        content: list[TextBlockLocal]
        model: str = "Test AI Model Local"
        type: Literal["message"] = "message"
        input_file_info: FileInfoLocal = FileInfoLocal()

    import tests.func.test_feature_pickling as tfp  # noqa: PLW0406

    # This emulates having the functions and classes declared in the __main__ script.
    cloudpickle.register_pickle_by_value(tfp)

    chain = (
        dc.read_storage(source, type="text", session=ctc.session)
        .filter(dc.C("file.path").glob("*cat*"))
        .settings(parallel=2)
        .map(
            message=lambda file: (
                AIMessageLocal(
                    id=(name := file.name),
                    content=[TextBlockLocal(text=json.dumps({"file_name": name}))],
                    input_file_info=FileInfoLocal(file_name=name, byte_size=file.size),
                )
                if isinstance(file, File)
                else AIMessageLocal()
            ),
            output=AIMessageLocal,
        )
    )

    df = chain.to_pandas()

    df = sort_df_for_tests(df)

    common_df_asserts(df)
    assert df["message"]["model"].tolist() == [
        "Test AI Model Local",
        "Test AI Model Local",
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_feature_udf_parallel_local_pydantic(cloud_test_catalog_tmpfile):
    ctc = cloud_test_catalog_tmpfile
    source = ctc.src_uri

    class FileInfoLocalPydantic(BaseModel):
        file_name: str = ""
        byte_size: int = 0

    class TextBlockLocalPydantic(BaseModel):
        text: str = ""
        type: str = "text"

    class AIMessageLocalPydantic(BaseModel):
        id: str = ""
        content: list[TextBlockLocalPydantic]
        model: str = "Test AI Model Local Pydantic"
        type: Literal["message"] = "message"
        input_file_info: FileInfoLocalPydantic = FileInfoLocalPydantic()

    import tests.func.test_feature_pickling as tfp  # noqa: PLW0406

    # This emulates having the functions and classes declared in the __main__ script.
    cloudpickle.register_pickle_by_value(tfp)

    chain = (
        dc.read_storage(source, type="text", session=ctc.session)
        .filter(dc.C("file.path").glob("*cat*"))
        .settings(parallel=2)
        .map(
            message=lambda file: (
                AIMessageLocalPydantic(
                    id=(name := file.name),
                    content=[
                        TextBlockLocalPydantic(text=json.dumps({"file_name": name}))
                    ],
                    input_file_info=FileInfoLocalPydantic(
                        file_name=name, byte_size=file.size
                    ),
                )
                if isinstance(file, File)
                else AIMessageLocalPydantic()
            ),
            output=AIMessageLocalPydantic,
        )
    )

    df = chain.to_pandas()

    df = sort_df_for_tests(df)

    common_df_asserts(df)
    assert df["message"]["model"].tolist() == [
        "Test AI Model Local Pydantic",
        "Test AI Model Local Pydantic",
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_feature_udf_parallel_local_pydantic_old(cloud_test_catalog_tmpfile):
    ctc = cloud_test_catalog_tmpfile
    catalog = ctc.catalog
    source = ctc.src_uri
    catalog.index([source])

    class FileInfoLocalPydantic(BaseModel):
        file_name: str = ""
        byte_size: int = 0

    class TextBlockLocalPydantic(BaseModel):
        text: str = ""
        type: str = "text"

    class AIMessageLocalPydantic(BaseModel):
        id: str = ""
        content: list[TextBlockLocalPydantic]
        model: str = "Test AI Model Local Pydantic Old"
        type: Literal["message"] = "message"
        input_file_info: FileInfoLocalPydantic = FileInfoLocalPydantic()

    import tests.func.test_feature_pickling as tfp  # noqa: PLW0406

    # This emulates having the functions and classes declared in the __main__ script.
    cloudpickle.register_pickle_by_value(tfp)

    chain = (
        dc.read_storage(source, type="text", session=ctc.session)
        .filter(dc.C("file.path").glob("*cat*"))
        .settings(parallel=2)
        .map(
            message=lambda file: (
                AIMessageLocalPydantic(
                    id=(name := file.name),
                    content=[
                        TextBlockLocalPydantic(text=json.dumps({"file_name": name}))
                    ],
                    input_file_info=FileInfoLocalPydantic(
                        file_name=name, byte_size=file.size
                    ),
                )
                if isinstance(file, File)
                else AIMessageLocalPydantic()
            ),
            output=AIMessageLocalPydantic,
        )
    )

    df = chain.to_pandas()

    df = sort_df_for_tests(df)

    common_df_asserts(df)
    assert df["message"]["model"].tolist() == [
        "Test AI Model Local Pydantic Old",
        "Test AI Model Local Pydantic Old",
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_feature_udf_parallel_dynamic(cloud_test_catalog_tmpfile):
    ctc = cloud_test_catalog_tmpfile
    source = ctc.src_uri
    session = ctc.session

    file_info_dynamic = create_feature_model(
        "FileInfoDynamic",
        {
            "file_name": (str, ""),
            "byte_size": (int, 0),
        },
    )

    text_block_dynamic = create_feature_model(
        "TextBlockDynamic",
        {
            "text": (str, ""),
            "type": (str, "text"),
        },
    )

    ai_message_dynamic = create_feature_model(
        "AIMessageDynamic",
        {
            "id": (str, ""),
            "content": list[text_block_dynamic],
            "model": (str, "Test AI Model Dynamic"),
            "type": (Literal["message"], "message"),
            "input_file_info": (file_info_dynamic, file_info_dynamic()),
        },
    )

    import tests.func.test_feature_pickling as tfp  # noqa: PLW0406

    # This emulates having the functions and classes declared in the __main__ script.
    cloudpickle.register_pickle_by_value(tfp)

    chain = (
        dc.read_storage(source, type="text", session=session)
        .filter(dc.C("file__path").glob("*cat*"))
        .settings(parallel=2)
        .map(
            message=lambda file: (
                ai_message_dynamic(
                    id=(name := file.name),
                    content=[text_block_dynamic(text=json.dumps({"file_name": name}))],
                    input_file_info=file_info_dynamic(
                        file_name=name, byte_size=file.size
                    ),
                )
                if isinstance(file, File)
                else ai_message_dynamic()
            ),
            output=ai_message_dynamic,
        )
    )

    df = chain.to_pandas()

    df = sort_df_for_tests(df)

    common_df_asserts(df)
    assert df["message"]["model"].tolist() == [
        "Test AI Model Dynamic",
        "Test AI Model Dynamic",
    ]
