import glob
import os
import subprocess
import sys

import pytest

get_started_examples = sorted(glob.glob("examples/get_started/**/*.py", recursive=True))

llm_and_nlp_examples = sorted(glob.glob("examples/llm_and_nlp/**/*.py", recursive=True))

multimodal_examples = sorted(glob.glob("examples/multimodal/**/*.py", recursive=True))

incremental_processing_examples = sorted(
    glob.glob("examples/incremental_processing/**/*.py", recursive=True)
)

computer_vision_examples = sorted(
    [
        filename
        for filename in glob.glob("examples/computer_vision/**/*.py", recursive=True)
        # fashion product images tutorial out of scope
        # and hf download painfully slow
        if "image_desc" not in filename and "fashion_product_images" not in filename
    ]
)


def smoke_test(example: str, env: dict | None = None):
    try:
        completed_process = subprocess.run(  # noqa: S603
            [sys.executable, example],
            env={**os.environ, **(env or {})},
            capture_output=True,
            cwd=os.path.abspath(os.path.join(__file__, "..", "..", "..")),
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Example failed: {example}")
        print()
        print()
        print("stdout:")
        print(e.stdout.decode("utf-8"))
        print()
        print()
        print("stderr:")
        print(e.stderr.decode("utf-8"))
        pytest.fail("subprocess returned a non-zero exit code")

    example_has_some_output = bool(completed_process.stdout or completed_process.stderr)
    assert example_has_some_output


@pytest.mark.examples
@pytest.mark.get_started
@pytest.mark.parametrize("example", get_started_examples)
def test_get_started_examples(example):
    smoke_test(example)


@pytest.mark.examples
@pytest.mark.llm_and_nlp
@pytest.mark.parametrize("example", llm_and_nlp_examples)
def test_llm_and_nlp_examples(example):
    # TODO fix anthropic api key
    pytest.skip("Skipping because of outdated ANTHROPIC_API_KEY")
    name = os.path.basename(example)
    if "hf-" in name:
        import huggingface_hub

        if not huggingface_hub.get_token():
            pytest.skip("Hugging Face token not set")
    if "claude" in name and "ANTHROPIC_API_KEY" not in os.environ:
        pytest.skip("ANTHROPIC_API_KEY not set")
    smoke_test(example)


@pytest.mark.examples
@pytest.mark.multimodal
@pytest.mark.parametrize("example", multimodal_examples)
def test_multimodal(example):
    if "openai" in os.path.basename(example) and "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY not set")
    smoke_test(
        example,
        {
            "IMAGE_TARS": "gs://datachain-demo/datacomp-small/shards/00001286.tar",
            "PARQUET_METADATA": "gs://datachain-demo/datacomp-small/metadata/036d6b9ae87a00e738f8fc554130b65b.parquet",
            "NPZ_METADATA": "gs://datachain-demo/datacomp-small/metadata/036d6b9ae87a00e738f8fc554130b65b.npz",
        },
    )


@pytest.mark.examples
@pytest.mark.incremental_processing
@pytest.mark.parametrize("example", incremental_processing_examples)
def test_incremental_processing_examples(example):
    smoke_test(example)


@pytest.mark.examples
@pytest.mark.computer_vision
@pytest.mark.parametrize("example", computer_vision_examples)
def test_computer_vision_examples(example):
    smoke_test(example)
