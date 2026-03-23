import base64
import os
import sys

import openai

import datachain as dc
from datachain import C, File

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    print("This example requires an OpenAI API key")
    print("Add your key using the OPENAI_API_KEY environment variable.")
    sys.exit(1)


def describe_image(file: File, client: openai.OpenAI) -> tuple[str, str]:
    base64_image = base64.b64encode(file.read()).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )

        openai_description = response.choices[0].message.content or ""
        error = ""

    except openai.OpenAIError as e:
        error = str(e)
        openai_description = ""

    return (openai_description, error)


if __name__ == "__main__":
    (
        dc.read_storage("gs://datachain-demo/dogs-and-cats/", anon=True)
        .filter(C("file.path").glob("*cat*.jpg"))
        .limit(10)
        .setup(client=lambda: openai.OpenAI(api_key=OPENAI_API_KEY, timeout=30))
        .map(describe_image, output={"description": str, "error": str})
        .select("file.source", "file.path", "description", "error")
        .show()
    )
