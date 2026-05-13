---
title: Getting Started with Python
---

# Getting Started with Python

## Installation

=== "pip"

    ```bash
    pip install datachain
    ```

=== "uv"

    ```bash
    uv add datachain
    ```

## Selecting files using JSON metadata

A storage consists of images of cats and dogs
(`dog.1048.jpg`, `cat.1009.jpg`), annotated with
ground truth and model inferences in the _`json-pairs`_ format, where
each image has a matching JSON file like `cat.1009.json`:

``` json
{
    "class": "cat", "id": "1009", "num_annotators": 8,
    "inference": {"class": "dog", "confidence": 0.68}
}
```

Example of downloading only _`high-confidence cat`_ inferred images
using JSON metadata:

``` py
import datachain as dc

meta = dc.read_json("gs://datachain-demo/dogs-and-cats/*json", column="meta", anon=True)
images = dc.read_storage("gs://datachain-demo/dogs-and-cats/*jpg", anon=True)

images_id = images.map(id=lambda file: file.path.split('.')[-2])
annotated = images_id.merge(meta, on="id", right_on="meta.id")

likely_cats = annotated.filter((dc.Column("meta.inference.confidence") > 0.93) \
                               & (dc.Column("meta.inference.class_") == "cat"))
likely_cats.to_storage("high-confidence-cats/", signal="file")
```

## Data curation with a local AI model

Batch inference with a simple sentiment model using the
`transformers` library:

``` shell
pip install transformers
```

Note, `transformers` works only if `torch`, `tensorflow` >= 2.0, or `flax` are installed.

The code below downloads files from the cloud, and applies a
user-defined function to each one of them. All files with a positive
sentiment detected are then copied to the local directory.

``` py
from transformers import pipeline
import datachain as dc

classifier = pipeline("sentiment-analysis", device="cpu",
                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def is_positive_dialogue_ending(file) -> bool:
    dialogue_ending = file.read()[-512:]
    return classifier(dialogue_ending)[0]["label"] == "POSITIVE"

chain = (
   dc.read_storage("gs://datachain-demo/chatbot-KiT/",
                          column="file", type="text", anon=True)
   .settings(parallel=8, cache=True)
   .map(is_positive=is_positive_dialogue_ending)
   .save("file_response")
)

positive_chain = chain.filter(Column("is_positive") == True)
positive_chain.to_storage("./output")

print(f"{positive_chain.count()} files were exported")
```

13 files were exported

``` shell
$ ls output/datachain-demo/chatbot-KiT/
15.txt 20.txt 24.txt 27.txt 28.txt 29.txt 33.txt 37.txt 38.txt 43.txt ...
$ ls output/datachain-demo/chatbot-KiT/ | wc -l
13
```

## LLM judging chatbots

LLMs can work as universal classifiers. In the example below, we employ
a free API from Mistral to judge the [publicly
available](https://radar.kit.edu/radar/en/dataset/FdJmclKpjHzLfExE.ExpBot%2B-%2BA%2Bdataset%2Bof%2B79%2Bdialogs%2Bwith%2Ban%2Bexperimental%2Bcustomer%2Bservice%2Bchatbot)
chatbot dialogs. Please get a free Mistral API key at
<https://console.mistral.ai>

``` shell
$ pip install mistralai (Requires version >=1.0.0)
$ export MISTRAL_API_KEY=_your_key_
```

DataChain can parallelize API calls; the free Mistral tier supports up
to 4 requests at the same time.

``` py
import os
from mistralai import Mistral
import datachain as dc

PROMPT = "Was this dialog successful? Answer in a single word: Success or Failure."

def eval_dialogue(file: dc.File) -> bool:
     client = Mistral(api_key = os.environ["MISTRAL_API_KEY"])
     response = client.chat.complete(
         model="open-mixtral-8x22b",
         messages=[{"role": "system", "content": PROMPT},
                   {"role": "user", "content": file.read()}])
     result = response.choices[0].message.content
     return result.lower().startswith("success")

chain = (
   dc.read_storage("gs://datachain-demo/chatbot-KiT/", column="file", anon=True)
   .map(is_success=eval_dialogue)
   .save("mistral_files")
)

successful_chain = chain.filter(dc.Column("is_success") == True)
successful_chain.to_storage("./output_mistral")

print(f"{successful_chain.count()} files were exported")
```

With the instruction above, the Mistral model considers 31/50 files to
hold the successful dialogues:

``` shell
$ ls output_mistral/datachain-demo/chatbot-KiT/
1.txt  15.txt 18.txt 2.txt  22.txt 25.txt 28.txt 33.txt 37.txt 4.txt  41.txt ...
$ ls output_mistral/datachain-demo/chatbot-KiT/ | wc -l
31
```

## Serializing Python objects

LLM responses may contain valuable information for analytics, such as
the number of tokens used or the model performance parameters.

Instead of extracting this information from the Mistral response data
structure (class `ChatCompletionResponse`), DataChain can
serialize the entire LLM response to the internal DB:

``` py
from mistralai import Mistral
from mistralai.models import ChatCompletionResponse
import datachain as dc

PROMPT = "Was this dialog successful? Answer in a single word: Success or Failure."

def eval_dialog(file: dc.File) -> ChatCompletionResponse:
     client = MistralClient()
     return client.chat(
         model="open-mixtral-8x22b",
         messages=[{"role": "system", "content": PROMPT},
                   {"role": "user", "content": file.read()}])

chain = (
   dc.read_storage("gs://datachain-demo/chatbot-KiT/", column="file", anon=True)
   .settings(parallel=4, cache=True)
   .map(response=eval_dialog)
   .map(status=lambda response: response.choices[0].message.content.lower()[:7])
   .save("response")
)

chain.select("file.path", "status", "response.usage").show(5)

success_rate = chain.filter(dc.Column("status") == "success").count() / chain.count()
print(f"{100*success_rate:.1f}% dialogs were successful")
```

Output:

``` shell
file   status      response     response          response
path                  usage        usage             usage
              prompt_tokens total_tokens completion_tokens
0   1.txt  success           547          548                 1
1  10.txt  failure          3576         3578                 2
2  11.txt  failure           626          628                 2
3  12.txt  failure          1144         1182                38
4  13.txt  success          1100         1101                 1

[Limited by 5 rows]
64.0% dialogs were successful
```

## Iterating over Python data structures

In the previous examples, datasets were saved in the embedded database
(`SQLite` in folder `.datachain` of the working directory). These datasets were automatically versioned, and
can be accessed using `dc.read_dataset("dataset_name")`.

Here is how to retrieve a saved dataset and iterate over the objects:

``` py
import datachain as dc

chain = dc.read_dataset("response")

# Iterating one-by-one: support out-of-memory workflow
for file, response in chain.limit(5).to_iter("file", "response"):
    # verify the collected Python objects
    assert isinstance(response, ChatCompletionResponse)

    status = response.choices[0].message.content[:7]
    tokens = response.usage.total_tokens
    print(f"{file.get_fs_path()}: {status}, file size: {file.size}, tokens: {tokens}")
```

Output:

``` shell
gs://datachain-demo/chatbot-KiT/1.txt: Success, file size: 1776, tokens: 548
gs://datachain-demo/chatbot-KiT/10.txt: Failure, file size: 11576, tokens: 3578
gs://datachain-demo/chatbot-KiT/11.txt: Failure, file size: 2045, tokens: 628
gs://datachain-demo/chatbot-KiT/12.txt: Failure, file size: 3833, tokens: 1207
gs://datachain-demo/chatbot-KiT/13.txt: Success, file size: 3657, tokens: 1101
```

## Vectorized analytics over Python objects

Some operations can run inside the DB without deserialization. For
instance, let's calculate the total cost of using the LLM APIs,
assuming the Mixtral call costs $2 per 1M input tokens and $6 per 1M
output tokens:

``` py
import datachain as dc
chain = dc.read_dataset("mistral_dataset")

cost = chain.sum("response.usage.prompt_tokens")*0.000002 \
           + chain.sum("response.usage.completion_tokens")*0.000006
print(f"Spent ${cost:.2f} on {chain.count()} calls")
```

Output:

``` shell
Spent $0.08 on 50 calls
```

## PyTorch data loader

Chain results can be exported or passed directly to PyTorch dataloader.
For example, if we are interested in passing image and a label based on
file name suffix, the following code will do it:

``` py
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

import datachain as dc

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

chain = (
    dc.read_storage("gs://datachain-demo/dogs-and-cats/", type="image", anon=True)
    .map(label=lambda name: name.split(".")[0], params=["file.path"])
    .select("file", "label").to_pytorch(
        transform=processor.image_processor,
        tokenizer=processor.tokenizer,
    )
)

loader = DataLoader(chain, batch_size=1)

```

## Next Steps

- [Concepts](../concepts/index.md): understand the Dataset DB, Datasets, and the dual engine
- [Examples](../examples.md): more code examples and notebooks
- [Guides](../guide/index.md): in-depth coverage of every capability
