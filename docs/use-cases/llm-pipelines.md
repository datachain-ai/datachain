---
title: LLM Pipelines
---

# LLM Pipelines

DataChain parallelizes LLM API calls, serializes structured responses as typed columns, and tracks costs automatically.

## LLM Classification

```python
import os
from mistralai import Mistral
import datachain as dc

PROMPT = "Was this dialog successful? Answer in a single word: Success or Failure."

def eval_dialogue(file: dc.File) -> bool:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    response = client.chat.complete(
        model="open-mixtral-8x22b",
        messages=[{"role": "system", "content": PROMPT},
                  {"role": "user", "content": file.read()}])
    result = response.choices[0].message.content
    return result.lower().startswith("success")

chain = (
    dc.read_storage("gs://datachain-demo/chatbot-KiT/", column="file", anon=True)
    .settings(parallel=4)
    .map(is_success=eval_dialogue)
    .save("mistral_evaluations")
)
```

## Serializing Full LLM Responses

Instead of extracting individual fields, serialize the entire response object:

```python
from mistralai.models import ChatCompletionResponse
import datachain as dc

def eval_dialog(file: dc.File) -> ChatCompletionResponse:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    return client.chat.complete(
        model="open-mixtral-8x22b",
        messages=[{"role": "system", "content": PROMPT},
                  {"role": "user", "content": file.read()}])

chain = (
    dc.read_storage("gs://datachain-demo/chatbot-KiT/", column="file", anon=True)
    .settings(parallel=4, cache=True)
    .map(response=eval_dialog)
    .map(status=lambda response: response.choices[0].message.content.lower()[:7])
    .save("llm_responses")
)
```

Nested fields are queryable at warehouse speed:

```python
import datachain as dc

chain = dc.read_dataset("llm_responses")
chain.select("file.path", "status", "response.usage").show(5)
```

## Cost Tracking

Compute API costs without deserialization:

```python
import datachain as dc

chain = dc.read_dataset("llm_responses")

cost = (
    chain.sum("response.usage.prompt_tokens") * 0.000002
    + chain.sum("response.usage.completion_tokens") * 0.000006
)
print(f"Spent ${cost:.2f} on {chain.count()} calls")
```

## Structured Output Validation

Use Pydantic to validate LLM outputs before they enter the dataset:

```python
import datachain as dc
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    topics: list[str]

def analyze(file: dc.File, client) -> Analysis:
    resp = client.messages.create(model="claude-sonnet-4-20250514", ...)
    return Analysis.model_validate_json(resp.content[0].text)

chain = (
    dc.read_storage("s3://docs/")
    .setup(client=lambda: create_anthropic_client())
    .settings(parallel=4)
    .map(analysis=analyze)
    .save("document_analyses")
)
```

## Comparative Model Evaluation

Run two models on the same dataset and compare:

```python
import datachain as dc

base = dc.read_dataset("documents")

dc.read_dataset("documents") \
    .setup(client=lambda: model_a_client()) \
    .map(response_a=run_model_a) \
    .save("model_a_results")

dc.read_dataset("documents") \
    .setup(client=lambda: model_b_client()) \
    .map(response_b=run_model_b) \
    .save("model_b_results")

a = dc.read_dataset("model_a_results")
b = dc.read_dataset("model_b_results")
comparison = a.merge(b, on="file.path")
comparison.save("model_comparison")
```

## Setup Pattern for API Clients

Initialize expensive resources once per worker:

```python
import google.generativeai as genai
import datachain as dc

def classify(file: dc.File, model: genai.GenerativeModel) -> str:
    text = file.read_text()
    response = model.generate_content(f"Classify this document: {text}")
    return response.text

chain = (
    dc.read_storage("s3://docs/")
    .setup(model=lambda: genai.GenerativeModel("gemini-2.0-flash"))
    .settings(parallel=4)
    .map(category=classify)
    .save("classified_docs")
)
```
