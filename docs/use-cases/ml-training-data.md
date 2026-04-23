---
title: ML Training Data
---

# ML Training Data

DataChain handles the full pipeline from raw storage through enrichment to training -- without writing intermediate files.

## Data Curation with a Local Model

```python
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

positive_chain = chain.filter(dc.C("is_positive") == True)
positive_chain.to_storage("./output")
print(f"{positive_chain.count()} files were exported")
```

## PyTorch DataLoader Integration

```python
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
import datachain as dc

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

ds = (
    dc.read_storage("gs://bucket/images/", type="image", anon=True)
    .map(label=lambda name: name.split(".")[0], params=["file.path"])
    .select("file", "label")
    .to_pytorch(
        transform=processor.image_processor,
        tokenizer=processor.tokenizer,
    )
)

loader = DataLoader(ds, batch_size=16)
```

## Train/Test Split

```python
import datachain as dc
from datachain.toolkit import train_test_split

chain = dc.read_dataset("labeled_images")

train, test = train_test_split(chain, [0.7, 0.3])
train, test, val = train_test_split(chain, [0.7, 0.2, 0.1])

# Each split is a full chain
train.save("train_split")
test.save("test_split")

# Or feed directly to PyTorch
train_loader = DataLoader(train.to_pytorch(...), batch_size=32)
```

## Embedding Computation at Scale

```python
import numpy as np
import datachain as dc
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("clip-ViT-B-32")

def clip_embedding(file: dc.ImageFile) -> list[float]:
    img = file.read().convert("RGB")
    emb = model.encode(img).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb.tolist()

chain = (
    dc.read_storage("s3://bucket/images/", type="image")
    .settings(parallel=8, cache=True)
    .map(emb=clip_embedding)
    .save("image_embeddings")
)
```

## Iterating Over Saved Datasets

```python
import datachain as dc

chain = dc.read_dataset("llm_responses")

for file, response in chain.limit(5).to_iter("file", "response"):
    status = response.choices[0].message.content[:7]
    tokens = response.usage.total_tokens
    print(f"{file.get_fs_path()}: {status}, tokens: {tokens}")
```

## Format Conversion on Export

```python
import datachain as dc

# Audio format conversion
def convert_to_mp3(file: dc.AudioFile) -> str:
    return file.save("output/", format="mp3")

dc.read_storage("s3://audio/", type="audio").map(mp3=convert_to_mp3)

# Export to multiple formats
chain = dc.read_dataset("results")
chain.to_parquet("output/results.parquet")
chain.to_csv("output/results.csv")
chain.to_json("output/results.json")
```
