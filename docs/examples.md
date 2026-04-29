---
title: Examples
---

# Examples

Runnable notebooks covering end-to-end workflows.

## Notebooks

- **Multimodal (CLIP fine-tuning)**: [GitHub](https://github.com/datachain-ai/datachain-examples/blob/main/multimodal/clip_fine_tuning.ipynb) or [Google Colab](https://colab.research.google.com/github/datachain-ai/datachain-examples/blob/main/multimodal/clip_fine_tuning.ipynb)
- **LLM evaluations**: [GitHub](https://github.com/datachain-ai/datachain-examples/blob/main/llm/llm_chatbot_evaluation.ipynb) or [Google Colab](https://colab.research.google.com/github/datachain-ai/datachain-examples/blob/main/llm/llm_chatbot_evaluation.ipynb)
- **Reading JSON metadata**: [GitHub](https://github.com/datachain-ai/datachain-examples/blob/main/formats/json-metadata-tutorial.ipynb) or [Google Colab](https://colab.research.google.com/github/datachain-ai/datachain-examples/blob/main/formats/json-metadata-tutorial.ipynb)
- **Processing video data**: [GitHub](https://github.com/datachain-ai/datachain-examples/blob/main/computer_vision/video_pose_detection_yolo/video-pose-detection-yolov11.ipynb) or [Google Colab](https://colab.research.google.com/github/datachain-ai/datachain-examples/blob/main/computer_vision/video_pose_detection_yolo/video-pose-detection-yolov11.ipynb)

## Image Captioning with BLIP

!!! example "Image Captioning with BLIP"

    Caption images from cloud storage using the BLIP Large model, with `setup()` for one-time model initialization:

    ```python
    import datachain as dc # (1)!
    from transformers import Pipeline, pipeline
    from datachain import File

    def process(file: File, pipeline: Pipeline) -> str:
        image = file.read().convert("RGB")
        return pipeline(image)[0]["generated_text"]

    chain = (
        dc.read_storage("gs://datachain-demo/newyorker_caption_contest/images", type="image", anon=True)
        .limit(5)
        .settings(cache=True)
        .setup(pipeline=lambda: pipeline("image-to-text", model="Salesforce/blip-image-captioning-large"))
        .map(scene=process)
        .persist()
    )
    ```

    1. `pip install datachain[hf]`

    ```python
    import matplotlib.pyplot as plt
    from textwrap import wrap

    count = chain.count()
    _, axes = plt.subplots(1, count, figsize=(15, 5))

    for ax, (img_file, caption) in zip(axes, chain.to_iter("file", "scene")):
        ax.imshow(img_file.read(), cmap="gray")
        ax.axis("off")
        wrapped_caption = "\n".join(wrap(caption.strip(), 40))
        ax.set_title(wrapped_caption, fontsize=10, pad=20)

    plt.tight_layout()
    plt.show()
    ```

    ![Untitled](assets/captioned_cartoons.png)
