"""Live functional check for the `datachain.llm` operations.

`dc-llm-query.py` is a focused demo; this script runs every `llm` operation over
a tiny in-memory dataset and asserts on the results, so a break in the real
request or response wire format is caught here rather than only by the unit-test
fakes. It is exercised by the examples CI job.

Requires: an ANTHROPIC_API_KEY (chat, vision and PDF) and an OPENAI_API_KEY
(embeddings).
"""

from io import BytesIO

from PIL import Image as PILImage
from PIL import ImageDraw
from pydantic import BaseModel

import datachain as dc
from datachain import llm

CHAT = "anthropic/claude-haiku-4-5"
EMBED = "openai/text-embedding-3-small"

POSITIVE = "The bot understood me instantly and booked the whole trip. Wonderful."
NEGATIVE = "It ignored every question and then gave up. Completely useless."


class Review(BaseModel):
    sentiment: str
    summary: str


class Sentence(BaseModel):
    text: str


def _png(color: str) -> bytes:
    buf = BytesIO()
    PILImage.new("RGB", (64, 64), color).save(buf, format="PNG")
    return buf.getvalue()


def _pdf(text: str) -> bytes:
    img = PILImage.new("RGB", (480, 160), "white")
    ImageDraw.Draw(img).text((20, 60), text, fill="black")
    buf = BytesIO()
    img.save(buf, format="PDF")
    return buf.getvalue()


def check_text() -> None:
    rows = dc.read_values(review=[POSITIVE, NEGATIVE]).settings(llm=CHAT)

    summary = rows.map(out=llm.complete("review", "Summarize in one word.")).to_values(
        "out"
    )
    assert all(isinstance(s, str) and s for s in summary), summary

    label = rows.map(
        out=llm.classify("review", into=["positive", "negative"])
    ).to_values("out")
    assert label == ["positive", "negative"], label

    score = rows.map(out=llm.score("review", "How positive is this, 0..1?")).to_values(
        "out"
    )
    assert all(isinstance(s, float) and 0.0 <= s <= 1.0 for s in score), score

    print("text: complete/classify/score ok", label, score)


def check_structured() -> None:
    extracted = (
        dc.read_values(review=[NEGATIVE])
        .settings(llm=CHAT)
        .map(
            out=llm.complete("review", "Extract sentiment and summary.", schema=Review)
        )
        .to_values("out")
    )
    assert isinstance(extracted[0], Review), extracted
    assert extracted[0].sentiment and extracted[0].summary, extracted[0]

    # list[Model] is a 1:N stream consumed by .gen()
    sentences = (
        dc.read_values(review=[f"{POSITIVE} {NEGATIVE}"])
        .settings(llm=CHAT)
        .gen(
            out=llm.complete("review", "One item per sentence.", schema=list[Sentence])
        )
        .to_values("out")
    )
    assert len(sentences) >= 2, sentences
    assert all(isinstance(s, Sentence) and s.text for s in sentences), sentences

    print("structured: schema + gen list ok", len(sentences))


def check_embed() -> None:
    vectors = (
        dc.read_values(review=[POSITIVE, NEGATIVE])
        .map(vec=llm.embed("review", llm=EMBED))
        .to_values("vec")
    )
    assert all(v and all(isinstance(x, float) for x in v) for v in vectors), vectors
    assert len(vectors[0]) == len(vectors[1]), [len(v) for v in vectors]
    print("embed: ok dim", len(vectors[0]))


def check_image() -> None:
    color = (
        dc.read_values(img=[_png("red")])
        .settings(llm=CHAT)
        .map(
            out=llm.complete(
                "img", "Name the dominant color in one word.", type="image"
            )
        )
        .to_values("out")
    )
    # A vision model should return a non-empty description; exact wording varies,
    # so this validates the image wire format, not the model's color judgement.
    assert isinstance(color[0], str) and color[0], color
    print("image: ok", color[0])


def check_document() -> None:
    text = (
        dc.read_values(doc=[_pdf("DataChain")])
        .settings(llm=CHAT)
        .map(out=llm.complete("doc", "What does this document say?", type="document"))
        .to_values("out")
    )
    assert isinstance(text[0], str) and text[0], text
    print("document: ok", text[0][:40])


def check_context() -> None:
    judged = (
        dc.read_values(review=[NEGATIVE], rubric=["Penalize unanswered questions."])
        .settings(llm=CHAT)
        .map(out=llm.complete("review", "Judge the dialogue.", context="rubric"))
        .to_values("out")
    )
    assert isinstance(judged[0], str) and judged[0], judged
    print("context: ok")


def check_usage() -> None:
    toks = (
        dc.read_values(review=[POSITIVE, NEGATIVE])
        .settings(llm=CHAT)
        .map(
            llm.complete("review", "Summarize in one word.", include_usage=True),
            output={"out": str, "tok": llm.Usage},
        )
        .to_values("tok")
    )
    assert all(t.input_tokens > 0 for t in toks), toks
    assert all(t.output_tokens > 0 for t in toks), toks
    print("usage: ok", toks[0].input_tokens, toks[0].output_tokens)


if __name__ == "__main__":
    check_text()
    check_structured()
    check_embed()
    check_image()
    check_document()
    check_context()
    check_usage()
    print("all datachain.llm live checks passed")
