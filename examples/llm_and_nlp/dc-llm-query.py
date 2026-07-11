"""Rate chatbot dialogues with the built-in `datachain.llm` library.

This is the `dc.llm` counterpart to `claude-query.py`: instead of constructing a
provider client by hand in `.setup()`, the model call is a named operation used
inside `.map()`. The model is selected once with `.settings(llm=...)` and routed
by LiteLLM, so the same code runs against any provider by changing the string.

Requires: an ANTHROPIC_API_KEY in the environment.
"""

from pydantic import BaseModel

import datachain as dc
from datachain import C, llm

DATA = "gs://datachain-demo/chatbot-KiT"
MODEL = "anthropic/claude-haiku-4-5"

PROMPT = """Consider the dialogue between the 'user' and the 'bot'. The dialog is
successful if the 'bot' gathers the information and offers a plan (or says no such
plan exists), and a failure if the conversation ends early or the user requests
something the bot cannot do. Rate the dialogue and explain the rating in one
sentence."""


class Rating(BaseModel):
    status: str
    explanation: str


(
    dc.read_storage(DATA, type="text", anon=True)
    .filter(C("file.path").glob("*.txt"))
    .limit(5)
    .settings(llm=MODEL, cache=True)
    .map(rating=llm.complete("file", PROMPT, schema=Rating))
    .save("chatbot-ratings")
    .show()
)
