from pydantic import BaseModel


class Usage(BaseModel):
    """Token usage and retry count for a single model call.

    Emitted as a separate column when a `datachain.llm` operation is called with
    ``include_usage=True``; see the model-call functions.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0  # attempts beyond the first (0 = succeeded first try)
