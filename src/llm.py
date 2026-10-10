"""
llm.py - the single place the project talks to Ollama.

Every module calls generate() / embed() from here instead of importing ollama
directly, so the model names live in one spot and tests can monkeypatch two
functions instead of the whole client. `ollama` is imported lazily so unit
tests run on machines without it installed.
"""

MODEL = "qwen3"
EMBED_MODEL = "nomic-embed-text"

# Context window used by the longer prompts (planning, conversation, ...).
CTX = {"num_ctx": 8192}


def generate(prompt, options=None, format=None):
    """Run one completion (thinking disabled) and return the response text."""
    import ollama

    kwargs = {"model": MODEL, "prompt": prompt, "think": False}
    if options:
        kwargs["options"] = options
    if format is not None:
        kwargs["format"] = format
    return ollama.generate(**kwargs)["response"]


def embed(inputs):
    """Embed a string -> one vector, or a list of strings -> a list of vectors."""
    import ollama

    vectors = ollama.embed(model=EMBED_MODEL, input=inputs)["embeddings"]
    return vectors[0] if isinstance(inputs, str) else vectors
