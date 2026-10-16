"""
llm.py - the single place the project talks to language models.

Two roles, each routed to its own model:

    cognitive   the agents' minds: importance, reactions, planning, conversation,
                the board. Ollama (qwen3), through generate() / embed().
    executor    the agents' hands: writing and testing code for a board task.
                A strong hosted model, through chat() with tools. Claude by
                default (GA_EXECUTOR_PROVIDER=anthropic).

Every module calls generate() / embed() / chat() from here instead of a client
library, so model names live in one spot and tests monkeypatch these functions
instead of a whole client. Client libraries are imported lazily, so the tests
run without Ollama, the Anthropic SDK, or an API key.

Executor settings come from the environment:

    GA_EXECUTOR_PROVIDER   anthropic (default)
    GA_EXECUTOR_MODEL      claude-opus-5-5 (default)
    GA_EXECUTOR_EFFORT     low | medium | high (default) | xhigh | max
    ANTHROPIC_API_KEY      or any credential the Anthropic SDK resolves itself
"""

import os
from dataclasses import dataclass, field

MODEL = "qwen3"
EMBED_MODEL = "nomic-embed-text"

# Context window used by the longer prompts (planning, conversation, ...).
CTX = {"num_ctx": 8192}

COGNITIVE, EXECUTOR = "cognitive", "executor"

EXECUTOR_DEFAULTS = {"provider": "anthropic", "model": "claude-opus-5-5", "effort": "high"}
EXECUTOR_MAX_TOKENS = 64000     # streamed, so a long file write doesn't hit a timeout


class LLMError(RuntimeError):
    """The executor model can't be reached or configured."""


class ToolInputError(LLMError):
    """The model streamed a tool input that isn't valid JSON; retry the turn."""


def executor_config():
    """The executor's provider, model and effort, read from the environment
    each call so a run (or a test) can change them."""
    return {key: os.environ.get(f"GA_EXECUTOR_{key.upper()}") or default
            for key, default in EXECUTOR_DEFAULTS.items()}


# ---------------------------------------------------------------------------
# COGNITIVE: Ollama
# ---------------------------------------------------------------------------
def generate(prompt, options=None, format=None, role=COGNITIVE):
    """Run one completion and return the response text.

    The cognitive role runs on Ollama with thinking disabled; `options` and
    `format` are Ollama's. The executor role sends the prompt as one user turn."""
    if role == EXECUTOR:
        return chat([{"role": "user", "content": prompt}]).text
    if role != COGNITIVE:
        raise ValueError(f"unknown role {role!r}")

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


# ---------------------------------------------------------------------------
# EXECUTOR: a hosted model with tools
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Reply:
    """One executor turn, the same shape whatever the provider.

    `content` is the provider's own content, to append unchanged as the
    assistant turn (it can carry thinking blocks the next turn needs).
    `stop_reason` is end_turn, tool_use, max_tokens or refusal: a turn that
    stopped on max_tokens or refusal may hold a cut-off tool call, so its
    tool calls must not be run."""
    text: str
    tool_calls: list = field(default_factory=list)
    stop_reason: str = "end_turn"
    content: list = field(default_factory=list)
    model: str = ""


_clients = {}


def _anthropic_client():
    if "anthropic" not in _clients:
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("the executor needs the Anthropic SDK: pip install anthropic") from e
        _clients["anthropic"] = anthropic.Anthropic()
    return _clients["anthropic"]


def chat(messages, system=None, tools=None, role=EXECUTOR, max_tokens=EXECUTOR_MAX_TOKENS):
    """One executor turn: send the conversation (and tools), return a Reply.

    Tools are plain JSON-schema definitions ({name, description, input_schema}).
    The caller owns the loop: run the tool calls, append Reply.content as the
    assistant turn and the results as the next user turn, and call again."""
    if role != EXECUTOR:
        raise ValueError("chat() is for the executor role; the cognitive role uses generate()")
    config = executor_config()
    if config["provider"] != "anthropic":
        raise LLMError(f"unknown executor provider {config['provider']!r} (GA_EXECUTOR_PROVIDER)")
    return _anthropic_chat(config, messages, system, tools, max_tokens)


def _anthropic_chat(config, messages, system, tools, max_tokens):
    client = _anthropic_client()
    request = {
        "model": config["model"],
        "max_tokens": max_tokens,
        "messages": messages,
        "output_config": {"effort": config["effort"]},
        # A request a safety classifier declines is re-run on Anthropic's
        # recommended fallback model instead of coming back as a refusal.
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
    }
    if system:
        request["system"] = system
    if tools:
        # Streamed: let long tool inputs (whole files) arrive as they're
        # written. The caller validates each input before running the tool.
        request["tools"] = [{**t, "eager_input_streaming": True} for t in tools]

    import anthropic

    try:
        with client.beta.messages.stream(**request) as stream:
            message = stream.get_final_message()
    except anthropic.APIError as e:
        raise LLMError(f"executor call failed: {e}") from e
    except ValueError as e:
        # A streamed tool input that isn't JSON at all: there is no tool_use
        # block to answer, so the caller sends the same turn again.
        raise ToolInputError(f"a tool input couldn't be parsed: {e}") from e

    text = "".join(b.text for b in message.content if b.type == "text")
    calls = [ToolCall(b.id, b.name, b.input) for b in message.content if b.type == "tool_use"]
    return Reply(text=text, tool_calls=calls, stop_reason=message.stop_reason,
                 content=message.content, model=message.model)
