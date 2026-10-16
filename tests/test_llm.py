"""
llm.py: the cognitive role stays on Ollama (qwen3); the executor role goes to
a hosted model with tools. Stub client modules stand in for ollama and
anthropic, so these run without either installed and without a key.
"""

import sys
import types
from types import SimpleNamespace

import pytest

import llm


@pytest.fixture
def fake_ollama(monkeypatch):
    calls = []
    stub = types.ModuleType("ollama")
    stub.generate = lambda **kw: calls.append(("generate", kw)) or {"response": "7"}
    stub.embed = lambda **kw: calls.append(("embed", kw)) or {
        "embeddings": [[0.1, 0.2]] * (1 if isinstance(kw["input"], str) else len(kw["input"]))}
    monkeypatch.setitem(sys.modules, "ollama", stub)
    return calls


class _APIError(Exception):
    pass


@pytest.fixture
def fake_anthropic(monkeypatch):
    """A stub SDK whose beta stream returns `state["message"]` (or raises
    `state["raise"]`) and records every request."""
    state = {"requests": [], "raise": None, "message": SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Done: index.html written.")],
        stop_reason="end_turn", model="claude-opus-5-5")}

    class Stream:
        def __enter__(self):
            if state["raise"]:
                raise state["raise"]
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return state["message"]

    def stream(**request):
        state["requests"].append(request)
        return Stream()

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    stub = types.ModuleType("anthropic")
    stub.APIError = _APIError
    stub.Anthropic = lambda: client
    monkeypatch.setitem(sys.modules, "anthropic", stub)
    monkeypatch.setattr(llm, "_clients", {})
    for key in ("PROVIDER", "MODEL", "EFFORT"):
        monkeypatch.delenv(f"GA_EXECUTOR_{key}", raising=False)
    return state


WRITE_FILE = {"name": "write_file", "description": "Write a file in the workspace",
              "input_schema": {"type": "object", "properties": {"path": {"type": "string"},
                                                                "contents": {"type": "string"}},
                               "required": ["path", "contents"]}}


# --- cognitive -----------------------------------------------------------------
def test_the_cognitive_role_runs_on_ollama_without_thinking(fake_ollama):
    assert llm.generate("Rate this", options={"num_predict": 5}, format="json") == "7"
    [(kind, request)] = fake_ollama
    assert kind == "generate"
    assert request == {"model": "qwen3", "prompt": "Rate this", "think": False,
                       "options": {"num_predict": 5}, "format": "json"}


def test_embeddings_stay_on_ollama(fake_ollama):
    assert llm.embed("one") == [0.1, 0.2]
    assert llm.embed(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert {kw["model"] for _, kw in fake_ollama} == {"nomic-embed-text"}


def test_an_unknown_role_is_refused(fake_ollama):
    with pytest.raises(ValueError, match="unknown role"):
        llm.generate("hi", role="hands")
    with pytest.raises(ValueError, match="executor role"):
        llm.chat([{"role": "user", "content": "hi"}], role=llm.COGNITIVE)
    assert fake_ollama == []


# --- executor ------------------------------------------------------------------
def test_the_executor_role_goes_to_claude_with_tools(fake_anthropic):
    messages = [{"role": "user", "content": "Build the page for T-3."}]
    reply = llm.chat(messages, system="You are Eve's hands.", tools=[WRITE_FILE])

    [request] = fake_anthropic["requests"]
    assert request["model"] == "claude-opus-5-5"
    assert request["output_config"] == {"effort": "high"}
    assert request["system"] == "You are Eve's hands." and request["messages"] is messages
    assert request["fallbacks"] == "default"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["tools"] == [{**WRITE_FILE, "eager_input_streaming": True}]
    assert "eager_input_streaming" not in WRITE_FILE              # the caller's tool is untouched
    assert (reply.text, reply.tool_calls, reply.stop_reason) == ("Done: index.html written.", [], "end_turn")


def test_tool_calls_come_back_with_the_content_to_append(fake_anthropic):
    content = [SimpleNamespace(type="text", text="Writing it now."),
               SimpleNamespace(type="tool_use", id="toolu_1", name="write_file",
                               input={"path": "index.html", "contents": "<h1>Hi</h1>"})]
    fake_anthropic["message"] = SimpleNamespace(content=content, stop_reason="tool_use",
                                                model="claude-opus-5-5")
    reply = llm.chat([{"role": "user", "content": "go"}], tools=[WRITE_FILE])
    assert reply.tool_calls == [llm.ToolCall("toolu_1", "write_file",
                                             {"path": "index.html", "contents": "<h1>Hi</h1>"})]
    assert reply.stop_reason == "tool_use" and reply.content is content


def test_the_environment_picks_the_executor_model_and_effort(fake_anthropic, monkeypatch):
    monkeypatch.setenv("GA_EXECUTOR_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("GA_EXECUTOR_EFFORT", "xhigh")
    llm.chat([{"role": "user", "content": "go"}])
    [request] = fake_anthropic["requests"]
    assert request["model"] == "claude-sonnet-5-5"
    assert request["output_config"] == {"effort": "xhigh"}
    assert "tools" not in request and "system" not in request


def test_an_executor_prompt_through_generate_is_one_user_turn(fake_anthropic):
    assert llm.generate("Summarize the diff", role=llm.EXECUTOR) == "Done: index.html written."
    assert fake_anthropic["requests"][0]["messages"] == [{"role": "user", "content": "Summarize the diff"}]


def test_executor_failures_are_named(fake_anthropic, monkeypatch):
    fake_anthropic["raise"] = _APIError("rate limited")
    with pytest.raises(llm.LLMError, match="executor call failed: rate limited"):
        llm.chat([{"role": "user", "content": "go"}])

    fake_anthropic["raise"] = ValueError("Unterminated string")
    with pytest.raises(llm.ToolInputError):
        llm.chat([{"role": "user", "content": "go"}], tools=[WRITE_FILE])

    monkeypatch.setenv("GA_EXECUTOR_PROVIDER", "gemini")
    with pytest.raises(llm.LLMError, match="unknown executor provider 'gemini'"):
        llm.chat([{"role": "user", "content": "go"}])


def test_a_missing_sdk_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)              # import fails
    monkeypatch.setattr(llm, "_clients", {})
    monkeypatch.delenv("GA_EXECUTOR_PROVIDER", raising=False)
    with pytest.raises(llm.LLMError, match="pip install anthropic"):
        llm.chat([{"role": "user", "content": "go"}])
