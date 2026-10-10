import clock
from memory import Memory
from memory_retriever import MemoryRetriever
from memory_stream import MemoryStream, parse_importance


def test_parse_importance_handles_messy_llm_output():
    assert parse_importance("7") == 7
    assert parse_importance("Rating: 8/10") == 8
    assert parse_importance("42") == 10        # clamped
    assert parse_importance("0") == 1          # clamped
    assert parse_importance("no idea") == 5    # default
    assert parse_importance(None) == 5


def test_stream_adds_and_persists(fake_llm, isolated_data):
    stream = MemoryStream("adam")
    stream.add_observations(["Adam is at the desk.", "Eve walked in."])
    stream.add_plan("7:00 am: work")
    stream.flush()

    reloaded = MemoryStream("adam")
    assert [m.id for m in reloaded.memories] == [1, 2, 3]
    assert reloaded.memories[2].memory_type == "plan"


def test_stream_survives_bad_importance_reply(fake_llm, isolated_data, monkeypatch):
    import llm

    monkeypatch.setattr(llm, "generate", lambda *a, **k: "I cannot rate this")
    stream = MemoryStream("adam")
    memory = stream.add_observation("Adam drank coffee.")
    assert memory.importance == 5


def test_unreflected_only_returns_observations_since_last_reflection(fake_llm, isolated_data):
    stream = MemoryStream("adam")
    stream.add_observations([f"obs {i}" for i in range(4)])
    stream.add_plan("a plan")
    stream.add_reflection("an insight", [1, 2])
    stream.add_observations(["new 1", "new 2"])

    fresh = stream.unreflected()
    assert [m.description for m in fresh] == ["new 1", "new 2"]   # no plans, no old obs
    assert stream.last_reflection_id() == 6


def test_unreflected_respects_window(fake_llm, isolated_data):
    stream = MemoryStream("adam")
    stream.add_observations([f"obs {i}" for i in range(10)])
    assert [m.description for m in stream.unreflected(limit=3)] == ["obs 7", "obs 8", "obs 9"]


def _stream_with(isolated_data, memories):
    stream = MemoryStream("adam")
    stream.memories = memories
    return stream


def test_retriever_ranks_relevant_memory_first(isolated_data):
    clock.set_time("9:00 am")
    now = clock.now().isoformat()
    memories = [
        Memory(id=1, description="about python", importance=5, embedding=[1.0, 0.0], created_at=now, last_accessed_at=now),
        Memory(id=2, description="about cooking", importance=5, embedding=[0.0, 1.0], created_at=now, last_accessed_at=now),
    ]
    retriever = MemoryRetriever("adam", _stream_with(isolated_data, memories))
    retriever._query_cache["python?"] = __import__("numpy").array([1.0, 0.1])

    results = retriever.retrieve("python?", n=2)
    assert [r["memory"].id for r in results] == [1, 2]


def test_retrieve_touch_flag_controls_recency_refresh(isolated_data):
    clock.set_time("7:00 am")
    old = clock.now().isoformat()
    memory = Memory(id=1, description="m", importance=5, embedding=[1.0, 0.0], created_at=old, last_accessed_at=old)
    retriever = MemoryRetriever("adam", _stream_with(isolated_data, [memory]))
    retriever._query_cache["q"] = __import__("numpy").array([1.0, 0.0])

    clock.set_time("10:00 am")
    retriever.retrieve("q", touch=False)
    assert memory.last_accessed_at == old

    retriever.retrieve("q")
    assert memory.last_accessed_at == clock.now().isoformat()
