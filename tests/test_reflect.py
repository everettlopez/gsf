from memory_stream import MemoryStream
from memory_retriever import MemoryRetriever
from reflect import Reflector, parse_insights, parse_questions


def test_parse_questions_strips_markers_and_intro():
    text = "Here are the questions:\n1. What is Adam doing?\n2) **Who is Eve?**\n- Why?"
    assert parse_questions(text) == ["What is Adam doing?", "Who is Eve?", "Why?"]


def test_parse_insights_extracts_citations_and_normalizes_pronouns():
    text = (
        "Here are 2 insights:\n"
        "1. He is dedicated to clean code (because of 1, 5, 3)\n"
        "   * evidence line\n"
        "2. His work is steady (because of 9)\n"
    )
    result = parse_insights(text, valid_ids=[1, 3, 5], agent_name="Adam Beck")
    assert result[0] == ("Adam Beck is dedicated to clean code", [1, 5, 3])
    # 9 isn't a real memory id, so the insight falls back to all source memories
    assert result[1] == ("Adam Beck's work is steady", [1, 3, 5])


def _reflector(agent_id="adam"):
    agent = {"id": agent_id, "name": "Adam Beck"}
    stream = MemoryStream(agent_id)
    return Reflector(agent, stream, MemoryRetriever(agent_id, stream)), stream


def test_reflect_skips_when_there_is_little_new(fake_llm, isolated_data):
    reflector, stream = _reflector()
    stream.add_observations(["one", "two"])
    assert reflector.reflect() == []
    assert not any(m.memory_type == "reflection" for m in stream.memories)


def test_reflect_stores_cited_insights_and_resets_window(fake_llm, isolated_data):
    reflector, stream = _reflector()
    stream.add_observations([f"Adam did thing {i}" for i in range(8)])

    reflections = reflector.reflect()
    assert len(reflections) == 2
    assert all(r.memory_type == "reflection" and r.source_memory_ids for r in reflections)
    assert all(src <= 8 for r in reflections for src in r.source_memory_ids)

    # Nothing new happened since, so a second pass does nothing.
    assert reflector.reflect() == []



def test_insights_with_a_leading_citation_or_a_heading():
    from reflect import parse_insights

    text = ("insight 1 (because of 1, 2): Eve Smith is building the layout.\n"
            "Insight (because of 3): The team reviews each other's work.\n"
            "Collaborative Task Management\n"
            "Adam Beck owns the build (because of 2)")
    assert parse_insights(text, [1, 2, 3], "Adam Beck") == [
        ("Eve Smith is building the layout.", [1, 2]),
        ("The team reviews each other's work.", [3]),
        ("Adam Beck owns the build", [2]),
    ]


def test_insights_see_the_statements_in_time_order_with_their_times(fake_llm, isolated_data,
                                                                     monkeypatch):
    import clock
    import llm

    reflector, stream = _reflector()
    clock.set_time("8:40 am")
    later = stream.add_observations(["Task T-5 on the project board is review."])[0]
    clock.set_time("7:40 am")
    earlier = stream.add_observations(["Task T-5 on the project board is todo."])[0]
    prompts = []
    monkeypatch.setattr(llm, "generate", lambda prompt, **_: prompts.append(prompt) or "")
    reflector.generate_insights([later, earlier])
    assert (f"{earlier.id}. (7:40 am) Task T-5 on the project board is todo.\n"
            f"{later.id}. (8:40 am) Task T-5 on the project board is review.") in prompts[0]
    assert "A later statement about the same thing replaces an earlier one" in prompts[0]
