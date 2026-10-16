"""
Personas live in data/agents/<id>/persona.json. Their work and communication
styles have to reach the prompts that decide what an agent does (react or not,
what to do at the board, what to say), not only the planning profile. The
profile and the seeded memories both say a task belongs to its assignee.
"""

import pytest

import agent as ag
import llm
import persona
import project_board as pb
from conftest import fake_generate, seed_module
from conversation import run_conversation


@pytest.fixture
def prompts(monkeypatch):
    seen = []

    def gen(prompt, options=None, format=None):
        seen.append(prompt)
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    return seen


@pytest.mark.parametrize("agent_id", ["adam", "eve"])
def test_persona_has_the_papers_fields_and_styles(agent_id):
    p = persona.load(agent_id)
    for key in ("name", "age", "innate_traits", "occupation", "lifestyle", "previous_day",
                "work_style", "communication_style"):
        assert p[key], key
    profile = persona.profile(agent_id)
    assert profile.startswith(f"Name: {p['name']} (age: {p['age']})")
    assert p["work_style"][0] in profile and p["communication_style"][0] in profile
    assert profile.endswith("7) clocked out and finished work at 5:00 pm.")


def test_adam_and_eve_decide_in_their_own_ways():
    adam, eve = persona.style("adam"), persona.style("eve")
    assert "How Adam works:" in adam and "How Adam communicates:" in adam
    assert "How Eve works:" in eve and "How Eve communicates:" in eve
    assert "leaves a note on the project board" in eve and "walk over and talk" in adam
    assert persona.style("nobody") == ""


@pytest.mark.parametrize("agent_id,name", [("adam", "Adam"), ("eve", "Eve")])
def test_react_prompt_carries_the_agents_style(fake_llm, isolated_data, w, prompts, agent_id, name):
    rt = ag.Agent.load(agent_id, w)
    chunk = {"activity": "7:30 am: work on the website", "location": rt.agent["current_location"]}
    rt.planner.continue_plan_or_react(rt.agent, "working", chunk, ["Someone walked in."], w)
    react = [p for p in prompts if "DECISION:" in p][-1]
    assert f"How {name} works:" in react and f"How {name} communicates:" in react
    other = "Eve" if name == "Adam" else "Adam"
    assert f"How {other} works:" not in react


def test_board_prompt_carries_the_agents_style(w):
    prompt = pb.build_prompt(w["agents"]["eve"], w, "check the board", "summary", "")
    assert "How Eve works:" in prompt and "How Eve communicates:" in prompt


def test_each_speaker_talks_in_their_own_style(fake_llm, isolated_data, w, monkeypatch):
    lines = iter(["How is the backend going?", "Slowly, I'm still on the API.", "END"])
    spoken = []

    def gen(prompt, options=None, format=None):
        if "is talking with" in prompt:
            spoken.append(prompt)
            return next(lines)
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    eve, adam = ag.Agent.load("eve", w), ag.Agent.load("adam", w)
    run_conversation(eve, adam, "7:20 am", "ask Adam how the backend is going", w)
    assert "How Eve communicates:" in spoken[0] and "How Adam" not in spoken[0]
    assert "How Adam communicates:" in spoken[1] and "How Eve" not in spoken[1]


@pytest.mark.parametrize("agent_id", ["adam", "eve"])
def test_profiles_say_tasks_belong_to_their_assignee(agent_id):
    profile = " ".join(persona.profile(agent_id).split())
    assert "update their status" not in profile           # "anyone can ... update their status"
    assert "A task belongs to the person it is assigned to" in profile
    assert "offers to work on it together with" in profile


def test_seeds_say_tasks_belong_to_their_assignee():
    seeds = seed_module()
    for memories in (seeds.adam_seed_memories, seeds.eve_seed_memories):
        text = " ".join(memories)
        assert "belongs to the person it is assigned to" in text
        assert "take them on, and update their status" not in text
