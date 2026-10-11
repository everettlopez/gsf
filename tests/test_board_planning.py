"""The project board feeding back into planning: replans after a board visit,
and the agent's own progress showing up when they're at the board."""

import llm
import main
import project_board as pb
from conftest import fake_generate
from memory_retriever import MemoryRetriever
from memory_stream import MemoryStream


def _runtime(world, monkeypatch, prompts):
    def capture(prompt, options=None, format=None):
        if "Revise" in prompt:
            prompts.append(prompt)
            return ("1. work on T-1 'Set up the API' in the office from 9:05 am to 11:00 am\n"
                    "2. update T-1 on the project board from 11:00 am to 11:15 am\n"
                    "3. clock out at 5:00 pm")
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", capture)
    rt = main.make_runtime("adam", world)
    main.initial_planning(rt, world)
    return rt


def test_replans_around_tasks_after_leaving_the_board(
    fake_llm, isolated_data, world, monkeypatch
):
    prompts = []
    rt = _runtime(world, monkeypatch, prompts)
    pb.clear_workload_changed()
    pb.create_task(world, "adam", "Set up the API", area="backend", assign_to_self=True)

    rt.agent["interacting_with"] = pb.BOARD_OBJECT
    main.replan_after_board(rt, "9:00 am", world)
    assert prompts == []                              # still at the board: wait

    rt.agent["interacting_with"] = "computer"
    main.replan_after_board(rt, "9:00 am", world)

    [prompt] = prompts
    assert "T-1 'Set up the API' (todo)" in prompt
    assert "update it on the project board" in prompt
    assert "before lunch and before" in prompt
    assert any("T-1 on the project board" in c["activity"] for c in rt.broad_chunks)
    assert not pb.workload_changed("adam")

    main.replan_after_board(rt, "9:05 am", world)
    assert len(prompts) == 1                          # one visit, one replan


def test_no_replan_without_a_workload_change(fake_llm, isolated_data, world, monkeypatch):
    prompts = []
    rt = _runtime(world, monkeypatch, prompts)
    pb.clear_workload_changed()
    pb.create_task(world, "adam", "Write docs")       # created, not taken on
    main.replan_after_board(rt, "9:00 am", world)
    assert prompts == []


def test_task_guidance_is_empty_without_tasks(fake_llm, isolated_data, world, monkeypatch):
    rt = _runtime(world, monkeypatch, [])
    assert rt.planner.task_guidance(rt.agent, []) == ""


def test_progress_context_uses_work_memories_not_board_lines(fake_llm, isolated_data, world):
    adam = world["agents"]["adam"]
    stream = MemoryStream("adam")
    stream.add_observations([
        "Adam Beck is currently writing the API endpoint code.",
        "Adam Beck is currently testing the API functionality.",
        "Task T-1 on the project board, 'Set up the API' (backend, high priority), is todo.",
    ])
    retriever = MemoryRetriever("adam", stream)
    pb.create_task(world, "adam", "Set up the API", assign_to_self=True)

    context = pb._progress_context(adam, world, retriever)
    assert "writing the API endpoint code" in context
    assert "testing the API functionality" in context
    assert "project board" not in context

    prompt = pb.build_prompt(adam, world, "update my tasks", "summary", "", context)
    assert "What Adam Beck has been doing lately:" in prompt
    assert "approve_task" in prompt


def test_progress_context_is_empty_without_open_tasks(fake_llm, isolated_data, world):
    retriever = MemoryRetriever("adam", MemoryStream("adam"))
    assert pb._progress_context(world["agents"]["adam"], world, retriever) == ""
