"""
LocalExecutor: the tool-use loop on the executor model, against a real git
workspace (temp dir) and a scripted model in place of llm.chat. No API, no
Docker: commands go to a recording runner.
"""

import shutil

import pytest

import agent as ag
import llm
import project_board as pb
from conftest import board_api_and_pages
from executor import TOOLS, LocalExecutor, _invalid_input
from llm import Reply, ToolCall
from workspace import CommandResult, Workspace

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

PAGE = "<!doctype html><title>Home</title><h1>Welcome</h1>"


class Runner:
    """Records commands; run_tests results come from `outcomes` in order."""

    def __init__(self, *outcomes):
        self.outcomes, self.commands = list(outcomes), []

    def run(self, path, command, timeout=None):
        self.commands.append(command)
        return self.outcomes.pop(0) if self.outcomes else CommandResult(0, "ok")


def _turn(*calls, stop="tool_use", text=""):
    tool_calls = [ToolCall(f"toolu_{i}", name, args) for i, (name, args) in enumerate(calls)]
    return Reply(text=text, tool_calls=tool_calls, stop_reason=stop,
                 content=[{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input}
                          for c in tool_calls] or [{"type": "text", "text": text}])


@pytest.fixture
def script(monkeypatch):
    """Replace llm.chat with a script of replies; keeps every request."""
    state = {"replies": [], "requests": []}

    def chat(messages, system=None, tools=None, **kw):
        state["requests"].append({"messages": list(messages), "system": system, "tools": tools})
        reply = state["replies"].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm, "chat", chat)
    return state


@pytest.fixture
def eve(fake_llm, isolated_data, w, tmp_path):
    board_api_and_pages(w)                                      # T-2 is Eve's, in progress
    rt = ag.Agent.load("eve", w)
    rt.workspace = Workspace(rt.agent, tmp_path / "workspaces", runner=Runner())
    return rt, pb.get_task(w, "T-2")


def _results(request):
    return request["messages"][-1]["content"]


def test_a_session_writes_tests_and_reports_finished_work(eve, script):
    rt, task = eve
    rt.workspace.runner = Runner(CommandResult(0, "Ran 2 tests\nOK"))
    script["replies"] = [
        _turn(("list_files", {})),
        _turn(("write_file", {"path": "index.html", "contents": PAGE}),
              ("write_file", {"path": "tests/test_home.py", "contents": "import unittest\n"})),
        _turn(("run_tests", {"command": "python -m unittest discover tests"})),
        _turn(("report", {"summary": "The home page is built and its tests pass."})),
    ]
    result = LocalExecutor().run(rt, task, rt.workspace)

    assert result.summary == "The home page is built and its tests pass."
    assert result.tests_passed is True and result.finished
    assert result.files_changed == ["index.html", "tests/test_home.py"]
    assert rt.workspace.git("rev-parse", "--abbrev-ref", "HEAD") == "task/T-2"
    assert rt.workspace.git("log", "-1", "--format=%s").startswith("T-2: The home page is built")
    assert rt.workspace.git("ls-remote", "origin", "task/T-2")          # published for review
    assert "README.md" in _results(script["requests"][1])[0]["content"]  # list_files answered


def test_the_model_gets_the_task_its_notes_and_earlier_sessions(eve, script):
    rt, task = eve
    task["history"].append({"time": "10:00 am", "agent": "adam", "event": "note",
                            "note": "Use the /api/pages endpoint for the menu"})
    rt.work_log["T-2"] = [{"result": type("R", (), {"summary": "Built the hero section."})()}]
    script["replies"] = [_turn(("report", {"summary": "Started."}))]
    LocalExecutor().run(rt, task, rt.workspace)

    request = script["requests"][0]
    brief = request["messages"][0]["content"]
    assert brief.startswith("Task T-2: Design and implement the home page layout")
    assert "- adam at 10:00 am: Use the /api/pages endpoint for the menu" in brief
    assert "- Built the hero section." in brief
    assert "You are the hands of Eve Smith, a frontend engineer" in request["system"]
    assert {t["name"] for t in request["tools"]} == {t["name"] for t in TOOLS}


def test_tests_only_count_if_nothing_changed_after_them(eve, script):
    rt, task = eve
    script["replies"] = [
        _turn(("write_file", {"path": "index.html", "contents": PAGE})),
        _turn(("run_tests", {"command": "python -m unittest"})),
        _turn(("write_file", {"path": "index.html", "contents": PAGE + "<p>more</p>"})),
        _turn(("report", {"summary": "Done, I think."})),
    ]
    result = LocalExecutor().run(rt, task, rt.workspace)
    assert result.tests_passed is None and not result.finished
    assert not rt.workspace.git("ls-remote", "origin", "task/T-2")      # not published


def test_failing_tests_are_reported_as_failing(eve, script):
    rt, task = eve
    rt.workspace.runner = Runner(CommandResult(1, "FAILED (failures=1)"))
    script["replies"] = [
        _turn(("write_file", {"path": "index.html", "contents": PAGE})),
        _turn(("run_tests", {"command": "python -m unittest"})),
        _turn(("report", {"summary": "The nav test fails."})),
    ]
    result = LocalExecutor().run(rt, task, rt.workspace)
    assert result.tests_passed is False
    assert "exit code 1\nFAILED (failures=1)" in _results(script["requests"][2])[0]["content"]


def test_a_blocked_report_carries_the_reason(eve, script):
    rt, task = eve
    script["replies"] = [_turn(("report", {"summary": "The form is laid out.",
                                           "blocked_reason": "POST /contact isn't built yet (T-1)"}))]
    result = LocalExecutor().run(rt, task, rt.workspace)
    assert result.blocked and result.blocked_reason == "POST /contact isn't built yet (T-1)"


@pytest.mark.parametrize("call,error", [
    (("write_file", {"path": "../adam/api.py", "contents": "x"}), "outside the workspace"),
    (("write_file", {"path": "index.html"}), "Invalid input: missing contents"),
    (("read_file", {"path": "index.html", "mode": "r"}), "Invalid input: unknown mode"),
    (("read_file", {"path": 7}), "Invalid input: path must be text"),
    (("delete_everything", {}), "Invalid input: there is no tool 'delete_everything'"),
])
def test_bad_tool_calls_are_answered_as_errors_and_the_loop_goes_on(eve, script, call, error):
    rt, task = eve
    script["replies"] = [_turn(call), _turn(("report", {"summary": "Recovered."}))]
    result = LocalExecutor().run(rt, task, rt.workspace)
    [answer] = _results(script["requests"][1])
    assert answer["is_error"] and error in answer["content"]
    assert result.summary == "Recovered."
    assert not (rt.workspace.path.parent / "adam" / "api.py").exists()


def test_a_cut_off_turn_runs_none_of_its_tool_calls(eve, script):
    rt, task = eve
    script["replies"] = [
        _turn(("write_file", {"path": "index.html", "contents": "<h1>half"}), stop="max_tokens"),
        _turn(("report", {"summary": "Rewrote it in parts."})),
    ]
    LocalExecutor().run(rt, task, rt.workspace)
    [answer] = _results(script["requests"][1])
    assert answer["is_error"] and "Cut off" in answer["content"]
    assert not (rt.workspace.path / "index.html").exists()


def test_a_refusal_or_unreadable_input_ends_the_session_blocked(eve, script):
    rt, task = eve
    script["replies"] = [_turn(stop="refusal")]
    assert LocalExecutor().run(rt, task, rt.workspace).blocked_reason == \
        "the executor model declined to work on this task"

    script["replies"] = [llm.ToolInputError("bad json")] * 3
    result = LocalExecutor().run(rt, task, rt.workspace)
    assert "couldn't be read" in result.blocked_reason
    assert len(script["requests"]) == 1 + 3                            # retried, then gave up


def test_a_session_has_a_turn_limit(eve, script):
    rt, task = eve
    script["replies"] = [_turn(("list_files", {}))] * 3
    result = LocalExecutor(max_turns=3).run(rt, task, rt.workspace)
    assert result.summary == "The session ran out of turns before it was finished."
    assert not result.blocked and result.tests_passed is None


def test_a_plain_answer_ends_the_session(eve, script):
    rt, task = eve
    script["replies"] = [_turn(stop="end_turn", text="The page already exists; nothing to do.")]
    assert LocalExecutor().run(rt, task, rt.workspace).summary == \
        "The page already exists; nothing to do."


def test_the_tool_schemas_reject_what_they_should():
    assert _invalid_input("list_files", {}) == ""
    assert _invalid_input("report", {"summary": "x", "blocked_reason": "y"}) == ""
    assert _invalid_input("report", "summary") == "the input must be an object"
    assert all(t["input_schema"]["additionalProperties"] is False for t in TOOLS)


# --- skills ---------------------------------------------------------------------
@pytest.mark.parametrize("task,names", [
    ({"area": "frontend", "title": "Page layout"}, ["shared", "frontend"]),
    ({"area": "backend", "title": "Content file and build script"}, ["shared", "backend"]),
    ({"area": "backend", "title": "Ship the site", "description": "build dist/"},
     ["shared", "backend", "deploy"]),
    ({"area": "devops", "title": "Deploy the site"}, ["shared", "deploy"]),
    ({"area": "general", "title": "Write the README"}, ["shared"]),
])
def test_a_task_gets_the_shared_contract_its_lane_and_deploy_when_shipping(task, names):
    import skills

    assert skills.for_task(task) == names
    assert skills.for_review() == ["shared", "review"]


def test_skills_load_without_their_front_matter():
    import skills

    text = skills.load(["shared", "frontend", "no-such-skill"])
    assert not text.startswith("---") and "name: shared" not in text
    assert "{{placeholders}}" in text and "# Frontend" in text


def test_the_work_session_carries_the_tasks_skills(eve, script):
    rt, task = eve
    script["replies"] = [_turn(("report", {"summary": "Started."}))]
    LocalExecutor().run(rt, task, rt.workspace)
    system = script["requests"][0]["system"]
    assert "# Frontend: the layout and the styles" in system             # T-2 is frontend
    assert "The contract between template and content" in system
    assert "# Backend" not in system


# --- review sessions ------------------------------------------------------------------
@pytest.fixture
def adams_branch(eve, tmp_path, w):
    """Adam's T-1 published from his own workspace, ready for Eve to review."""
    rt, _ = eve
    adam = Workspace(w["agents"]["adam"], tmp_path / "workspaces", runner=Runner())
    adam.start_task("T-1")
    adam.write_file("build.py", "print('built')\n")
    adam.commit("T-1: the build")
    adam.publish("T-1")
    return rt, pb.get_task(w, "T-1")


def test_a_review_reads_and_runs_the_branch_and_returns_a_verdict(adams_branch, script):
    rt, task = adams_branch
    rt.workspace.runner = Runner(CommandResult(0, "Ran 3 tests\nOK"))
    script["replies"] = [
        _turn(("show_changes", {})),
        _turn(("run_command", {"command": "echo junk > scratch.txt"}),
              ("run_tests", {"command": "python -m unittest discover -s tests -v"})),
        _turn(("review_report", {"verdict": "changes", "summary": "The build runs but checks nothing.",
                                 "changes": "- build.py should fail on a missing key\n- add tests"})),
    ]
    result = LocalExecutor().review(rt, task, rt.workspace, owner_name="Adam Beck")

    assert (result.verdict, result.tests_passed) == ("changes", True)
    assert result.changes == ["build.py should fail on a missing key", "add tests"]
    diff = _results(script["requests"][1])[0]["content"]
    assert "build.py" in diff and "print('built')" in diff
    first = script["requests"][0]
    assert first["messages"][0]["content"].startswith("Review Adam Beck's task, now in review.")
    assert "# Review: a teammate's task" in first["system"]
    assert {t["name"] for t in first["tools"]} == {"list_files", "read_file", "run_command",
                                                   "run_tests", "show_changes", "review_report"}
    assert rt.workspace.git("status", "--porcelain") == ""             # nothing left behind


def test_a_review_needs_a_published_branch_and_a_real_verdict(eve, script, w):
    rt, _ = eve
    result = LocalExecutor().review(rt, pb.get_task(w, "T-1"), rt.workspace)
    assert result.verdict == "none" and script["requests"] == []        # no model call


def test_a_review_verdict_must_be_approve_or_changes(adams_branch, script):
    rt, task = adams_branch
    script["replies"] = [_turn(("review_report", {"verdict": "lgtm", "summary": "fine"})),
                         _turn(("review_report", {"verdict": "approve", "summary": "Fine."}))]
    result = LocalExecutor().review(rt, task, rt.workspace)
    [answer] = _results(script["requests"][1])
    assert answer["is_error"] and "verdict must be one of approve, changes" in answer["content"]
    assert result.verdict == "approve"
