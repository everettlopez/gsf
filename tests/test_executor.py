"""
executor.py and the hands in agent.py: when an agent's step is work on a task
it owns and has in progress, the executor does the work (once per plan block),
and the outcome comes back as memories. A finished task becomes a promise to
move it to review; a blocked one sends the agent to the board to say so. The
executor never works on a teammate's task, at the board, in a talk or on a break.
"""

import pytest

import agent as ag
import project_board as pb
from conftest import board_api_and_pages
from executor import ExecResult, FakeExecutor

PROGRESS = ExecResult("Built the hero section and the nav bar.", ["index.html", "styles.css"], None)
DONE = ExecResult("The home page is complete.", ["index.html"], tests_passed=True)
STUCK = ExecResult("Couldn't wire the contact form.", [], tests_passed=False,
                   blocked_reason="the /contact endpoint doesn't exist yet")


def _eve_at_her_desk(w, results, activity="9:00 am: Build the hero section"):
    """Eve, with T-2 in progress, in office_2, on a work step of a 9-11 block."""
    board_api_and_pages(w)
    executor = FakeExecutor(results)
    rt = ag.Agent.load("eve", w, executor=executor, workspace="workspaces/eve")
    rt.agent["current_location"] = "office_2"
    rt.broad_chunks = [{"activity": "work on T-2 'Design and implement the home page layout' "
                                    "from 9:00 am to 11:00 am", "location": "office_2",
                        "start_time": "9:00 am", "duration": 120, "target_agent": None}]
    rt.chunk = {"activity": activity, "location": "office_2", "start_time": "9:00 am",
                "duration": 15, "target_agent": None}
    return rt, executor


def _memories(rt):
    return [m.description for m in rt.stream.memories]


# --- when the hands work ---------------------------------------------------------
def test_work_on_an_own_task_in_progress_goes_to_the_executor(fake_llm, isolated_data, w):
    rt, executor = _eve_at_her_desk(w, [PROGRESS])
    assert ag.work_on_task(rt, "9:00 am", w) is PROGRESS
    assert executor.calls == [("eve", "T-2", "workspaces/eve")]
    memories = _memories(rt)
    assert ("Eve Smith worked on T-2 'Design and implement the home page layout' at 9:00 am: "
            "Built the hero section and the nav bar.") in memories
    assert "Eve Smith changed index.html, styles.css for T-2." in memories
    assert not any("tests for T-2" in m for m in memories)            # none were run


def test_one_session_per_plan_block(fake_llm, isolated_data, w):
    rt, executor = _eve_at_her_desk(w, [PROGRESS])
    ag.work_on_task(rt, "9:00 am", w)
    rt.chunk = dict(rt.chunk, activity="9:15 am: Style the hero section", start_time="9:15 am")
    assert ag.work_on_task(rt, "9:15 am", w) is None                  # same 9-11 block
    rt.broad_chunks.append({"activity": "work on T-2 from 11:00 am to 12:00 pm", "location": "office_2",
                            "start_time": "11:00 am", "duration": 60, "target_agent": None})
    rt.chunk = dict(rt.chunk, activity="11:00 am: Add the footer", start_time="11:00 am")
    ag.work_on_task(rt, "11:00 am", w)
    assert len(executor.calls) == 2


@pytest.mark.parametrize("change", [
    {"activity": "9:00 am: Review T-1's API responses"},                      # Adam's task
    {"activity": "9:00 am: take a coffee break"},
    {"activity": "9:00 am: move T-2 to review on the project board", "board_work": True},
    {"activity": "9:00 am: pair with Adam Beck on the API", "target_agent": "adam"},
    {"location": "conference_room"},                                           # not at her desk
])
def test_the_hands_stay_in_their_lane(fake_llm, isolated_data, w, change):
    rt, executor = _eve_at_her_desk(w, [PROGRESS])
    if change.get("location"):
        rt.agent["current_location"] = change["location"]
    rt.chunk = {**rt.chunk, **{k: v for k, v in change.items() if k != "location"}}
    assert ag.work_on_task(rt, "9:00 am", w) is None
    assert executor.calls == []


def test_only_a_task_in_progress_is_worked_on(fake_llm, isolated_data, w):
    rt, executor = _eve_at_her_desk(w, [PROGRESS], activity="9:00 am: work on T-3")
    assert pb.get_task(w, "T-3")["status"] == "todo"                   # not started at the board
    assert ag.work_on_task(rt, "9:00 am", w) is None
    rt.executor = None
    rt.chunk["activity"] = "9:00 am: work on T-2"
    assert ag.work_on_task(rt, "9:00 am", w) is None                   # no hands: the paper's agents
    assert executor.calls == []


def test_acting_on_a_work_step_uses_the_hands_but_talking_does_not(fake_llm, isolated_data, w):
    rt, executor = _eve_at_her_desk(w, [PROGRESS])
    rt.decision = "1"
    rt.act(20, "9:00 am", w)
    assert len(executor.calls) == 1
    rt.decision = "3"                                                  # in a conversation
    rt.chunk = dict(rt.chunk, start_time="11:00 am")
    rt.act(21, "11:00 am", w)
    assert len(executor.calls) == 1


# --- what comes of it ---------------------------------------------------------------
def test_a_finished_task_becomes_a_promise_to_move_it_to_review(fake_llm, isolated_data, w):
    rt, executor = _eve_at_her_desk(w, [DONE])
    ag.work_on_task(rt, "9:00 am", w)
    assert "The tests for T-2 pass." in _memories(rt)
    assert pb.commitments_for("eve") == [
        "move T-2 to review on the project board, noting for the reviewer: The home page is "
        "complete. Changed: index.html. Tests pass on branch task/T-2."]
    assert pb.is_board_commitment(pb.commitments_for("eve")[0])

    rt.broad_chunks[0]["start_time"] = "11:00 am"                      # a later block
    assert ag.work_on_task(rt, "11:00 am", w) is None                  # nothing new to do
    pb.add_note(w, "adam", "T-2", "The hero image is missing alt text")
    ag.work_on_task(rt, "11:00 am", w)                                 # the board changed
    assert len(executor.calls) == 2


def test_a_blocked_task_sends_the_agent_to_the_board_with_the_reason(fake_llm, isolated_data, w):
    rt, _ = _eve_at_her_desk(w, [STUCK])
    ag.work_on_task(rt, "9:00 am", w)
    assert "Eve Smith is stuck on T-2: the /contact endpoint doesn't exist yet." in _memories(rt)
    assert "The tests for T-2 fail." in _memories(rt)
    promise = "mark T-2 blocked on the project board: the /contact endpoint doesn't exist yet"
    assert pb.commitments_for("eve") == [promise]

    visit = ag.get_chunk_lazy(rt, "9:05 am", w)                         # the very next step
    assert visit["board_work"] and visit["location"] == "conference_room"
    assert visit["activity"] == f"9:05 am: {promise}"
    prompt = pb.build_prompt(rt.agent, w, visit["activity"], "summary", "",
                             commitments=pb.commitments_for("eve"))
    assert f"- {promise}" in prompt                                     # the board visit knows why


def test_a_blocked_report_goes_to_the_board_even_mid_block_then_the_block_resumes(
        fake_llm, isolated_data, w):
    rt, _ = _eve_at_her_desk(w, [STUCK])
    block = {"activity": "9:00 am: keep working on T-2", "location": "office_2", "pinned": True,
             "start_time": "9:00 am", "duration": 75, "target_agent": None}
    rt.override = dict(block)                       # a long step under way (after a talk, say)
    ag.work_on_task(rt, "9:00 am", w)
    visit = ag.get_chunk_lazy(rt, "9:05 am", w)
    assert visit["board_work"] and visit["location"] == "conference_room"
    assert ag.get_chunk_lazy(rt, "9:15 am", w) == block                 # back to work after it


# --- the executor interface ----------------------------------------------------------
def test_the_fake_executor_plays_its_script_and_repeats_the_last_result():
    class A:
        agent = {"id": "eve"}

    fake = FakeExecutor([PROGRESS, DONE])
    task = {"id": "T-2"}
    assert [fake.run(A, task, None) for _ in range(3)] == [PROGRESS, DONE, DONE]
    assert FakeExecutor(lambda a, t, ws: STUCK).run(A, task, None) is STUCK


def test_a_result_is_finished_only_when_tests_pass_and_nothing_blocks_it():
    assert DONE.finished and not DONE.blocked
    assert not PROGRESS.finished                                        # no tests run
    assert STUCK.blocked and not STUCK.finished
    assert not ExecResult("x", tests_passed=True, blocked_reason="no key").finished


def test_many_changed_files_are_summarized(fake_llm, isolated_data, w):
    files = [f"page{i}.html" for i in range(9)]
    rt, _ = _eve_at_her_desk(w, [ExecResult("Added pages.", files)])
    ag.work_on_task(rt, "9:00 am", w)
    assert ("Eve Smith changed page0.html, page1.html, page2.html, page3.html, page4.html, "
            "page5.html and 3 more for T-2.") in _memories(rt)


# --- choosing the hands, and shipping approved work ------------------------------
def test_the_environment_chooses_the_hands(monkeypatch, tmp_path):
    import executor
    import workspace

    monkeypatch.delenv("GA_EXECUTOR", raising=False)
    assert executor.hands_for({"id": "eve"}) == (None, None)            # the paper's agents
    monkeypatch.setenv("GA_EXECUTOR", "fake")
    hands, place = executor.hands_for({"id": "eve"})
    assert isinstance(hands.inner, FakeExecutor) and place is None
    monkeypatch.setenv("GA_EXECUTOR", "local")
    monkeypatch.setenv("GA_SANDBOX", "local")
    monkeypatch.setattr(workspace, "WORKSPACES", tmp_path)
    hands, place = executor.hands_for({"id": "eve", "name": "Eve Smith"})
    assert isinstance(hands.inner, executor.LocalExecutor) and place.path == tmp_path / "eve"
    monkeypatch.setenv("GA_EXECUTOR", "robot")
    with pytest.raises(ValueError, match="unknown GA_EXECUTOR"):
        executor.hands_for({"id": "eve"})


def test_a_done_task_is_merged_into_main_once(fake_llm, isolated_data, w, tmp_path):
    from workspace import Workspace

    rt, _ = _eve_at_her_desk(w, [DONE])
    rt.workspace = Workspace(rt.agent, tmp_path / "workspaces", runner=object())
    rt.workspace.start_task("T-2")
    rt.workspace.write_file("index.html", "<h1>Welcome</h1>")
    rt.workspace.commit("T-2: home page")
    rt.workspace.publish("T-2")
    rt.work_log["T-2"] = [{"block": "9:00 am", "result": DONE, "history": 0}]

    ag.ship_approved_work(rt, w)                                         # still in progress
    assert rt.shipped == set()

    pb.get_task(w, "T-2")["status"] = "done"
    ag.ship_approved_work(rt, w)
    ag.ship_approved_work(rt, w)
    assert rt.shipped == {"T-2"}
    assert "index.html" in rt.workspace.git("ls-tree", "--name-only", "origin/main").split()
    merged = [m for m in _memories(rt) if "merged T-2" in m]
    assert merged == ["Eve Smith merged T-2 'Design and implement the home page layout' "
                      "into the website's main branch."]


def test_a_failed_merge_is_remembered(fake_llm, isolated_data, w, tmp_path):
    from workspace import Workspace

    rt, _ = _eve_at_her_desk(w, [DONE])
    rt.workspace = Workspace(rt.agent, tmp_path / "workspaces", runner=object())
    rt.work_log["T-2"] = [{"block": "9:00 am", "result": DONE, "history": 0}]
    pb.get_task(w, "T-2")["status"] = "done"                             # but never published
    ag.ship_approved_work(rt, w)
    assert any(m.startswith("Eve Smith couldn't merge T-2") for m in _memories(rt))


# --- reviewing a teammate's task ----------------------------------------------------
def _adams_task_in_review(w):
    """T-1 (Adam's) sent to review once."""
    pb.update_status(w, "adam", "T-1", "review", note="the content file and build are ready")
    return pb.get_task(w, "T-1")


def test_a_teammates_task_in_review_is_reviewed_before_own_work(fake_llm, isolated_data, w):
    from executor import ReviewResult

    rt, executor = _eve_at_her_desk(w, [PROGRESS])
    _adams_task_in_review(w)
    result = ag.work_on_task(rt, "9:00 am", w)
    assert isinstance(result, ReviewResult) and executor.review_calls == [("eve", "T-1")]
    assert executor.calls == []                                         # her own work waits
    memories = _memories(rt)
    assert any(m.startswith("Eve Smith reviewed Adam Beck's T-1") for m in memories)
    assert "Eve Smith thinks T-1 is ready to approve." in memories
    assert pb.commitments_for("eve") == [
        "approve T-1 on the project board: The branch does what the task says"]
    assert pb.is_board_commitment(pb.commitments_for("eve")[0])

    rt.broad_chunks[0]["start_time"] = "11:00 am"                       # next block: her own work
    ag.work_on_task(rt, "11:00 am", w)
    assert executor.review_calls == [("eve", "T-1")] and len(executor.calls) == 1


def test_changes_requested_become_a_note_promise_and_a_new_round_is_reviewed_again(
        fake_llm, isolated_data, w):
    from executor import ReviewResult

    rt, executor = _eve_at_her_desk(w, [PROGRESS])
    executor.reviews = [ReviewResult("changes", "The build works but misses a check.",
                                     ["build.py doesn't escape the tagline",
                                      "no test for a missing key"], tests_passed=True)]
    task = _adams_task_in_review(w)
    ag.work_on_task(rt, "9:00 am", w)
    promise = pb.commitments_for("eve")[0]
    assert promise == ("leave a note on T-1 on the project board asking Adam Beck for changes: "
                       "build.py doesn't escape the tagline; no test for a missing key")
    assert pb.is_board_commitment(promise)

    pb.add_note(w, "eve", "T-1", "build.py doesn't escape the tagline")  # at the board
    pb.update_status(w, "adam", "T-1", "in_progress")                    # Adam reopens it
    pb.update_status(w, "adam", "T-1", "review", note="fixed")            # second round
    assert task["status"] == "review"
    rt.broad_chunks[0]["start_time"] = "1:00 pm"
    ag.work_on_task(rt, "1:00 pm", w)
    assert len(executor.review_calls) == 2


def test_no_review_away_from_the_desk_or_of_an_approved_task(fake_llm, isolated_data, w):
    rt, executor = _eve_at_her_desk(w, [PROGRESS])
    task = _adams_task_in_review(w)
    rt.chunk = dict(rt.chunk, board_work=True, activity="9:00 am: approve T-1 on the project board")
    assert ag.review_at_hand(rt, w) is None
    rt.chunk = dict(rt.chunk, board_work=False, activity="9:00 am: Build the hero section")
    pb.approve_task(w, "eve", "T-1")
    assert ag.review_at_hand(rt, w) is None                              # she already approved it
    task["approved_by"].clear()
    assert ag.review_at_hand(rt, w) is task


def test_an_unpublished_branch_is_retried_later(fake_llm, isolated_data, w):
    from executor import ReviewResult

    rt, executor = _eve_at_her_desk(w, [PROGRESS])
    executor.reviews = [ReviewResult("none", "T-1 has no published branch to review yet.")]
    _adams_task_in_review(w)
    ag.work_on_task(rt, "9:00 am", w)
    assert "Eve Smith couldn't review Adam Beck's T-1" in _memories(rt)[-1]
    assert pb.commitments_for("eve") == [] and ag.review_at_hand(rt, w) is not None


# --- the run's session budget -------------------------------------------------------
def test_the_agents_share_one_session_budget(fake_llm, isolated_data, w):
    from executor import Budgeted, SessionBudget

    budget = SessionBudget(1)
    rt, fake = _eve_at_her_desk(w, [PROGRESS])
    rt.executor = Budgeted(fake, budget)
    adam = ag.Agent.load("adam", w, executor=Budgeted(FakeExecutor(), budget))
    assert ag.work_on_task(rt, "9:00 am", w) is PROGRESS
    assert not adam.executor.available()                                # Eve spent it
    rt.broad_chunks[0]["start_time"] = "11:00 am"
    assert ag.work_on_task(rt, "11:00 am", w) is None and len(fake.calls) == 1


def test_the_budget_comes_from_the_environment(monkeypatch):
    import executor

    monkeypatch.delenv("GA_EXECUTOR_MAX_SESSIONS", raising=False)
    monkeypatch.setenv("GA_EXECUTOR", "fake")
    assert executor.run_budget().left is None                           # fake: no limit
    monkeypatch.setenv("GA_EXECUTOR", "local")
    assert executor.run_budget().left == executor.DEFAULT_MAX_SESSIONS  # real hands: capped
    monkeypatch.setenv("GA_EXECUTOR_MAX_SESSIONS", "5")
    assert executor.run_budget().left == 5


def test_a_failing_executor_stops_the_hands_and_the_agent_carries_on(fake_llm, isolated_data, w):
    import llm

    def out_of_credit(agent, task, workspace):
        raise llm.LLMError("executor call failed: your credit balance is too low")

    rt, _ = _eve_at_her_desk(w, out_of_credit)
    before = len(rt.stream.memories)
    assert ag.work_on_task(rt, "9:00 am", w) is None
    assert rt.executor is None                                          # no more sessions this run
    assert len(rt.stream.memories) == before and rt.work_log.get("T-2") == []
    rt.decision = "1"
    rt.act(21, "9:05 am", w)                                            # the step still runs


def test_the_budget_adds_up_what_sessions_cost(fake_llm, isolated_data, w):
    from executor import Budgeted, SessionBudget

    budget = SessionBudget()
    rt, fake = _eve_at_her_desk(w, [ExecResult("Built it.", ["index.html"], True, cost=0.42)])
    rt.executor = Budgeted(fake, budget)
    ag.work_on_task(rt, "9:00 am", w)
    assert round(budget.spent, 2) == 0.42


def test_a_long_report_becomes_a_short_memory(fake_llm, isolated_data, w):
    report = "Built the template and the styles. 23 tests pass. More.\n\n**Details:** lots"
    rt, _ = _eve_at_her_desk(w, [ExecResult(report, ["index.html"], True)])
    ag.work_on_task(rt, "9:00 am", w)
    assert ("Eve Smith worked on T-2 'Design and implement the home page layout' at 9:00 am: "
            "Built the template and the styles. 23 tests pass.") in _memories(rt)
    assert not any("Details" in m for m in _memories(rt))



# --- the loop: finishing, asking and reviewing lead somewhere ---------------------
def test_finished_work_goes_straight_to_the_board_for_review(fake_llm, isolated_data, w):
    rt, _ = _eve_at_her_desk(w, [DONE])
    ag.work_on_task(rt, "9:00 am", w)
    visit = ag.get_chunk_lazy(rt, "9:05 am", w)
    assert visit["board_work"] and visit["activity"].startswith("9:05 am: move T-2 to review")
    assert pb.is_tested("eve", "T-2")


def test_a_question_from_the_hands_becomes_a_talk_with_the_teammate(fake_llm, isolated_data, w):
    asks = ExecResult("Laid out the contact section.", ["site/template.html"], None,
                      question="Adam, will the contact email come from site.json as contact_email?")
    rt, _ = _eve_at_her_desk(w, [asks])
    adam = w["agents"]["adam"]
    adam["current_location"] = "office_1"
    ag.work_on_task(rt, "9:00 am", w)
    assert ("Eve Smith wants to ask Adam Beck about T-2: Adam, will the contact email come from "
            "site.json as contact_email?") in _memories(rt)
    talk = ag.get_chunk_lazy(rt, "9:05 am", w)
    assert talk["target_agent"] == "adam" and talk["question"]
    assert not pb.commitments_for("eve")                                 # not a board matter

    # Even right after a talk, a question about the task gets asked once they're together.
    import main
    main.remember_conversation(rt, ag.Agent.load("adam", w), "8:55 am", "the stand-up")
    rt.agent["current_location"] = "office_1"
    rt.chunk = talk
    assert ag.planned_partner(rt, w, "9:05 am") is adam


def test_a_blocked_report_with_a_question_asks_instead_of_marking_blocked(fake_llm, isolated_data, w):
    stuck = ExecResult("Can't fill the form.", [], False, blocked_reason="no contact key",
                       question="Adam, what key holds the contact email?")
    rt, _ = _eve_at_her_desk(w, [stuck])
    ag.work_on_task(rt, "9:00 am", w)
    assert ag.get_chunk_lazy(rt, "9:05 am", w)["target_agent"] == "adam"
    assert pb.commitments_for("eve") == []


def test_a_review_verdict_goes_straight_to_the_board(fake_llm, isolated_data, w):
    rt, _ = _eve_at_her_desk(w, [PROGRESS])
    _adams_task_in_review(w)
    ag.work_on_task(rt, "9:00 am", w)
    visit = ag.get_chunk_lazy(rt, "9:05 am", w)
    assert visit["board_work"] and visit["activity"].startswith("9:05 am: approve T-1")



# --- from the 75-step run ------------------------------------------------------------
def test_an_approved_task_sends_its_owner_to_the_board_once(fake_llm, isolated_data, w):
    rt, _ = _eve_at_her_desk(w, [PROGRESS])
    pb.update_status(w, "eve", "T-2", "review", note="check the layout")
    pb.approve_task(w, "adam", "T-2")
    ag.finish_approved_tasks(rt, "9:00 am", w)
    visit = ag.get_chunk_lazy(rt, "9:05 am", w)
    assert visit["board_work"] and visit["activity"] == (
        "9:05 am: move T-2 to done on the project board: Adam Beck approved it")
    rt.override = None
    ag.finish_approved_tasks(rt, "9:10 am", w)
    assert rt.override is None                                          # once per approval


def test_no_done_trip_for_a_teammates_task_or_while_at_the_board(fake_llm, isolated_data, w):
    rt, _ = _eve_at_her_desk(w, [PROGRESS])
    _adams_task_in_review(w)
    pb.approve_task(w, "eve", "T-1")
    ag.finish_approved_tasks(rt, "9:00 am", w)                          # Adam's task, not hers
    assert rt.override is None
    pb.update_status(w, "eve", "T-2", "review", note="check it")
    pb.approve_task(w, "adam", "T-2")
    rt.agent["interacting_with"] = pb.BOARD_OBJECT
    ag.finish_approved_tasks(rt, "9:00 am", w)
    assert rt.override is None


def test_a_question_queued_behind_a_board_trip_is_still_asked_after_it(fake_llm, isolated_data, w):
    asks = ExecResult("Built the layout.", ["site/template.html"], True,
                      question="Adam, what goes inside each team list item?")
    rt, _ = _eve_at_her_desk(w, [asks])
    ag.work_on_task(rt, "7:35 am", w)                   # question, then the trip to move it to review
    assert ag.get_chunk_lazy(rt, "7:40 am", w)["board_work"]
    ask = ag.get_chunk_lazy(rt, "7:50 am", w)          # in the run this had expired by now
    assert ask is not None and ask.get("question") and ask["start_time"] == "7:50 am"
    assert ask["activity"].startswith("7:50 am: ask Adam Beck about T-2")


def test_a_promise_about_later_doesnt_send_anyone_to_the_board_now(fake_llm, isolated_data, w):
    rt, _ = _eve_at_her_desk(w, [PROGRESS])
    rt.agent["current_location"] = "conference_room"
    ag.queue_board_visit(rt, "I will wait for the reviewer's confirmation before moving T-2 to done",
                         "9:00 am", w)
    assert rt.queued is None
    ag.queue_board_visit(rt, "I'll move T-2 to review now", "9:00 am", w)
    assert rt.queued["board_work"]


# --- reviews when the budget runs low or out ------------------------------------------
def test_the_last_sessions_are_kept_for_reviews():
    from executor import Budgeted, SessionBudget

    budget = SessionBudget(8)
    hands = Budgeted(FakeExecutor(), budget)
    assert budget.reserve == 2
    for _ in range(6):
        assert hands.available("work")
        budget.take()
    assert not hands.available("work") and hands.available("review")
    budget.take()
    budget.take()
    assert not hands.available("review")
    assert SessionBudget(3).reserve == 0 and SessionBudget().allows("work")   # small or no cap


def test_with_only_the_reserve_left_the_hands_review_but_dont_work(fake_llm, isolated_data, w):
    from executor import Budgeted, ReviewResult, SessionBudget

    rt, fake = _eve_at_her_desk(w, [PROGRESS])
    rt.executor = Budgeted(fake, SessionBudget(1, reserve=1))
    assert ag.work_on_task(rt, "9:00 am", w) is None and fake.calls == []    # her own work waits
    _adams_task_in_review(w)
    assert isinstance(ag.work_on_task(rt, "9:00 am", w), ReviewResult)
    assert fake.review_calls == [("eve", "T-1")]


def test_with_the_hands_done_a_teammates_task_is_reviewed_at_the_board(fake_llm, isolated_data, w):
    from executor import Budgeted, SessionBudget

    rt, fake = _eve_at_her_desk(w, [PROGRESS])
    rt.executor = Budgeted(fake, SessionBudget(0))
    task = _adams_task_in_review(w)
    assert ag.work_on_task(rt, "9:00 am", w) is None
    assert rt.executor is None and rt.hands_out and fake.review_calls == []
    promise = pb.commitments_for("eve")[0]
    assert promise.startswith("review Adam Beck's T-1 on the project board: approve T-1 if")
    assert pb.is_board_commitment(promise)
    assert rt.override["board_work"]

    rt.broad_chunks[0]["start_time"] = "11:00 am"
    assert ag.work_on_task(rt, "11:00 am", w) is None                   # once per review round
    assert len(pb.commitments_for("eve")) == 1
    pb.apply_action(w, w["agents"]["eve"], {"action": "approve_task", "task_id": "T-1"})
    assert task["approved_by"] == ["eve"]                               # the card is enough now


def test_an_agent_without_hands_never_reviews_at_the_board_for_them(fake_llm, isolated_data, w):
    rt, _ = _eve_at_her_desk(w, [PROGRESS])
    rt.executor = None                                                  # GA_EXECUTOR=none
    _adams_task_in_review(w)
    assert ag.work_on_task(rt, "9:00 am", w) is None
    assert pb.commitments_for("eve") == []


def test_with_hands_an_approval_needs_their_review_first(fake_llm, isolated_data, w):
    from executor import ReviewResult

    rt, executor = _eve_at_her_desk(w, [PROGRESS])
    executor.reviews = [ReviewResult("changes", "The build misses a check.", ["test a missing key"]),
                        ReviewResult("approve", "The branch does what the task says")]
    task = _adams_task_in_review(w)
    approve = {"action": "approve_task", "task_id": "T-1"}
    eve = w["agents"]["eve"]
    rt.chunk = dict(rt.chunk, board_work=True)                          # at the board first
    ag.work_on_task(rt, "9:00 am", w)
    with pytest.raises(pb.BoardError, match="hasn't been reviewed yet"):
        pb.apply_action(w, eve, approve)

    rt.chunk = dict(rt.chunk, board_work=False)
    ag.work_on_task(rt, "9:00 am", w)                                   # verdict: changes
    with pytest.raises(pb.BoardError, match="hasn't been reviewed yet"):
        pb.apply_action(w, eve, approve)

    pb.update_status(w, "adam", "T-1", "in_progress")
    pb.update_status(w, "adam", "T-1", "review", note="added the test")  # second round
    rt.broad_chunks[0]["start_time"] = "11:00 am"
    ag.work_on_task(rt, "11:00 am", w)                                  # verdict: approve
    pb.apply_action(w, eve, approve)
    assert task["approved_by"] == ["eve"]
