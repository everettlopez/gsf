"""
Runs longer than a day (GA_DAYS): the clock moves on to the next workday, the
board keeps time across nights, and each morning the agents plan afresh from
the day before while their memories, the board and the hands' work carry over.
"""

import agent as ag
import chronicle
import clock
import log
import main
import persona
import project_board as pb
from memory_stream import MemoryStream


# --- the clock -----------------------------------------------------------------------
def test_workdays_have_names_and_skip_the_weekend():
    assert clock.day() == 1 and clock.day_name() == "Thursday"
    start = clock.now().date()
    clock.next_day()
    assert (clock.day(), clock.day_name(), clock.now().strftime("%I:%M %p")) == (2, "Friday", "07:00 AM")
    assert (clock.now().date() - start).days == 1
    clock.next_day()
    assert clock.day_name() == "Monday"
    clock.reset()
    assert clock.day() == 1


# --- the board across a night --------------------------------------------------------
def test_yesterdays_work_on_a_task_counts_as_long_ago(w):
    clock.set_time("4:30 pm")
    pb.create_task(w, "eve", "Board page", area="frontend", assign_to_self=True)
    pb.update_status(w, "eve", "T-1", "in_progress")
    clock.next_day()
    clock.set_time("7:10 am")
    task = pb.get_task(w, "T-1")
    assert pb.minutes_in_progress(task) == 14 * 60 + 40               # not -1280
    assert pb.started_at(task) == "Thursday 4:30 pm"
    pb.apply_action(w, w["agents"]["eve"], {"action": "update_status", "task_id": "T-1",
                                            "status": "review", "note": "the board page is done"})
    assert task["status"] == "review"                                  # not "too soon"
    assert task["history"][-1]["day"] == 2


# --- the run ---------------------------------------------------------------------------
def test_run_length_comes_from_days_or_steps(monkeypatch):
    monkeypatch.setattr(main, "NUM_STEPS", 30)
    assert main.run_length() == (1, 30)
    monkeypatch.setenv("GA_STEPS", "8")
    assert main.run_length() == (1, 8)
    monkeypatch.setenv("GA_DAYS", "2")
    assert main.run_length() == (2, 8)                                 # 2 days of 8 steps
    monkeypatch.delenv("GA_STEPS")
    assert main.run_length() == (2, main.DAY_STEPS)                    # 2 whole workdays


def test_a_two_day_run_plans_each_morning_from_the_day_before(fake_llm, isolated_data,
                                                              monkeypatch, capsys):
    monkeypatch.setenv("GA_DAYS", "2")
    monkeypatch.setenv("GA_STEPS", "4")
    monkeypatch.setattr(ag, "REFLECTION_THRESHOLD", None)
    prompts = []
    import llm

    real = llm.generate

    def remember(prompt, options=None, format=None):
        prompts.append(prompt)
        return real(prompt, options, format)

    monkeypatch.setattr(llm, "generate", remember)
    log.timings.clear()
    main.main()
    out = capsys.readouterr().out

    assert "2 days x 4 steps" in out
    assert "STEP 4/8 · Thursday 7:15 am" in out and "STEP 5/8 · Friday 7:00 am" in out
    assert "Thursday is over: everyone goes home" in out
    assert clock.day() == 2 and clock.day_name() == "Friday"

    day_plans = [p for p in prompts if "Here is " in p and "plan today in broad strokes" in p]
    assert len(day_plans) == 4                                         # each agent, each morning
    assert all("Today is Thursday" in p for p in day_plans[:2])
    assert all("Today is Friday" in p for p in day_plans[2:])
    assert "On Thursday, Adam:" in day_plans[2] and "On Wednesday, Adam:" in day_plans[0]

    saved = chronicle.load()
    assert [d["name"] for d in saved["days"]] == ["Thursday", "Friday"]
    assert {s["day"] for s in saved["steps"]} == {1, 2}
    assert len(saved["steps"]) == 8 * 2                                # every step, both agents
    memories = [m.description for m in MemoryStream("adam").memories]
    assert "Adam Beck finished the workday on Thursday and went home." in memories



def test_a_new_day_lets_go_of_yesterdays_plan_but_not_the_work(fake_llm, isolated_data, w):
    rt = ag.Agent.load("eve", w)
    ag.initial_planning(rt, w)
    rt.override = {"activity": "7:45 am: move T-1 to review", "start_time": "7:45 am"}
    rt.work_log["T-1"] = [{"block": "day 1 9:00 am", "result": None, "history": 0}]
    rt.agent["current_location"] = "office_2"
    ag.end_day(rt)
    clock.next_day()
    ag.start_day(rt, w)
    assert rt.override is None and rt.agent["current_location"] == "reception"
    assert rt.work_log["T-1"]                                          # the work carries over
    assert rt.broad_chunks                                             # planned afresh
    assert ag._work_block(rt, "9:00 am").startswith("day 2 ")          # a new day's blocks

