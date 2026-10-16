# Generative Agents

A modular implementation of the Generative Agents architecture introduced in the paper *Generative Agents: Interactive Simulacra of Human Behavior* by Joon Sung Park et al. 

## Overview

Generative Agents are autonomous agents capable of maintaining long-term memories, reflecting on experiences, planning future actions, and producing contextually grounded behavior over time.

Rather than responding only to the immediate prompt, agents maintain an evolving internal state composed of:

- Experiences and observations
- Retrieved memories
- Reflections and insights
- Plans and goals

This repository aims to provide a clean, extensible, and production-oriented implementation of those concepts.

The architecture is built around three core capabilities:

1. **Memory**
   - Storing observations and experiences
   - Maintaining an evolving memory stream
   - Preserving historical context

2. **Retrieval**
   - Selecting relevant memories when reasoning
   - Balancing relevance, recency, and importance
   - Supplying context for downstream decisions

3. **Reasoning**
   - Generating reflections from past experiences
   - Creating and revising plans
   - Producing coherent actions over time

These mechanisms closely follow the memory, retrieval, reflection, and planning framework described in the original Generative Agents paper. 

---

## Architecture

The high-level agent lifecycle follows the pattern shown in the original paper:

```text
Perceive
   ↓
Memory Stream
   ↓
Memory Retrieval
   ↓
Reasoning / Reflection
   ↓
Planning
   ↓
Action
   ↓
Memory Update
   ↓
Reflection (when perceived importance adds up to 150)
```

---

## Running

```bash
pip install -r requirements.txt        # numpy, ollama, lemminflect, anthropic
ollama pull qwen3 && ollama pull nomic-embed-text

python scripts/seed_memories.py --reset   # fresh start: clear memories + board, then seed
python src/main.py                     # run the simulation
```

Two model roles (`src/llm.py`): the agents' minds (importance, reactions, planning,
conversation, the board) run on Ollama `qwen3`; their hands (the executor that will
write and test code for a board task) run on a hosted model with tools, Claude Opus 5.5
by default. The executor reads `GA_EXECUTOR_MODEL`, `GA_EXECUTOR_EFFORT` (default `high`)
and the usual Anthropic credentials (`ANTHROPIC_API_KEY` or an `ant auth login` profile).

The hands are off by default (`GA_EXECUTOR=none`: the agents plan, talk and keep the board,
as in the paper). `GA_EXECUTOR=fake` dry-runs the whole flow with scripted results;
`GA_EXECUTOR=local` does real work:

```bash
GA_EXECUTOR=local python src/main.py    # commands run in Docker (GA_SANDBOX=docker)
```

When an agent's step is work on a task it owns and has in progress, at its own desk, its
executor works on that task once per plan block, in `workspaces/<agent>/` (a git checkout
of the local sandbox repo `workspaces/origin.git`, branch `task/T-n`). The clock waits.
What happened comes back as memories ("Eve Smith worked on T-2 ...", "The tests for T-2
pass."). Finished work becomes a promise to move the task to review at the next board
visit; a blocked task sends the agent to the board to mark it blocked, with the reason.
Review and approval stay at the board; once a task is done there, its branch is merged
into the sandbox's main. Commands the model asks for run in a throwaway container with
no network (`GA_SANDBOX_IMAGE`, default `python:3.12-slim`); without Docker they are
refused unless `GA_SANDBOX=local`.

Reviews are grounded in the code: at a desk-work step, a teammate's task waiting in review
comes first. The reviewer's hands check out the teammate's published branch, read the diff
and run the tests, and the verdict becomes a memory and a promise for the board (approve,
or leave a note asking for changes); approving still happens at the board. The hands work
from the lane skills in `skills/` (a shared contract, frontend, backend, deploy, review),
picked by the task's area. A run spends at most `GA_EXECUTOR_MAX_SESSIONS` executor sessions
(default 20 with local hands), shared by both agents. The GitHub version of this flow is
designed, not built: see [docs/github-integration.md](docs/github-integration.md).

What the hands find changes what the agent does next: finished work (the tests pass when
the host reruns them) sends the agent straight to the board to move the task to review,
without the usual two-hour wait; a question only a teammate can answer sends it to ask
them, and the next session on the task gets the answer; a review verdict sends it to the
board to approve or leave the note. Each session's cost is estimated in the log, with a
total at the end. `GA_STEPS` sets the run's length (default 30 steps of 5 minutes).

To see the site the agents have built so far, combined from every task branch and built
and tested in the sandbox:

```bash
python scripts/preview_site.py          # writes preview/ and opens the page
```

The run length (start time, steps) is at the top of `src/main.py`; the agents'
settings (cooldowns, reflection threshold, paper-strict mode) are at the top of
`src/agent.py`, and who each agent is lives in `data/agents/<id>/persona.json`. Reacting regenerates the plan as
in the paper by default (`REACT_REPLANS`). `PAPER_STRICT = True` also turns off the
cost shortcuts (the react question every step about everything perceived, no
cooldown, the whole plan decomposed up front, an object update every step), which
is much slower on a local model.

### Reading the logs

Verbosity is controlled by `GA_LOG`:

| Level | Shows |
| --- | --- |
| `quiet` | warnings, conversations, reflections |
| `info` (default) | the above plus a step header, one line per agent per step, reactions, commitments, replans |
| `debug` | everything: observations, decisions and why, chosen rooms, per-call timings |

```text
━━ STEP 2/30 · 7:05 am ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  💬 ADAM ↔ EVE  (reception)  — greet Eve and suggest syncing
      ADAM : "Morning Eve, want to sync in the conference room?"
      EVE  : "Sounds good, see you there."
  ▸ agreed to meet in conference_room: Sync on the project (30 min)
  ADAM  reception        talking with Eve Smith
  EVE   reception        talking with Adam Beck
```

`GA_LOG=debug python src/main.py` while tuning; `GA_LOG=quiet` for long runs.

## Code map

| File | Role |
| --- | --- |
| `src/main.py` | World clock, step loop, and conversations between agents |
| `src/agent.py` | One agent: perceive → decide → act/react → reflect, and its plan |
| `src/executor.py` | The hands: `ExecResult`, `FakeExecutor`, `LocalExecutor` (tool loop on the executor model) |
| `src/skills.py`, `skills/` | Lane skills (SKILL.md) loaded into the executor by task area |
| `src/workspace.py` | Each agent's git checkout of the sandbox repo, and the Docker/local command runners |
| `src/persona.py` | Loads `data/agents/<id>/persona.json` into the agent's profile |
| `src/memory_stream.py`, `memory.py` | The memory stream (observations, plans, reflections) |
| `src/memory_retriever.py` | Recency + importance + relevance retrieval |
| `src/plan.py` | Daily plan, hour/minute decomposition, reactions, agent summaries |
| `src/conversation.py` | Turn-based dialogue between agents |
| `src/reflect.py` | Reflection: questions → retrieval → cited insights |
| `src/world.py`, `perceive.py` | World state and what agents observe |
| `src/project_board.py` | The shared project board: structured tasks and the actions agents take on them |
| `src/llm.py` | The only module that talks to models: Ollama for the minds, Claude for the executor |
| `src/log.py` | Logging levels, output layout, and timings |

## Project board

The team's work is tracked on a `project_board` object in the conference room. The
data behind it is the top-level `project` record in `data/world/world.json`
(goal, description, status, and a list of structured tasks). Each task has an id
(`T-1`), title, description, area, priority, status
(`todo | in_progress | blocked | review | done`), assignee, creator,
dependencies, a "done when" line, timestamps, and a history of every change.

Nothing about the board is special-cased in the agent loop. It follows the paper's
pattern for objects:

1. **Seeded identity.** The profiles in `plan.py` and the seeds in
   `scripts/seed_memories.py` give each agent the team goal, their side of the
   work, and the habit of checking the board each morning, so it shows up in
   their day plan.
2. **Perception.** An agent in the conference room reads the board as
   observations ("The project board has no tasks on it yet.", "Task T-2 ... is
   todo and unassigned."). Like other observations, a line is stored only when it
   changes, and a new line can trigger a reaction.
3. **Action.** When an agent interacts with the board, one schema-constrained LLM
   call proposes up to 4 actions (`create_task`, `claim_task`, `assign_task`,
   `update_status`, `edit_task`, `approve_task`, `add_note`). The prompt includes what the agent has actually
   been doing on each of their open tasks (retrieved from memory), so status
   changes follow real work. Plain Python validates and applies the actions (no
   duplicate titles, no moving someone else's task, no starting work before its
   dependencies are done) and writes each change to the agent's memory stream.
   The LLM never edits the board JSON directly.
4. **Review.** The assignee moves their task `todo → in_progress → review`. A
   teammate then approves it at the board, and only after that can the assignee
   move it to `done`. Moving it back out of review clears the approvals. Only a
   task that was started (in progress or blocked) can go to review, and only
   started work (in progress or in review) can be blocked; a task that has to
   wait for another gets it in `depends_on` instead. Agents
   at the board can't ask for review within `MIN_MINUTES_BEFORE_REVIEW` of
   starting (review is for finished work), and once a task is in review the
   assignee leaves it there until a teammate approves it or asks for changes in
   a note on the task.
5. **Replanning.** When an agent takes on or finishes a task at the board, the rest
   of their day is replanned around their open tasks once they step away: work
   blocks name the task, and short trips back to the board are scheduled when a
   task starts, finishes or gets stuck, and before lunch and clock-out.
6. **Follow-through.** Anything clearly about the board ("review the project
   board", "update task statuses", "assign T-4 to Adam") is routed to the board
   without an LLM call, and other work in the conference room can't touch the
   board. A promise about the board made in conversation ("move T-1 to review",
   "approve T-4"; not just "keep working on T-2") is kept for the agent's
   next board visit and shown first in that prompt; when it names one of their
   tasks, the prompt says to start that task (and how to set aside the one in
   progress). If they made it while
   standing at the board, they go straight back to it after the conversation.
   Refused actions ("T-1 needs a teammate to approve it first") are remembered,
   and an agent who is assigned a task is notified and replans around it.
7. **Planning from the project's state (paper 4.3).** The day plan is conditioned
   on what the agent remembers about the project (the paper's "summary of their
   recent experiences"), and every replan after a board visit or conversation
   includes what the board means for that agent: gaps in their area, unassigned
   work, reviews and dependencies waiting on them, and their own tasks. When a
   replanned work block names one of their tasks, it starts with a short board
   step if the task is still todo and the last block for it ends with one, so
   starting and finishing work always go through the board. Those board steps
   are checked against the board when the agent gets there: "start T-1" for a
   task already in progress becomes a progress report, and "start T-2" while
   T-1 is still in progress asks whether T-1 is finished first.
8. **Notes.** Handing a task to a teammate, blocking a task, and asking for
   review all require a note (what it needs, what blocks it, what to check).
   `done_when` is optional. Tasks are kept to a few hours of work. The
   handoff note goes with the assignment notice. A note can't contradict the
   board: "T-2 can start since T-1 is complete" is refused while T-1 is in
   progress (the same goes for "approved" and "in review"). A note is for a
   teammate to act on, so one that only narrates what the writer is doing
   ("Eve is currently addressing bugs") is refused unless it asks something or
   names the teammate it is for.
9. **One task at a time.** An agent can have one task in progress; to start
   another they move the current one to review or blocked. One visit starts at
   most one task, so a WIP refusal is reported once. Re-sending a status
   a task already has is skipped quietly (not refused, not remembered). Tasks
   are things to build or ship: titles that describe a meeting or a review are
   refused. Notes can go on a task or on the board itself. Several refusals for
   the same mistake in one visit are reported as one ("T-2 is not in review
   (same for T-3, T-4)"). An agent holds at most `MAX_OPEN_TASKS_PER_AGENT` (3)
   open tasks: past that, a task they add stays unassigned for whoever is free,
   and the board shows each person's open-task count. Reactions follow the same
   rule: "start working on T-5" while T-1 is in progress (or while T-5 waits on
   another task) is ignored instead of replanning the day around it.
10. **Conversations.** Both speakers see what the board says about them and each
    other, so they talk about specific tasks. After a talk both remember it
    ("Adam and Eve talked at 10:25 about …"); the same two agents don't start
    another conversation within `CONVERSATION_COOLDOWN_MINUTES` unless the
    reaction names a task; and a promise an agent already planned around doesn't
    trigger another replan. A conversation also starts without the react LLM
    when the current plan chunk is about a teammate who is in the same room
    ("collaborate with Eve", or a reaction like "greet Eve" that walked over
    to her). A teammate walking into the room is considered even during the
    react cooldown, and a reaction that only changes rooms to keep doing the
    same work ("move to office_1 to keep debugging") moves the agent without
    replanning or starting the cooldown; a move with another purpose ("go to
    the conference room to update the board") is a real reaction. Solo work never lands in a teammate's private office, and the
    location LLM is only asked when the answer isn't obvious (solo work goes to
    the agent's own office without a call). Finer steps of a "collaborate with
    Eve" block only follow Eve when they are still about working with her
    ("discuss", "pair with", "review ... with Eve"; not "the discussion"), and
    a block longer than an hour only follows a teammate when it is real
    collaboration ("work on T-1 with Eve" all morning doesn't).
    Speakers are shown exactly where each task in play stands ("in progress
    since 7:10 am: not in review, not approved, not done") and told that talking
    doesn't change the board; something a speaker says they already did ("I've
    approved T-1") is not taken as a commitment. A conversation ends when a
    speaker starts repeating themselves (half their words the same as an earlier
    line), or claims to have already done what they promised earlier in the
    same talk ("I'll outline the API" ... "I've outlined the API"; nothing gets
    done while two agents talk), in which case that line is dropped. Right after
    two agents talk, a planned "collaborate with" or meeting step becomes the
    agent's own work in their own office (the talk was the meeting), and the
    replan is told they just talked. Every replan keeps the remaining plans
    with a teammate (a stand-up, "collaborate with Adam 10:15-11:30") at their
    time, putting them back if the new plan dropped them, except ones within the
    conversation cooldown of their last talk. A reaction about a teammate's task
    that names nobody ("check on T-2 progress", T-2 being Eve's) becomes a talk
    with that teammate.
11. **No gaps.** Replans must cover the rest of the day without gaps; if one
    still leaves a gap, the agent keeps working on their in-progress task (or
    prepares for the next one they can start) in their office until the next
    block starts. When a work block is broken into hours and 5-15 minute steps,
    the prompt names the task the agent is on, and steps that would wander off
    to the board or a "project management tool" are dropped (board visits are
    planned on their own).
12. **Dependencies.** A task lists only the tasks it truly can't start without
    (a page can be built against mock data before the API exists). When none of
    an agent's tasks can start, the board says so and points at `edit_task` to
    drop a dependency that isn't really needed; plans don't schedule work on a
    waiting task, and promise or start steps for one explain why it can't start.
13. **Honest plans.** Plans only name tasks that are on the board (made-up ids
    are removed) and only people who work here. No prompt carries an example
    task id; prompts say which ids exist, or that the board is empty. A
    conversation line naming a task that isn't on the board is asked for once
    more and then has the id removed, commitments lose made-up ids, and notes
    can't mention them. "Mark T-1 as done" or "move it
    to the done column" is board work and happens at the board. A replan keeps
    the automatic start/finish steps for each task plus one board visit before
    clocking out; plain "update the board" visits right after another board
    visit, repeated steps for the same task, and zero-length blocks are dropped.
    Initial plans get the same cleanup. A board visit lasts at most 15 minutes
    and is one step (one board LLM call), never broken into 5-minute pieces; the
    day starts with a 5-minute clock-in instead of an hour of filler. Task
    titles lose a copied "T-1:" prefix, and a board reply that runs out of
    tokens in its reason still keeps its actions. A plan block that gives a task
    the wrong title ("T-1 'Backend data structure'") gets the real one (a
    shortened title is fine), and a planned "approve T-1" while T-1 isn't in
    review becomes a check on T-1 when the agent gets to the board. The action
    shown in the log follows the board too ("reporting progress on T-2 (already
    in progress)" instead of a stale "starting T-2").
14. **Reactions (paper 4.3.1).** When an agent reacts, it carries out the
    reaction as a short block and then regenerates the rest of its day from
    when that block ends (`REACT_REPLANS`; False brings back the old 10-minute
    detour). `REACT_COOLDOWN_STEPS` is 2, the reaction's own 10 minutes.
    Reactions that only repeat what is already happening are ignored without
    starting the cooldown: the reaction the agent is already carrying out, a
    restatement of the current plan step (including "go to the board" while the
    step is a board visit) or a "continue ..." reaction, and
    going to update the board while at it or within
    `BOARD_REACTION_GAP_MINUTES` of using it. A reaction that is only about one
    of the room's ordinary objects ("check the time_clock", "go to reception to
    check the time_clock with Adam") is ignored too; one that does something
    with it ("fix the broken coffee machine") is not.
15. **Everyone works on their own tasks.** A task belongs to its assignee:
    only they work on it and move it on the board. The seeded identity says
    so (profiles and seed memories), and every layer holds to it:
    - the board: an agent only takes on tasks in their own area (`area` in
      `world.json`; "general" work is anyone's); work for another area is
      added unassigned for its owner, and can only be handed to them;
    - plans: a work block names the agent's own tasks. A teammate's task only
      appears in a review of it or in a block working on it together with the
      teammate (they're named: "pair with Eve on T-2"). Otherwise a board step
      for it is left out, and a work block about it becomes the agent's own work
      at the same time. This holds for the hour and 5-15 minute steps too;
    - a task id next to another task's title is the wrong id: it moves to the
      task with that title, or is dropped and the described work stays;
    - conversations: speakers are told whose tasks are whose, and that helping
      with a teammate's task means offering to work on it together, which the
      teammate can accept or turn down;
    - promises: one about a teammate's task counts only if it's a review of it
      or working on it together with them; "I'll get T-2 ready for review"
      from someone who doesn't own T-2 is dropped.
16. **Agreed meetings (paper: Isabella's party).** When two agents talk, each
    sees what they've planned with the other later today, so they can confirm a
    time and place or suggest one. A meeting they both agree on, right afterwards
    or later today ("stand-up at 8:00 in the conference room"), goes into both
    their plans as a block pinned to that place, and both remember the
    agreement. Every replan keeps it, even one right after their talk, and a
    replan's own version of time with that teammate is cut around it instead of
    doubling it. When they're both there they talk (inside the conversation
    cooldown too: the meeting is the point), once; then each goes back to their
    own task. A plan block with a teammate that nobody agreed to is still just
    one agent's intention.
17. **Reflection and conversation.** Board memories feed reflection, reactions and
   conversations like any other memory. Changes on the board (a task moving to
   review, a new note) are new observations for whoever walks up next, so they can
   start a conversation there.

`python scripts/seed_memories.py --reset` clears the board (tasks and notes) along
with the memory streams, and puts everyone back at reception for a new day.

## Reflection

As in the paper (section 4.2), an agent reflects when the importance of the
events it has perceived since its last reflection adds up to
`REFLECTION_THRESHOLD` (150, the paper's value; `None` turns reflection off).
Seeded memories don't count toward the first one: they aren't events the agent
perceived. Each pass looks at the (up to) 100 most recent observations, asks
for the 3 most salient questions, retrieves memories for each, and stores 5
insights as `reflection` memories that cite their sources. Reflections are
retrieved like any other memory, and the agent's cached summary is rebuilt
afterwards. How often this fires depends on how much an agent perceives: time
at the board means many observations, so expect more passes on busy mornings.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs without Ollama or an API key: `tests/conftest.py` swaps `llm.generate` / `llm.embed`
for deterministic fakes, and includes full-loop runs covering planning, reflection,
and the conversation → commitment → replan path.
