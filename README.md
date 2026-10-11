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
Reflection (twice a day)
```

---

## Running

```bash
pip install -r requirements.txt        # numpy, ollama, lemminflect
ollama pull qwen3 && ollama pull nomic-embed-text

python scripts/seed_memories.py --reset   # fresh start: clear memories + board, then seed
python src/main.py                     # run the simulation
```

Simulation settings (start time, steps, cooldowns, reflection times, paper-strict
mode) are constants at the top of `src/main.py`. Reacting regenerates the plan as
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
| `src/main.py` | Simulation loop: perceive → decide → act → reflect |
| `src/memory_stream.py`, `memory.py` | The memory stream (observations, plans, reflections) |
| `src/memory_retriever.py` | Recency + importance + relevance retrieval |
| `src/plan.py` | Daily plan, hour/minute decomposition, reactions, agent summaries |
| `src/conversation.py` | Turn-based dialogue between agents |
| `src/reflect.py` | Reflection: questions → retrieval → cited insights |
| `src/world.py`, `perceive.py` | World state and what agents observe |
| `src/project_board.py` | The shared project board: structured tasks and the actions agents take on them |
| `src/llm.py` | The only module that talks to Ollama |
| `src/log.py` | Logging levels and output layout |

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
   task that was started (in progress or blocked) can go to review. Agents
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
   board. A promise about the board made in conversation is kept for the agent's
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
   review all require a note (what it needs, what blocks it, what to check). The
   handoff note goes with the assignment notice. A note can't contradict the
   board: "T-2 can start since T-1 is complete" is refused while T-1 is in
   progress (the same goes for "approved" and "in review").
9. **One task at a time.** An agent can have one task in progress; to start
   another they move the current one to review or blocked. One visit starts at
   most one task, so a WIP refusal is reported once. Re-sending a status
   a task already has is skipped quietly (not refused, not remembered). Tasks
   are things to build or ship: titles that describe a meeting or a review are
   refused. Notes can go on a task or on the board itself. Several refusals for
   the same mistake in one visit are reported as one ("T-2 is not in review
   (same for T-3, T-4)").
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
    approved T-1") is not taken as a commitment.
11. **No gaps.** Replans must cover the rest of the day without gaps; if one
    still leaves a gap, the agent keeps working on their in-progress task in
    their office until the next block starts.
12. **Dependencies.** A task lists only the tasks it truly can't start without
    (a page can be built against mock data before the API exists). When none of
    an agent's tasks can start, the board says so and points at `edit_task` to
    drop a dependency that isn't really needed; plans don't schedule work on a
    waiting task, and promise or start steps for one explain why it can't start.
13. **Honest plans.** Plans only name tasks that are on the board (made-up ids
    are removed) and only people who work here. "Mark T-1 as done" or "move it
    to the done column" is board work and happens at the board. A replan keeps
    the automatic start/finish steps for each task plus one board visit before
    clocking out; plain "update the board" visits right after another board
    visit, repeated steps for the same task, and zero-length blocks are dropped.
    Initial plans get the same cleanup. A board visit lasts at most 15 minutes
    and is one step (one board LLM call), never broken into 5-minute pieces; the
    day starts with a 5-minute clock-in instead of an hour of filler. Task
    titles lose a copied "T-1:" prefix, and a board reply that runs out of
    tokens in its reason still keeps its actions.
14. **Reactions (paper 4.3.1).** When an agent reacts, it carries out the
    reaction as a short block and then regenerates the rest of its day from
    when that block ends (`REACT_REPLANS`; False brings back the old 10-minute
    detour). `REACT_COOLDOWN_STEPS` is 2, the reaction's own 10 minutes.
    Reactions that only repeat what is already happening are ignored without
    starting the cooldown: the reaction the agent is already carrying out, and
    going to update the board while at it or within
    `BOARD_REACTION_GAP_MINUTES` of using it.
15. **Reflection and conversation.** Board memories feed reflection, reactions and
   conversations like any other memory. Changes on the board (a task moving to
   review, a new note) are new observations for whoever walks up next, so they can
   start a conversation there.

`python scripts/seed_memories.py --reset` clears the board (tasks and notes) along
with the memory streams, and puts everyone back at reception for a new day.

## Reflection

Agents reflect twice per simulated day (`REFLECTION_TIMES`, default 12:00 pm and
5:00 pm). Each pass looks at the (up to) 100 most recent observations since the
last reflection, asks for the 3 most salient questions, retrieves memories for
each, and stores 5 insights as `reflection` memories that cite their sources.
Reflections are retrieved like any other memory, and the agent's cached summary is
rebuilt afterwards.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs without Ollama: `tests/conftest.py` swaps `llm.generate` / `llm.embed`
for deterministic fakes, and includes full-loop runs covering planning, reflection,
and the conversation → commitment → replan path.
