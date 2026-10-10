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
mode) are constants at the top of `src/main.py`.

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
   call proposes up to 4 actions (`create_task`, `claim_task`, `update_status`,
   `approve_task`, `add_note`). The prompt includes what the agent has actually
   been doing on each of their open tasks (retrieved from memory), so status
   changes follow real work. Plain Python validates and applies the actions (no
   duplicate titles, no moving someone else's task, no starting work before its
   dependencies are done) and writes each change to the agent's memory stream.
   The LLM never edits the board JSON directly.
4. **Review.** The assignee moves their task `todo → in_progress → review`. A
   teammate then approves it at the board, and only after that can the assignee
   move it to `done`. Moving it back out of review clears the approvals.
5. **Replanning.** When an agent takes on or finishes a task at the board, the rest
   of their day is replanned around their open tasks once they step away: work
   blocks name the task, and short trips back to the board are scheduled when a
   task starts, finishes or gets stuck, and before lunch and clock-out.
6. **Reflection and conversation.** Board memories feed reflection, reactions and
   conversations like any other memory. Changes on the board (a task moving to
   review, a new note) are new observations for whoever walks up next, so they can
   start a conversation there.

`python scripts/seed_memories.py --reset` clears the board along with the memory
streams.

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
