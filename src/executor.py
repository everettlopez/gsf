"""
executor.py - the agents' hands.

The generative-agents layer is the mind: it decides, plans, talks and keeps the
board. When an agent's current step is work on a task it owns and that task is
in progress, the mind hands the task to an executor, which does the actual
work in the agent's own workspace and reports back:

    executor.run(agent, task, workspace) -> ExecResult

The simulation clock waits while it works (the call is synchronous). The
result goes back into the agent's memory stream as observations, where
importance scoring, retrieval, reflection and conversation pick it up like
anything else the agent experienced (see agent.work_on_task). The executor
never touches the board: starting, blocking, review and approval stay with the
agents at the board.

    FakeExecutor    scripted results, for tests and dry runs
    LocalExecutor   a tool-use loop on the executor model (llm.chat) that reads,
                    writes and tests files in the agent's workspace (workspace.py)
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# How the tests are run, here and in the skills (skills/shared/SKILL.md).
TEST_COMMAND = "python -m unittest discover -s tests"


def short(text, sentences=2, limit=300):
    """The gist of a report: its first sentence or two, as plain text. Models
    write long markdown reports; a memory, a commit subject or a board note
    gets this, and the full report goes in the commit body."""
    text = (text or "").strip().split("\n\n")[0]
    text = re.sub(r"[`*#>]+", "", text)
    text = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+", text)
    gist = " ".join(parts[:sentences])
    return gist if len(gist) <= limit else gist[:limit - 1].rsplit(" ", 1)[0] + "…"


@dataclass
class ExecResult:
    """What one work session on a task produced."""
    summary: str                                    # one or two plain sentences
    files_changed: list = field(default_factory=list)
    tests_passed: Optional[bool] = None             # None: no tests were run
    blocked_reason: str = ""                        # why the work can't go on, if it can't
    question: str = ""                              # what to ask the teammate, if anything
    cost: float = 0.0                               # estimated $ of the session

    @property
    def blocked(self):
        return bool(self.blocked_reason)

    @property
    def finished(self):
        """Done as far as the hands can tell: tests pass and nothing blocks it.
        Whether it really is done is for a teammate to judge in review."""
        return bool(self.tests_passed) and not self.blocked


@dataclass
class ReviewResult:
    """What a review of a teammate's task found."""
    verdict: str                                    # approve | changes | none (couldn't review)
    summary: str
    changes: list = field(default_factory=list)     # for "changes": one concrete request each
    tests_passed: Optional[bool] = None
    cost: float = 0.0


class Executor:
    """Does the work on one task for one agent (run), or reviews a teammate's
    task (review)."""

    def run(self, agent, task, workspace):
        raise NotImplementedError

    def review(self, agent, task, workspace, owner_name=""):
        raise NotImplementedError

    def available(self, kind="work"):
        """Can this executor take another session of this kind, "work" or
        "review"? (See Budgeted.)"""
        return True


REVIEW_RESERVE = 2                  # sessions only reviews may spend


class SessionBudget:
    """How many executor sessions (work or review) one run may spend, shared
    by every agent. None means no limit.

    The last few are kept for reviews: work that has nobody to review it
    waits in review all day. In the 75-step run the last session finished
    Eve's T-3, and it sat in review from 9:45 until the run ended."""

    def __init__(self, sessions=None, reserve=None):
        self.left = sessions
        self.spent = 0.0                            # estimated $ so far
        if reserve is None:
            reserve = 0 if sessions is None else min(REVIEW_RESERVE, sessions // 4)
        self.reserve = reserve

    def allows(self, kind="work"):
        if self.left is None:
            return True
        return self.left > (self.reserve if kind == "work" else 0)

    def take(self):
        if self.left is None:
            return True
        if self.left <= 0:
            return False
        self.left -= 1
        return True


class Budgeted(Executor):
    """An executor that stops taking sessions when the run's budget is spent."""

    def __init__(self, inner, budget):
        self.inner, self.budget = inner, budget

    def available(self, kind="work"):
        return self.budget.allows(kind)

    def run(self, agent, task, workspace):
        self.budget.take()
        result = self.inner.run(agent, task, workspace)
        self.budget.spent += result.cost
        return result

    def review(self, agent, task, workspace, owner_name=""):
        self.budget.take()
        result = self.inner.review(agent, task, workspace, owner_name)
        self.budget.spent += result.cost
        return result


class FakeExecutor(Executor):
    """Returns scripted results and remembers what it was asked to do.

    `results` is a list of ExecResult (used in order, the last one repeating)
    or a function (agent, task, workspace) -> ExecResult."""

    def __init__(self, results=None, reviews=None):
        self.results = results if results is not None else [
            ExecResult("Made progress on the task.", tests_passed=None)]
        self.reviews = reviews if reviews is not None else [
            ReviewResult("approve", "The branch does what the task says.", tests_passed=True)]
        self.calls = []                                 # (agent id, task id, workspace)
        self.review_calls = []                          # (agent id, task id)

    def review(self, agent, task, workspace, owner_name=""):
        self.review_calls.append((agent.agent["id"], task["id"]))
        return self.reviews[min(len(self.review_calls), len(self.reviews)) - 1]

    def run(self, agent, task, workspace):
        self.calls.append((agent.agent["id"], task["id"], workspace))
        if callable(self.results):
            return self.results(agent, task, workspace)
        return self.results[min(len(self.calls), len(self.results)) - 1]


# ---------------------------------------------------------------------------
# LocalExecutor: a tool-use loop on the executor model
# ---------------------------------------------------------------------------
def _tool(name, description, properties, required=None):
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": properties,
                             "required": required if required is not None else list(properties),
                             "additionalProperties": False}}


_PATH = {"type": "string", "description": "Path relative to the workspace root"}

TOOLS = [
    _tool("list_files", "List the files under a directory of the workspace (default: all of it).",
          {"path": _PATH}, required=[]),
    _tool("read_file", "Read a text file from the workspace.", {"path": _PATH}),
    _tool("write_file", "Create or overwrite a text file in the workspace with its full contents.",
          {"path": _PATH, "contents": {"type": "string", "description": "The whole file"}}),
    _tool("run_command", "Run a shell command in the workspace root (a container with no "
          "network) and get its exit code and output.", {"command": {"type": "string"}}),
    _tool("run_tests", "Run tests with this command, to check the work as you go.",
          {"command": {"type": "string"}}),
    _tool("report", "End this work session and report back.",
          {"summary": {"type": "string", "description": "One or two plain sentences: what is "
                       "done now and what is left"},
           "blocked_reason": {"type": "string", "description": "Only if the work can't go on "
                              "without something outside this task (a decision, an account, a "
                              "teammate's answer): what that is"},
           "question": {"type": "string", "description": "Only if a teammate's answer would "
                        "help this task: the question to ask them, naming the task"}},
          required=["summary"]),
]
REVIEW_TOOLS = [t for t in TOOLS if t["name"] in ("list_files", "read_file", "run_command",
                                                  "run_tests")] + [
    _tool("show_changes", "Show what the branch changes against main (git diff).", {}),
    _tool("review_report", "End the review with a verdict.",
          {"verdict": {"type": "string", "enum": ["approve", "changes"]},
           "summary": {"type": "string", "description": "One or two plain sentences: what the "
                       "branch does and how it holds up"},
           "changes": {"type": "string", "description": "For a changes verdict: each change "
                       "needed on its own line, most important first"}},
          required=["verdict", "summary"]),
]


def _invalid_input(name, given, schemas=None):
    """Why a tool input doesn't match its schema, or "". Streamed tool inputs
    aren't validated by the API, so they are checked here before running."""
    schemas = schemas or {t["name"]: t["input_schema"] for t in TOOLS}
    schema = schemas.get(name)
    if schema is None:
        return f"there is no tool {name!r}"
    if not isinstance(given, dict):
        return "the input must be an object"
    missing = [k for k in schema["required"] if k not in given]
    extra = [k for k in given if k not in schema["properties"]]
    wrong = [k for k, v in given.items() if k in schema["properties"] and not isinstance(v, str)]
    off = [f"{k} must be one of {', '.join(schema['properties'][k]['enum'])}"
           for k, v in given.items() if k in schema["properties"]
           and "enum" in schema["properties"][k] and v not in schema["properties"][k]["enum"]]
    problems = ([f"missing {', '.join(missing)}"] if missing else []) + \
               ([f"unknown {', '.join(extra)}"] if extra else []) + \
               ([f"{', '.join(wrong)} must be text"] if wrong else []) + off
    return "; ".join(problems)


class _Session:
    """What happened in one work session, as the tools saw it."""

    def __init__(self):
        self.report = None
        self.usage = {}              # tokens over the session
        self.model = ""
        self.counter = 0             # tool calls so far
        self.last_change = 0         # the call that last changed files (a write or a command)
        self.tests = None            # (call, passed) of the last run_tests

    def tests_passed(self):
        if self.tests is None or self.tests[0] < self.last_change:
            return None              # never run, or run before the latest change
        return self.tests[1]

    def add_usage(self, reply):
        self.model = reply.model or self.model
        for key, value in (reply.usage or {}).items():
            self.usage[key] = self.usage.get(key, 0) + value

    def cost(self):
        import llm
        return llm.cost(self.model, self.usage)


class LocalExecutor(Executor):
    """Works on a task, or reviews a teammate's, with the executor model
    (llm.chat) and the tools above, in the agent's workspace.

    A work session runs on the task's branch; the next one picks up from the
    branch, the task's notes and the summaries of earlier sessions. A review
    session reads and runs a teammate's published branch and changes nothing.
    The skills for the task (skills.py) go into the system prompt.

    tests_passed is not the model's word for it: the host runs the tests
    itself when the session ends (TEST_COMMAND) and takes that outcome."""

    def __init__(self, max_turns=30, command_timeout=None):
        self.max_turns = max_turns
        self.command_timeout = command_timeout

    # -- prompts -------------------------------------------------------------
    @staticmethod
    def _team(agent):
        team = getattr(agent, "team", None) or []
        if not team:
            return ""
        people = ", ".join(f"{name} ({area})" for name, area in team)
        return (f"The team is {people}; nobody else works at the company, so content that "
                f"names the team names them.")

    @staticmethod
    def _who(agent):
        import persona

        a = agent.agent
        try:
            p = persona.load(a["id"])
            return " ".join(p["occupation"][:2]), " ".join(p.get("work_style", []))
        except FileNotFoundError:
            return f"{a['name']} is a software engineer.", ""

    def system_prompt(self, agent, skill_names):
        import skills

        a = agent.agent
        who, style = self._who(agent)
        return f"""You are the hands of {a['name']}, a {a.get('area') or 'software'} engineer on a \
two-person team building the company website. {who} {style}

{self._team(agent)}

{a['name']} decides what to work on; you do the work on the one task you are given, in \
{a['name']}'s own copy of the team's repository, on the task's branch. Stay inside that task \
and {a['name']}'s area: a teammate's work is theirs to change.

If a teammate could answer something that would help, put it in the report's question: \
they will be asked, and the answer comes with the next session on this task.

Each task is built on its own branch from main, and the teammates' halves meet only once both \
are approved and merged. So a teammate's files missing from your branch is expected, not a \
blocker: build and test against a small stand-in, as the conventions below say. Report \
blocked only when this task itself can't go on without something outside it (a decision, an \
account, a teammate's answer to a question).

Commands run in a Linux container with Python 3.12 and no network, so nothing can be \
installed: build with what is there and test with the standard library. You don't commit \
(there is no git in the container): when the session ends, the work is committed and, if its \
tests pass, published for review. The tests are run again then with \
`{TEST_COMMAND}`, and that run is what counts. End the session with report: a sentence or \
two on what is done and what is left.

{skills.load(skill_names)}"""

    def review_prompt(self, agent):
        import skills

        a = agent.agent
        who, _ = self._who(agent)
        return f"""You are the hands of {a['name']}, reviewing a teammate's task on a two-person \
team building the company website. {who}

The teammate's branch is checked out in {a['name']}'s copy of the repository. Read it, run it, \
and end with review_report. Change nothing: anything written here is thrown away. Commands run \
in a Linux container with Python 3.12 and no network; use show_changes for the diff.

{skills.load(skills.for_review())}"""

    @staticmethod
    def _task_lines(task):
        lines = [f"Task {task['id']}: {task['title']}"]
        if task.get("description"):
            lines.append(f"What it is: {task['description']}")
        if task.get("done_when"):
            lines.append(f"Done when: {task['done_when']}")
        notes = [e for e in task.get("history", []) if e.get("event") == "note"]
        if notes:
            lines.append("Notes on the task, oldest first:")
            lines += [f"- {e['agent']} at {e['time']}: {e['note']}" for e in notes]
        return lines

    @staticmethod
    def _known(agent, task, n=6):
        """What the agent remembers that bears on the task (a teammate's answer,
        what was agreed in a talk), from its memory stream. Not its plans, and
        not the work reports, which come under earlier sessions."""
        retriever = getattr(agent, "retriever", None)
        if retriever is None:
            return []
        name, out = agent.agent["name"], []
        for found in retriever.retrieve(f"{task['id']} {task['title']}", n=n * 2, touch=False):
            memory = found["memory"]
            if memory.memory_type == "plan" or memory.description.startswith(f"{name} worked on"):
                continue
            out.append(memory.description)
        return out[:n]

    def brief(self, agent, task):
        lines = self._task_lines(task)
        others = self._own_other_tasks(agent, task)
        if others:
            lines.append(f"{agent.agent['name']}'s other tasks (build on their work, don't redo it; "
                         f"any of it that is finished is already on this branch):")
            for t in others:
                state = t["status"].replace("_", " ")
                if t["status"] == "review" and t.get("approved_by"):
                    state += ", approved"
                lines.append(f"- {t['id']} '{t['title']}' ({state})")
        before = [s["result"].summary for s in agent.work_log.get(task["id"], [])]
        if before:
            lines.append("Earlier work sessions on this task (their files are on the branch):")
            lines += [f"- {short(s)}" for s in before]
        known = self._known(agent, task)
        if known:
            lines.append(f"What {agent.agent['name']} remembers that bears on it:")
            lines += [f"- {k}" for k in known]
        lines.append("Work on it now.")
        return "\n".join(lines)

    def review_brief(self, task, owner_name):
        return "\n".join([f"Review {owner_name}'s task, now in review.", *self._task_lines(task),
                          "Review it now."])

    # -- tools ---------------------------------------------------------------
    def _call(self, session, workspace, call, schemas):
        from workspace import WorkspaceError

        session.counter += 1
        problem = _invalid_input(call.name, call.input, schemas)
        if problem:
            return self._result(call, f"Invalid input: {problem}", error=True)
        args = call.input
        try:
            if call.name == "list_files":
                files = workspace.list_files(args.get("path") or ".")
                return self._result(call, "\n".join(files) or "(no files)")
            if call.name == "read_file":
                return self._result(call, workspace.read_file(args["path"]))
            if call.name == "write_file":
                workspace.write_file(args["path"], args["contents"])
                session.last_change = session.counter
                return self._result(call, f"Wrote {args['path']}.")
            if call.name == "show_changes":
                return self._result(call, workspace.diff() or "(no changes against main)")
            if call.name in ("run_command", "run_tests"):
                timeout = {"timeout": self.command_timeout} if self.command_timeout else {}
                done = workspace.run(args["command"], **timeout)
                if call.name == "run_tests":
                    session.tests = (session.counter, done.ok)
                else:
                    session.last_change = session.counter     # a command can change files
                return self._result(call, f"exit code {done.exit_code}\n{done.output}")
            session.report = args                             # report / review_report
            return self._result(call, "Reported.")
        except WorkspaceError as e:
            return self._result(call, str(e), error=True)

    @staticmethod
    def _result(call, text, error=False):
        block = {"type": "tool_result", "tool_use_id": call.id, "content": text}
        if error:
            block["is_error"] = True
        return block

    # -- the loop ------------------------------------------------------------
    def _final_tests(self, workspace):
        """The tests, run by the host at the end of the session: whether they
        pass is this, not what the model said. None if there are none."""
        try:
            if not [f for f in workspace.list_files("tests") if f.endswith(".py")]:
                return None
        except Exception:
            return None
        done = workspace.run(TEST_COMMAND)
        if "NO TESTS RAN" in done.output or "Ran 0 tests" in done.output:
            return None
        return done.ok

    def _session(self, workspace, system, brief, tools):
        """One bounded tool-use conversation. Returns (session, final text, why
        it was stopped from outside the work, or "")."""
        import llm

        schemas = {t["name"]: t["input_schema"] for t in tools}
        session = _Session()
        messages = [{"role": "user", "content": brief}]
        unreadable = 0
        for _ in range(self.max_turns):
            try:
                reply = llm.chat(messages, system=system, tools=tools)
                session.add_usage(reply)
            except llm.ToolInputError:
                unreadable += 1                   # no tool_use to answer: send the turn again
                if unreadable > 2:
                    return session, "", "the executor model kept sending tool input that couldn't be read"
                continue
            if reply.stop_reason == "refusal":
                return session, "", "the executor model declined to work on this task"
            messages.append({"role": "assistant", "content": reply.content})
            if reply.stop_reason == "max_tokens":
                # A cut-off turn may hold a half-written tool call: none of it runs.
                cut = [self._result(c, "Cut off at the output limit, so not run. Write smaller "
                                       "pieces.", error=True) for c in reply.tool_calls]
                messages.append({"role": "user", "content": cut or
                                 "You were cut off at the output limit. Continue in smaller pieces."})
                continue
            if not reply.tool_calls:
                return session, reply.text.strip(), ""
            messages.append({"role": "user", "content": [
                self._call(session, workspace, call, schemas) for call in reply.tool_calls]})
            if session.report:
                return session, "", ""
        return session, "The session ran out of turns before it was finished.", ""

    @staticmethod
    def _own_other_tasks(agent, task):
        world = getattr(agent, "world", None)
        if not world or "project" not in world:
            return []
        return [t for t in world["project"]["tasks"]
                if t["assignee"] == agent.agent["id"] and t["id"] != task["id"]]

    def run(self, agent, task, workspace):
        import skills

        others = self._own_other_tasks(agent, task)
        workspace.start_task(task["id"], build_on=[t["id"] for t in others
                                                    if t["status"] in ("review", "blocked", "done")])
        session, text, stopped = self._session(
            workspace, self.system_prompt(agent, skills.for_task(task)),
            self.brief(agent, task), TOOLS)
        report = session.report or {}
        result = ExecResult(summary=(report.get("summary") or text).strip() or "Worked on the task.",
                            tests_passed=self._final_tests(workspace),
                            blocked_reason=stopped or (report.get("blocked_reason") or "").strip(),
                            question=(report.get("question") or "").strip(),
                            cost=session.cost())
        workspace.commit(f"{task['id']}: {short(result.summary, 1, 120)}", body=result.summary)
        result.files_changed = workspace.changed_files()
        if result.finished:
            workspace.publish(task["id"])
        return result

    def review(self, agent, task, workspace, owner_name=""):
        if not workspace.checkout_review(task["id"]):
            return ReviewResult("none", f"{task['id']} has no published branch to review yet.")
        try:
            session, text, stopped = self._session(
                workspace, self.review_prompt(agent),
                self.review_brief(task, owner_name or task.get("assignee") or "a teammate"),
                REVIEW_TOOLS)
            tests = self._final_tests(workspace)
        finally:
            workspace.discard_changes()
        report = session.report or {}
        if stopped or not report:
            return ReviewResult("none", stopped or text or "The review didn't reach a verdict.",
                                tests_passed=tests, cost=session.cost())
        changes = [c.strip(" -*") for c in (report.get("changes") or "").splitlines() if c.strip(" -*")]
        return ReviewResult(report["verdict"], report["summary"].strip(), changes,
                            tests_passed=tests, cost=session.cost())


DEFAULT_MAX_SESSIONS = 20          # per run, for real (local) hands


def run_budget():
    """The session budget for one run: GA_EXECUTOR_MAX_SESSIONS, or 20 for
    local hands (each session is a paid model conversation), no limit for fake."""
    import os

    value = os.environ.get("GA_EXECUTOR_MAX_SESSIONS")
    if value:
        return SessionBudget(int(value))
    local = os.environ.get("GA_EXECUTOR") == "local"
    return SessionBudget(DEFAULT_MAX_SESSIONS if local else None)


def hands_for(agent, budget=None):
    """The executor and workspace for one agent, from GA_EXECUTOR:

        none   (default) no hands: the agents plan, talk and keep the board, as in the paper
        fake   FakeExecutor, no workspace: a dry run of the whole flow, no model, no files
        local  LocalExecutor in workspaces/<agent>/ (commands in Docker, see workspace.py)

    Pass the run's budget (run_budget()) so the agents share one cap."""
    import os

    budget = budget or SessionBudget()
    choice = os.environ.get("GA_EXECUTOR", "none")
    if choice == "none":
        return None, None
    if choice == "fake":
        return Budgeted(FakeExecutor(), budget), None
    if choice == "local":
        from workspace import Workspace
        return Budgeted(LocalExecutor(), budget), Workspace(agent)
    raise ValueError(f"unknown GA_EXECUTOR {choice!r} (none, fake or local)")
