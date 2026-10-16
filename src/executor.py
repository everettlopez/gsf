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

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExecResult:
    """What one work session on a task produced."""
    summary: str                                    # one or two plain sentences
    files_changed: list = field(default_factory=list)
    tests_passed: Optional[bool] = None             # None: no tests were run
    blocked_reason: str = ""                        # why the work can't go on, if it can't

    @property
    def blocked(self):
        return bool(self.blocked_reason)

    @property
    def finished(self):
        """Done as far as the hands can tell: tests pass and nothing blocks it.
        Whether it really is done is for a teammate to judge in review."""
        return bool(self.tests_passed) and not self.blocked


class Executor:
    """Does the work on one task for one agent, then returns an ExecResult."""

    def run(self, agent, task, workspace):
        raise NotImplementedError


class FakeExecutor(Executor):
    """Returns scripted results and remembers what it was asked to do.

    `results` is a list of ExecResult (used in order, the last one repeating)
    or a function (agent, task, workspace) -> ExecResult."""

    def __init__(self, results=None):
        self.results = results if results is not None else [
            ExecResult("Made progress on the task.", tests_passed=None)]
        self.calls = []                                 # (agent id, task id, workspace)

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
    _tool("run_tests", "Run the task's tests with this command. Whether the task's tests pass "
          "is taken from the last run_tests after the last file change.",
          {"command": {"type": "string"}}),
    _tool("report", "End this work session and report back.",
          {"summary": {"type": "string", "description": "One or two plain sentences: what is "
                       "done now and what is left"},
           "blocked_reason": {"type": "string", "description": "Only if the work can't go on "
                              "without something outside this task (a teammate's work, an "
                              "account, a decision): what that is"}},
          required=["summary"]),
]
_SCHEMAS = {t["name"]: t["input_schema"] for t in TOOLS}


def _invalid_input(name, given):
    """Why a tool input doesn't match its schema, or "". Streamed tool inputs
    aren't validated by the API, so they are checked here before running."""
    schema = _SCHEMAS.get(name)
    if schema is None:
        return f"there is no tool {name!r}"
    if not isinstance(given, dict):
        return "the input must be an object"
    missing = [k for k in schema["required"] if k not in given]
    extra = [k for k in given if k not in schema["properties"]]
    wrong = [k for k, v in given.items() if k in schema["properties"] and not isinstance(v, str)]
    problems = ([f"missing {', '.join(missing)}"] if missing else []) + \
               ([f"unknown {', '.join(extra)}"] if extra else []) + \
               ([f"{', '.join(wrong)} must be text"] if wrong else [])
    return "; ".join(problems)


class _Session:
    """What happened in one work session, as the tools saw it."""

    def __init__(self):
        self.report = None
        self.counter = 0             # tool calls so far
        self.last_change = 0         # the call that last changed files (a write or a command)
        self.tests = None            # (call, passed) of the last run_tests

    def tests_passed(self):
        if self.tests is None or self.tests[0] < self.last_change:
            return None              # never run, or run before the latest change
        return self.tests[1]


class LocalExecutor(Executor):
    """Works on a task with the executor model (llm.chat) and the tools above,
    in the agent's workspace, on the task's branch. One run is one bounded work
    session; the next one picks up from the branch, the task's notes and the
    summaries of earlier sessions.

    tests_passed is not the model's word for it: it is the outcome of the last
    run_tests, and only if nothing changed after it."""

    def __init__(self, max_turns=30, command_timeout=None):
        self.max_turns = max_turns
        self.command_timeout = command_timeout

    # -- prompts -------------------------------------------------------------
    def system_prompt(self, agent):
        import persona

        a = agent.agent
        try:
            p = persona.load(a["id"])
            who = " ".join(p["occupation"][:2])
            style = " ".join(p.get("work_style", []))
        except FileNotFoundError:
            who, style = f"{a['name']} is a software engineer.", ""
        return f"""You are the hands of {a['name']}, a {a.get('area') or 'software'} engineer on a \
two-person team building the company website. {who} {style}

{a['name']} decides what to work on; you do the work on the one task you are given, in \
{a['name']}'s own copy of the team's repository, on the task's branch. Stay inside that task \
and {a['name']}'s area: a teammate's work is theirs to change. When the task needs something \
from a teammate, give that as the blocked reason instead of doing their part.

Commands run in a Linux container with Python 3.12 and no network, so nothing can be \
installed: build with what is there and test with the standard library (for a web page, \
html.parser and unittest are enough). Check the work with run_tests before reporting it \
finished, and end the session with report."""

    def brief(self, agent, task):
        lines = [f"Task {task['id']}: {task['title']}"]
        if task.get("description"):
            lines.append(f"What it is: {task['description']}")
        if task.get("done_when"):
            lines.append(f"Done when: {task['done_when']}")
        notes = [e for e in task.get("history", []) if e.get("event") == "note"]
        if notes:
            lines.append("Notes on the task, oldest first:")
            lines += [f"- {e['agent']} at {e['time']}: {e['note']}" for e in notes]
        before = [s["result"].summary for s in agent.work_log.get(task["id"], [])]
        if before:
            lines.append("Earlier work sessions on this task (their files are on the branch):")
            lines += [f"- {s}" for s in before]
        lines.append("Work on it now.")
        return "\n".join(lines)

    # -- tools ---------------------------------------------------------------
    def _call(self, session, workspace, call):
        from workspace import WorkspaceError

        session.counter += 1
        problem = _invalid_input(call.name, call.input)
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
            if call.name in ("run_command", "run_tests"):
                timeout = {"timeout": self.command_timeout} if self.command_timeout else {}
                done = workspace.run(args["command"], **timeout)
                if call.name == "run_tests":
                    session.tests = (session.counter, done.ok)
                else:
                    session.last_change = session.counter     # a command can change files
                return self._result(call, f"exit code {done.exit_code}\n{done.output}")
            session.report = args                             # report
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
    def run(self, agent, task, workspace):
        import llm

        workspace.start_task(task["id"])
        session = _Session()
        messages = [{"role": "user", "content": self.brief(agent, task)}]
        system = self.system_prompt(agent)
        summary, blocked, unreadable = "", "", 0

        for _ in range(self.max_turns):
            try:
                reply = llm.chat(messages, system=system, tools=TOOLS)
            except llm.ToolInputError:
                unreadable += 1                   # no tool_use to answer: send the turn again
                if unreadable > 2:
                    blocked = "the executor model kept sending tool input that couldn't be read"
                    break
                continue
            if reply.stop_reason == "refusal":
                blocked = "the executor model declined to work on this task"
                break
            messages.append({"role": "assistant", "content": reply.content})
            if reply.stop_reason == "max_tokens":
                # A cut-off turn may hold a half-written tool call: none of it runs.
                cut = [self._result(c, "Cut off at the output limit, so not run. Write smaller "
                                       "pieces.", error=True) for c in reply.tool_calls]
                messages.append({"role": "user", "content": cut or
                                 "You were cut off at the output limit. Continue in smaller pieces."})
                continue
            if not reply.tool_calls:
                summary = reply.text.strip()
                break
            messages.append({"role": "user", "content": [
                self._call(session, workspace, call) for call in reply.tool_calls]})
            if session.report:
                summary = session.report.get("summary", "").strip()
                blocked = (session.report.get("blocked_reason") or "").strip()
                break
        else:
            summary = "The work session ran out of turns before the task was finished."

        result = ExecResult(summary=summary or "Worked on the task.",
                            tests_passed=session.tests_passed(), blocked_reason=blocked)
        workspace.commit(f"{task['id']}: {result.summary}"[:200])
        result.files_changed = workspace.changed_files()
        if result.finished:
            workspace.publish(task["id"])
        return result


def hands_for(agent):
    """The executor and workspace for one agent, from GA_EXECUTOR:

        none   (default) no hands: the agents plan, talk and keep the board, as in the paper
        fake   FakeExecutor, no workspace: a dry run of the whole flow, no model, no files
        local  LocalExecutor in workspaces/<agent>/ (commands in Docker, see workspace.py)
    """
    import os

    choice = os.environ.get("GA_EXECUTOR", "none")
    if choice == "none":
        return None, None
    if choice == "fake":
        return FakeExecutor(), None
    if choice == "local":
        from workspace import Workspace
        return LocalExecutor(), Workspace(agent)
    raise ValueError(f"unknown GA_EXECUTOR {choice!r} (none, fake or local)")
