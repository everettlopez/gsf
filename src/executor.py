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
