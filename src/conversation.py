"""
conversation.py - turn-based dialogue between two agents (paper section 4.3.2)

Flow, per conversation:
    1. The initiator speaks, conditioned on their summary, status, relationship
       memories and the reaction that triggered the conversation.
    2. The listener perceives the line (it is stored in BOTH memory streams as
       an observation), retrieves memories about the speaker and the line, then
       either replies or ends the conversation.
    3. Repeat, alternating speakers, until someone ends it or MAX_TURNS.

Deviation from the paper: "decide whether to respond" and "generate the
response" are one LLM call (reply with SAY: <line> or END) to save a local-model
round trip per turn.
"""

import re

import llm
import log
import project_board
from llm import CTX

MAX_TURNS = 6          # total utterances, both speakers combined
MAX_LINE_TOKENS = 80


def _context(rt, other_name, last_line=None, limit=8):
    """Memories about the relationship, plus ones related to the last utterance."""
    name = rt.agent["name"]
    queries = [f"What is {name}'s relationship with {other_name}"]
    if last_line:
        queries.append(last_line)

    seen, lines = set(), []
    for q in queries:
        for result in rt.retriever.retrieve(q, n=5):
            desc = result["memory"].description
            if desc not in seen:
                seen.add(desc)
                lines.append(f"- {desc}")
    return "\n".join(lines[:limit])


def _remember(rt, text):
    rt.stream.add_observations([text])


def _clean(raw, speaker_name):
    """Strip 'SAY:', speaker-name prefixes and wrapping quotes."""
    line = raw.strip()
    line = re.sub(r"^(SAY|RESPONSE)\s*:\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(rf"^{re.escape(speaker_name)}\s*:\s*", "", line)
    line = line.strip().strip('"').strip()
    return line


def _board_context(speaker, listener, world, mentioned=""):
    """What the speaker knows from the project board that bears on this talk:
    their own situation, the listener's open tasks, and exactly where every
    task in play stands. Lets them talk about T-numbers instead of 'the
    integration issues', without turning "in progress" into "done"."""
    if not world or "project" not in world:
        return ""
    lines = project_board.status_for_agent(world, speaker.agent["id"])
    theirs = project_board.open_tasks_for(world, listener.agent["id"])
    other = listener.agent["name"]
    if theirs:
        lines.append(f"{other}'s open tasks: " + ", ".join(
            f"{t['id']} '{t['title']}' ({t['status'].replace('_', ' ')})" for t in theirs) + ".")
    body = "\n".join(f"    - {line}" for line in lines)

    ids = project_board.task_ids_in(mentioned) + [
        t["id"] for t in project_board.open_tasks_for(world, speaker.agent["id"]) + theirs]
    facts = "\n".join(f"    - {f}" for f in project_board.task_facts(world, ids))
    facts = f"""
    Where those tasks really stand on the board right now:
{facts}
""" if facts else ""
    return f"""
    What {speaker.agent['name']} knows from the project board:
{body}
{facts}
    Talking doesn't change the board. Nobody can start, approve, finish or move
    a task in a conversation: that only happens at the project board, a task is
    approved by a teammate only once it is in review, and a task that waits on
    another can't start until that one is done. So {speaker.agent['name']} only
    says a task is approved, done or unblocked if the board above says so, and
    talks about plans as plans ("I'll move T-1 to review once it works").
"""


def _generate_line(speaker, listener, current_time, status, history, intent, world=None):
    """Return the next utterance, or None if the speaker ends the conversation."""
    name = speaker.agent["name"]
    other = listener.agent["name"]

    last_line = history[-1][1] if history else None
    context = _context(speaker, other, last_line)

    if not history:
        # Initiator: speaks because of the reaction that triggered this.
        ask = (
            f"{name} decided to react: {intent}.\n"
            f"What would {name} say to {other}?\n"
            f"Reply with only the spoken line, one or two sentences."
        )
    else:
        transcript = "\n".join(f"{who}: {text}" for who, text in history)
        ask = (
            f"Here is the dialogue history:\n{transcript}\n\n"
            f"How would {name} respond to {other}? If {name} would naturally "
            f"end the conversation (goodbye, back to work, nothing left to say), "
            f"reply with exactly: END\n"
            f"Otherwise reply in exactly this format: SAY: <one or two sentences>"
        )

    prompt = f"""
    {speaker.planner.generate_agent_summary(speaker.agent)}

    It is {current_time}.
    {name}'s status: {status}

    Summary of relevant context from {name}'s memory:
    {context}
{_board_context(speaker, listener, world, intent + " " + " ".join(t for _, t in history))}
    {name} is talking with {other}. Be specific: name the tasks (T-1, T-2, ...)
    and the concrete next step when it helps, and don't repeat what was already said.
    Only talk about work as it really stands: {name} must not claim to have
    built, reviewed or approved anything that isn't there, and a task marked todo
    has not been started. If a memory (even something {name} said earlier)
    disagrees with the board, the board is right.
    {ask}
    """

    raw = llm.generate(
        prompt, options={**CTX, "num_predict": MAX_LINE_TOKENS, "temperature": 0.7}
    )

    if history and raw.strip().upper().startswith("END"):
        return None

    line = _clean(raw, name)
    return line or None


def run_conversation(initiator, partner, current_time, reaction, world=None):
    """
    Run a full conversation. Both arguments are AgentRuntime objects.
    Returns the list of (speaker_name, line) in order.
    """
    history = []
    speaker, listener = initiator, partner

    for _ in range(MAX_TURNS):
        line = _generate_line(
            speaker, listener, current_time,
            status=speaker.status, history=history, intent=reaction, world=world,
        )

        if line is None:
            log.debug(f"{speaker.tag} ends the conversation")
            break

        sname, lname = speaker.agent["name"], listener.agent["name"]

        # Reject a line this speaker already said earlier (echo loops)
        said = {t.lower() for who, t in history if who == sname}
        if line.lower() in said:
            log.debug(f"{speaker.tag} repeated an earlier line, ending")
            break

        history.append((sname, line))
        log.dialogue(speaker.tag, line)

        # Both agents experience the utterance as an observation.
        memory_text = f'{sname} said to {lname}: "{line}"'
        _remember(speaker, memory_text)
        _remember(listener, memory_text)

        speaker, listener = listener, speaker

    return history