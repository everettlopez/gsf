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
    # The example uses a real task, never an invented "T-1": with an empty
    # board the model copied the example ids into the conversation.
    mine = project_board.open_tasks_for(world, speaker.agent["id"])
    example = (f"I'll move {mine[0]['id']} to review once it works" if mine
               else "I'll add the API work to the board")
    # Whose tasks are whose. Without it Adam offered to "split up" Eve's
    # frontend tasks and promised to get her T-2 ready for review.
    owners = "\n".join(f"    - {line}"
                       for line in project_board.ownership_lines(world, speaker.agent["id"]))
    owners = f"""
    Who owns what:
{owners}
    So {speaker.agent['name']} talks about their own tasks as theirs to do, and about
    {other}'s tasks as {other}'s: asking how they're going, reviewing them, or
    offering to work on one together, which {other} can accept or turn down.
""" if owners else ""
    return f"""
    What {speaker.agent['name']} knows from the project board:
{body}
{facts}{owners}
    {project_board.task_id_line(world)}
    Talking doesn't change the board. Nobody can start, approve, finish or move
    a task in a conversation: that only happens at the project board, a task is
    approved by a teammate only once it is in review, and a task that waits on
    another can't start until that one is done. So {speaker.agent['name']} only
    says a task is approved, done or unblocked if the board above says so, and
    talks about plans as plans ("{example}").
"""


def _plans_with(speaker, listener, current_time, limit=2):
    """What the speaker has planned with the listener later today. Plans are
    memories the paper's agents bring into conversation; without them each
    agent kept its own "collaborate with ..." block that the other never knew
    about (Adam's 11:15 fell in Eve's lunch). Seeing them, the two can confirm a
    time and place, which then goes into both plans."""
    parse = speaker.planner._parse_time
    now = parse(current_time)
    chunks = getattr(speaker, "broad_chunks", None) or []
    upcoming = [c for c in chunks
                if c.get("target_agent") == listener.agent["id"]
                and parse(c.get("start_time")) is not None and now is not None
                and parse(c.get("start_time")) > now]
    if not upcoming:
        return ""
    name, other = speaker.agent["name"], listener.agent["name"]
    lines = "\n".join(f"    - {c['activity']}" + (" (agreed with " + other + ")" if c.get("agreed") else "")
                      for c in upcoming[:limit])
    return f"""
    {name}'s plan for later today with {other}:
{lines}
    If {name} wants to meet {other} later, they can suggest a time and place, and
    it only happens if {other} agrees.
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
{_board_context(speaker, listener, world, intent + " " + " ".join(t for _, t in history))}{_plans_with(speaker, listener, current_time)}
    {name} is talking with {other}. Be specific: name the tasks on the board by
    their ids and the concrete next step when it helps, and don't repeat what was
    already said.
    Only talk about work as it really stands: {name} must not claim to have
    built, reviewed or approved anything that isn't there, and a task marked todo
    has not been started. If a memory (even something {name} said earlier)
    disagrees with the board, the board is right. Nothing gets done while they
    talk: anything {name} or {other} says they will do is still not done when
    the conversation ends, so {name} never says it is finished.
    {ask}
    """

    raw = llm.generate(
        prompt, options={**CTX, "num_predict": MAX_LINE_TOKENS, "temperature": 0.7}
    )

    if history and raw.strip().upper().startswith("END"):
        return None

    line = _clean(raw, name)
    if line and world and "project" in world:
        made_up = project_board.unknown_task_ids(world, line)
        if made_up:
            # Once more, told plainly; whatever is still made up is cut out, so a
            # task that doesn't exist never lands in either agent's memory.
            log.debug(f"{speaker.tag} named tasks that don't exist ({', '.join(made_up)}), retrying")
            fix = (f"\n    Careful: {', '.join(made_up)} "
                   f"{'is' if len(made_up) == 1 else 'are'} not on the project board. "
                   f"{project_board.task_id_line(world)}\n    ")
            raw = llm.generate(prompt + fix, options={**CTX, "num_predict": MAX_LINE_TOKENS,
                                                       "temperature": 0.7})
            if history and raw.strip().upper().startswith("END"):
                return None
            line = project_board.strip_unknown_task_ids(world, _clean(raw, name))
    return line or None


REPEAT_SIMILARITY = 0.5     # share of words two lines have in common to count as a repeat


def _words(text):
    return set(re.findall(r"[a-z0-9-]{3,}", text.lower()))


def _similar(a, b, threshold=REPEAT_SIMILARITY):
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return a.strip().lower() == b.strip().lower()
    return len(wa & wb) / len(wa | wb) >= threshold


_PROMISE = re.compile(r"\b(?:i'?ll|i will|i'?m going to|let me)\b", re.IGNORECASE)
_DONE_CLAIM = re.compile(r"\b(?:i'?ve|i have|i just)\s+(?:already\s+|just\s+)?\w+ed\b[^.!?]*",
                         re.IGNORECASE)


def claims_promised_work(line, earlier):
    """Does `line` say the speaker already did what they promised earlier in
    this same conversation? Nothing happens while two agents talk, so it is
    made up: Adam said "I'll start by outlining the API endpoints and database
    structure" and, one line later, "I've outlined the API endpoints and
    database structure", which Eve then planned her morning around."""
    claim = _DONE_CLAIM.search(line)
    if not claim:
        return False
    said = _words(claim.group(0)) - {"i've", "have", "just", "already", "the", "and"}
    for promise in earlier:
        if _PROMISE.search(promise) and said:
            if len(said & _words(promise)) / len(said) >= 0.5:
                return True
    return False


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

        # A line this speaker already said, or nearly said, ends the talk. Turns
        # 3-6 of one conversation restated "I'll keep working on T-2 and wait
        # for T-1" in new words, for 60 more seconds of LLM calls.
        if any(_similar(line, t) for who, t in history if who == sname):
            log.debug(f"{speaker.tag} repeated themselves, ending")
            break

        if claims_promised_work(line, [t for who, t in history if who == sname]):
            # Dropped, not remembered: neither agent should believe it.
            log.debug(f"{speaker.tag} claimed to have already done what they just promised, ending: {line}")
            break

        history.append((sname, line))
        log.dialogue(speaker.tag, line)

        # Both agents experience the utterance as an observation.
        memory_text = f'{sname} said to {lname}: "{line}"'
        _remember(speaker, memory_text)
        _remember(listener, memory_text)

        speaker, listener = listener, speaker

    return history