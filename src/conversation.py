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

import ollama

from plan import CTX

MAX_TURNS = 6          # total utterances, both speakers combined
MAX_LINE_TOKENS = 80


def _summary(rt):
    """Agent summary, cached. Planner already has _summary_cache but
    generate_agent_summary() never fills it, so we do it here."""
    cache = rt.planner._summary_cache
    key = rt.agent["id"]
    if key not in cache:
        cache[key] = rt.planner.generate_agent_summary(rt.agent)
    return cache[key]


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
    # ASSUMPTION: matches the add_observations() listed in main.py's docstring.
    # Adjust if your MemoryStream signature differs.
    rt.stream.add_observations([text])


def _clean(raw, speaker_name):
    """Strip 'SAY:', speaker-name prefixes and wrapping quotes."""
    line = raw.strip()
    line = re.sub(r"^(SAY|RESPONSE)\s*:\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(rf"^{re.escape(speaker_name)}\s*:\s*", "", line)
    line = line.strip().strip('"').strip()
    return line


def _generate_line(speaker, listener, current_time, status, history, intent):
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
    {_summary(speaker)}

    It is {current_time}.
    {name}'s status: {status}

    Summary of relevant context from {name}'s memory:
    {context}

    {name} is talking with {other}.
    {ask}
    """

    raw = ollama.generate(
        model="qwen3",
        prompt=prompt,
        think=False,
        options={**CTX, "num_predict": MAX_LINE_TOKENS, "temperature": 0.7},
    )["response"]

    if history and raw.strip().upper().startswith("END"):
        return None

    line = _clean(raw, name)
    return line or None


def run_conversation(initiator, partner, current_time, reaction):
    """
    Run a full conversation. Both arguments are AgentRuntime objects.
    Returns the list of (speaker_name, line) in order.
    """
    history = []
    speaker, listener = initiator, partner

    for _ in range(MAX_TURNS):
        line = _generate_line(
            speaker, listener, current_time,
            status=speaker.status, history=history, intent=reaction,
        )

        if line is None:
            print(f"[{speaker.tag}] ENDS CONVERSATION")
            break

        sname, lname = speaker.agent["name"], listener.agent["name"]

        # Reject a line this speaker already said earlier (echo loops)
        said = {t.lower() for who, t in history if who == sname}
        if line.lower() in said:
            print(f"[{speaker.tag}] repeated an earlier line, ending")
            break

        history.append((sname, line))
        print(f'[{speaker.tag}] SAYS: "{line}"')

        # Both agents experience the utterance as an observation.
        memory_text = f'{sname} said to {lname}: "{line}"'
        _remember(speaker, memory_text)
        _remember(listener, memory_text)

        speaker, listener = listener, speaker

    return history