"""
reflect.py - reflection (paper section 4.2)

When it runs (see main.py, REFLECTION_TIMES) an agent:
    1. takes its most recent unreflected observations (paper: the last 100),
    2. asks the LLM for the 3 most salient high-level questions about them,
    3. retrieves memories relevant to each question,
    4. asks for 5 insights, each citing the memories behind it, and
    5. stores every insight in the memory stream as a Reflection, linked to the
       memories it cites. Reflections are retrievable like any other memory, so
       they feed back into plans, reactions and conversations.
"""

import re

import llm
import log

NUM_QUESTIONS = 3
NUM_INSIGHTS = 5
RECENT_WINDOW = 100         # paper: reflect over the 100 most recent records
MIN_MEMORIES = 5            # not worth an LLM round trip below this
MEMORIES_PER_QUESTION = 10


# ---------------------------------------------------------------------------
# Parsing (pure functions, unit-tested)
# ---------------------------------------------------------------------------
_LEADING_LIST_MARKER = re.compile(r"^(?:\d+[.)]|[-*•])\s*")
_CITATION = re.compile(r"\s*\((?:because of|from|based on)\s*([^)]*)\)\s*\.?$", re.IGNORECASE)


def _clean_line(line):
    line = line.strip().replace("**", "")
    return _LEADING_LIST_MARKER.sub("", line).strip()


def parse_questions(text, limit=NUM_QUESTIONS):
    questions = []
    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line or line.lower().startswith("here are") or line.endswith(":"):
            continue
        questions.append(line)
    return questions[:limit]


def parse_insights(text, valid_ids, agent_name, limit=NUM_INSIGHTS):
    """
    Parse "insight (because of 1, 5, 3)" lines into (insight, [cited ids]).
    Cited ids that aren't in `valid_ids` are dropped; if nothing valid is cited,
    the insight is linked to every memory it was generated from.
    """
    valid_ids = set(valid_ids)
    insights = []

    for raw in text.splitlines():
        line = raw.strip()
        # Sub-bullets under an insight are evidence/explanations, not insights.
        if not line or line.startswith(("*", "-")) or line.lower().startswith("here are"):
            continue

        line = _clean_line(line)

        cited = []
        match = _CITATION.search(line)
        if match:
            cited = [int(n) for n in re.findall(r"\d+", match.group(1))]
            line = line[:match.start()].strip()

        if not line:
            continue

        line = _normalize_pronoun(line, agent_name)
        sources = [i for i in cited if i in valid_ids] or sorted(valid_ids)
        insights.append((line, sources))

    return insights[:limit]


def _normalize_pronoun(line, name):
    """Reflections are stored as third-person statements about the agent."""
    for pronoun, replacement in (("He is ", f"{name} is "), ("She is ", f"{name} is "),
                                 ("He ", f"{name} "), ("She ", f"{name} "),
                                 ("His ", f"{name}'s "), ("Her ", f"{name}'s ")):
        if line.startswith(pronoun):
            return replacement + line[len(pronoun):]
    return line


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------
class Reflector:
    def __init__(self, agent, memory_stream, memory_retriever):
        self.agent = agent
        self.memory_stream = memory_stream
        self.memory_retriever = memory_retriever

    @property
    def name(self):
        return self.agent["name"]

    def reflect(self):
        """Run one reflection pass. Returns the list of new Reflection memories."""
        recent = self.memory_stream.unreflected(RECENT_WINDOW)

        if len(recent) < MIN_MEMORIES:
            log.debug(f"{self.name}: only {len(recent)} new memories, skipping reflection")
            return []

        questions = self.generate_questions(recent)
        log.debug(f"{self.name} reflection questions: {questions}")

        relevant = {}
        for question in questions:
            results = self.memory_retriever.retrieve(question, n=MEMORIES_PER_QUESTION)
            for result in results:
                relevant[result["memory"].id] = result["memory"]

        if not relevant:
            return []

        insights = self.generate_insights(list(relevant.values()))

        return [
            self.memory_stream.add_reflection(description=text, source_memory_ids=sources)
            for text, sources in insights
        ]

    def generate_questions(self, recent_memories):
        memory_text = "\n".join(m.description for m in recent_memories)

        prompt = f"""
{memory_text}

Given only the information above, what are {NUM_QUESTIONS} most salient high-level questions we can answer about the subjects in the statements? Return only the {NUM_QUESTIONS} questions as a numbered list."""

        return parse_questions(llm.generate(prompt, options=llm.CTX))

    def generate_insights(self, memories):
        memory_text = "\n".join(f"{m.id}. {m.description}" for m in memories)

        prompt = f"""
Statements about {self.name}

{memory_text}

What {NUM_INSIGHTS} high-level insights can you infer from the above statements?
(example format: insight (because of 1, 5, 3))
"""
        raw = llm.generate(prompt, options=llm.CTX)
        return parse_insights(raw, [m.id for m in memories], self.name)
