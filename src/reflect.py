from memory_stream import MemoryStream
from memory_retriever import MemoryRetriever

import ollama
import re

class Reflection:
    def __init__(self, agent):
        self.memory_stream = MemoryStream(agent["id"])
        self.memory_retriever = MemoryRetriever(agent["id"])

    def reflect(self):
        print("AGENT: REFLECTING")

        reflection_threshold = 38

        memories = self.memory_stream.get_memory_stream()

        recent_memories = memories[-20:]

        importance_sum = sum(memory.importance for memory in recent_memories)

        if importance_sum < reflection_threshold:
            print(f"Importance Sum of {importance_sum} does not meet the Reflection threshold of {reflection_threshold}")
            return

        questions = self.generate_reflection_questions(recent_memories)

        total_relevant_memories = []

        print("REFLECTION QUESTIONS")
        for question in questions:
            print(question)
            relevant_memories = self.retrieve(question)
            total_relevant_memories.extend(relevant_memories)

        unique_memories = {}

        for result in total_relevant_memories:
            memory = result["memory"]
            unique_memories[memory.id] = memory

        relevant_memories = list(unique_memories.values())

        insights = self.generate_reflection_insights(relevant_memories)

        for insight in insights:
            print(insight)
            self.memory_stream.add_reflection(description=insight, source_memory_ids=[memory.id for memory in relevant_memories])

        print("Reflection complete!")

    def generate_reflection_questions(self, recent_memories):
            print("Generating reflection questions")
            memory_text = "\n".join( [ memory.description for memory in recent_memories ] )
    
            prompt = f"""
                {memory_text}
    
                Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements? Return only the 3 questions."""
    
            response = ollama.generate(model="qwen3", prompt=prompt, think=False)
    
            text = response["response"]
    
            questions = []
    
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                line = re.sub(r"^\d+\.\s*", "", line)
                questions.append(line)
    
            return questions
    
    def generate_reflection_insights(self, relevant_memories):
        print("Generating insights")

        memory_text = "\n".join(
            f"{memory.id}. {memory.description}"
            for memory in relevant_memories
        )

        print(memory_text)

        prompt = f"""
    Statements about Adam Beck

    {memory_text}

    What 5 high-level insights can you infer from the above statements?
    (example format: insight (because of 1, 5, 3))
    """

        response = ollama.generate(
            model="qwen3",
            prompt=prompt,
            think=False
        )

        insights = []

        for line in response["response"].splitlines():
            line = line.strip()

            if not line:
                continue

            # Ignore intro text
            if line.lower().startswith("here are"):
                continue

            # Ignore evidence/explanation lines
            if line.startswith("*"):
                continue

            if line.startswith("-"):
                continue

            # Remove numbering
            line = re.sub(r"^\d+[\.\)]\s*", "", line)

            # Remove markdown bold
            line = line.replace("**", "")

            # Remove cited evidence
            line = re.sub(
                r"\s*\(because of.*?\)$",
                "",
                line,
                flags=re.IGNORECASE
            )

            line = line.strip()

            if not line:
                continue

            # Normalize pronouns
            if line.startswith("He is "):
                line = "Adam Beck is " + line[len("He is "):]

            elif line.startswith("He "):
                line = "Adam Beck " + line[len("He "):]

            elif line.startswith("His "):
                line = "Adam Beck's " + line[len("His "):]

            insights.append(line)

        return insights