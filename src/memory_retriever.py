from memory_stream import MemoryStream
from datetime import datetime

import numpy as np
import ollama
import re

class MemoryRetriever:
    def __init__(self):
        self.memory_stream = MemoryStream()

        # Retrieval weights from the paper
        self.alpha_recency = 0.25
        self.alpha_importance = 1.0
        self.alpha_relevance = 3.0

        # Recency decay factor from the paper
        self.recency_decay = 0.995

    def retrieve(self, query, n=5):
        memories = self.memory_stream.get_memory_stream()
        if not memories:
            return []

        query_embedding = self.generate_query_embedding(query)

        scored_results = []

        # Step 1: Calculate the raw component scores
        for memory in memories:
            relevance = self.calculate_relevance_score(
                query_embedding,
                memory.embedding
            )

            recency = self.calculate_recency_score(memory)
            importance = memory.importance

            scored_results.append({
                "memory": memory,
                "relevance": relevance,
                "recency": recency,
                "importance": importance
            })

        # Step 2: Collect each score category
        relevance_scores = np.array(
            [
                result["relevance"]
                for result in scored_results
            ],
            dtype=float
        )

        recency_scores = np.array(
            [
                result["recency"]
                for result in scored_results
            ],
            dtype=float
        )

        importance_scores = np.array(
            [
                result["importance"]
                for result in scored_results
            ],
            dtype=float
        )

        # Step 3: Normalize each category to the range [0, 1]
        normalized_relevance_scores = self.normalize_scores(
            relevance_scores
        )

        normalized_recency_scores = self.normalize_scores(
            recency_scores
        )

        normalized_importance_scores = self.normalize_scores(
            importance_scores
        )

        # Step 4: Calculate the weighted retrieval score
        for index, result in enumerate(scored_results):
            normalized_relevance = normalized_relevance_scores[index]
            normalized_recency = normalized_recency_scores[index]
            normalized_importance = normalized_importance_scores[index]

            relevance_component = (
                self.alpha_relevance * normalized_relevance
            )

            recency_component = (
                self.alpha_recency * normalized_recency
            )

            importance_component = (
                self.alpha_importance * normalized_importance
            )

            retrieval_score = (
                relevance_component
                + recency_component
                + importance_component
            )

            result["normalized_relevance"] = normalized_relevance
            result["normalized_recency"] = normalized_recency
            result["normalized_importance"] = normalized_importance

            result["relevance_component"] = relevance_component
            result["recency_component"] = recency_component
            result["importance_component"] = importance_component

            result["retrieval_score"] = retrieval_score

        # Step 5: Rank from highest retrieval score to lowest
        scored_results.sort(
            key=lambda result: result["retrieval_score"],
            reverse=True
        )

        top_results = scored_results[:n]

        now = datetime.now().isoformat()

        for result in top_results:
            result["memory"].last_accessed_at = now

        self.memory_stream.save(memories)

        # Step 6: Return only the top N results
        return top_results

    def calculate_recency_score(self, memory):
        last_accessed = datetime.fromisoformat(
            memory.last_accessed_at
        )

        now = datetime.now()

        hours_since_access = (
            now - last_accessed
        ).total_seconds() / 3600

        return self.recency_decay ** hours_since_access

    def generate_query_embedding(self, query):
        response = ollama.embed(
            model="nomic-embed-text",
            input=query
        )

        return response["embeddings"][0]

    def calculate_relevance_score(
        self,
        query_embedding,
        memory_embedding
    ):
        query_vector = np.array(
            query_embedding,
            dtype=float
        )

        memory_vector = np.array(
            memory_embedding,
            dtype=float
        )

        query_norm = np.linalg.norm(query_vector)
        memory_norm = np.linalg.norm(memory_vector)

        if query_norm == 0 or memory_norm == 0:
            return 0.0

        return float(
            np.dot(query_vector, memory_vector)
            / (query_norm * memory_norm)
        )

    def normalize_scores(self, scores):
        minimum = scores.min()
        maximum = scores.max()

        if maximum == minimum:
            return np.ones_like(
                scores,
                dtype=float
            )

        return (
            (scores - minimum)
            / (maximum - minimum)
        )

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
            [
                f"{memory.id}. {memory.description}"
                for i, memory in enumerate(relevant_memories)
            ]
        )

        print(memory_text)

        prompt = f"""
        Statements about Adam Beck
        {memory_text}
        What 5 high-level insights can you infer from the above statements? (example format: insight (because of 1, 5, 3))"""

        response = ollama.generate(model="qwen3", prompt=prompt, think=False)

        insights = []

        for line in response["response"].splitlines():
            line = line.strip()

            if not line:
                continue

            if line.lower().startswith("here are"):
                continue

            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"\*\*", "", line)
            line = re.sub(r"\s*\(because.*?\)$", "", line)

            # Normalize pronouns
            if line.startswith("He is "):
                line = "Adam Beck is " + line[len("He is "):]

            elif line.startswith("He "):
                line = "Adam Beck " + line[len("He "):]

            elif line.startswith("His "):
                line = "Adam Beck's " + line[len("His "):]

            insights.append(line)

        return insights