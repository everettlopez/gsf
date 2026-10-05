from memory_stream import MemoryStream
from datetime import datetime

import numpy as np
import ollama
import re

class MemoryRetriever:
    def __init__(self, agent_name: str, memory_stream=None):
        self.memory_stream = memory_stream or MemoryStream(agent_name)

        self.alpha_recency = 0.5
        self.alpha_importance = 1.0
        self.alpha_relevance = 3.0
        self.recency_decay = 0.995

        self._normed_matrix = None
        self._matrix_count = 0
        self._query_cache = {}

    def _get_matrix(self, memories):
        if self._normed_matrix is not None and self._matrix_count == len(memories):
            return self._normed_matrix

        dim = next((len(m.embedding) for m in memories if m.embedding), 0)
        matrix = np.zeros((len(memories), dim), dtype=float)

        for i, memory in enumerate(memories):
            if len(memory.embedding) == dim:
                matrix[i] = memory.embedding

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0   # zero vectors stay zero -> relevance 0

        self._normed_matrix = matrix / norms
        self._matrix_count = len(memories)
        return self._normed_matrix

    def generate_query_embedding(self, query):
        cached = self._query_cache.get(query)
        if cached is not None:
            return cached

        response = ollama.embed(model="nomic-embed-text", input=query)
        embedding = np.array(response["embeddings"][0], dtype=float)

        if len(self._query_cache) >= 256:
            self._query_cache.clear()
        self._query_cache[query] = embedding
        return embedding

    def retrieve(self, query, n=15):
        memories = self.memory_stream.get_memory_stream()
        if not memories:
            return []

        matrix = self._get_matrix(memories)

        query_vec = self.generate_query_embedding(query)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0 or matrix.shape[1] != len(query_vec):
            relevance = np.zeros(len(memories))
        else:
            relevance = matrix @ (query_vec / query_norm)

        now = datetime.now()
        hours = np.array([
            (now - datetime.fromisoformat(m.last_accessed_at)).total_seconds() / 3600
            for m in memories
        ])
        recency = self.recency_decay ** hours
        importance = np.array([m.importance for m in memories], dtype=float)

        norm_rel = self.normalize_scores(relevance)
        norm_rec = self.normalize_scores(recency)
        norm_imp = self.normalize_scores(importance)

        total = (
            self.alpha_relevance * norm_rel
            + self.alpha_recency * norm_rec
            + self.alpha_importance * norm_imp
        )

        k = min(n, len(memories))
        top = np.argpartition(-total, k - 1)[:k]
        top = top[np.argsort(-total[top])]

        now_iso = now.isoformat()
        results = []
        for i in top:
            memories[i].last_accessed_at = now_iso
            results.append({
                "memory": memories[i],
                "relevance": float(relevance[i]),
                "recency": float(recency[i]),
                "importance": float(importance[i]),
                "normalized_relevance": float(norm_rel[i]),
                "normalized_recency": float(norm_rec[i]),
                "normalized_importance": float(norm_imp[i]),
                "relevance_component": float(self.alpha_relevance * norm_rel[i]),
                "recency_component": float(self.alpha_recency * norm_rec[i]),
                "importance_component": float(self.alpha_importance * norm_imp[i]),
                "retrieval_score": float(total[i]),
            })

        self.memory_stream.mark_dirty()
        return results

    def calculate_recency_score(self, memory):
        last_accessed = datetime.fromisoformat(
            memory.last_accessed_at
        )

        now = datetime.now()

        hours_since_access = (
            now - last_accessed
        ).total_seconds() / 3600

        return self.recency_decay ** hours_since_access

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