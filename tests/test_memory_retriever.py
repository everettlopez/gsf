from pathlib import Path
import sys


sys.path.append(
    str(Path(__file__).resolve().parent.parent / "src")
)

from memory_retriever import MemoryRetriever


retriever = MemoryRetriever()

# query = "Python"
# memory = "Adam Beck primarily develops software using Python and TypeScript."

# similarity = retriever.calculate_relevance_score(retriever.generate_query_embedding(query), retriever.generate_query_embedding(memory))

# print(similarity)

# retriever.reflect()

results = retriever.retrieve(
    "What does Adam Beck enjoy?"
)

for result in results:
    print(
        result["memory"].id,
        result["memory"].memory_type,
        result["memory"].description,
        "\nRetrieval score:",
        result["retrieval_score"],
        "\nRelevance:",
        result["relevance"],
        "\nRecency:",
        result["recency"],
        "\nImportance:",
        result["importance"],
        "\n"
    )