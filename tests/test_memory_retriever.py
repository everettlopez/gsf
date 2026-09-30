from pathlib import Path
import sys


sys.path.append(
    str(Path(__file__).resolve().parent.parent / "src")
)

from memory_retriever import MemoryRetriever


retriever = MemoryRetriever()

results = retriever.retrieve(
    "What task is Agent currently working on?"
)

for result in results:
    print(
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