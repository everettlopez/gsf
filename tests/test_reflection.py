from pathlib import Path
import sys


sys.path.append(
    str(Path(__file__).resolve().parent.parent / "src")
)

from reflect import Reflection

test_reflection = Reflection()
