"""Shared fixtures for minifier tests."""
import sys
from pathlib import Path

# Make the repo root importable regardless of how pytest is invoked
REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GOLDEN_DIR = Path(__file__).parent / "golden"
PYTHON_GOLDEN = GOLDEN_DIR / "python"
JS_GOLDEN = GOLDEN_DIR / "javascript"
